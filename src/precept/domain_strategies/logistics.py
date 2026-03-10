"""
Logistics Domain Strategy for PRECEPT.

Handles shipping/port black swan scenarios.

Black Swan Types (from black_swan_gen.py):
- Port_Closure: Agent tries blocked ports
- Customs_Delay: Unexpected customs inspection delays
- Carrier_Failure: Shipping carrier goes offline

🚨 CRITICAL: This strategy does NOT have hardcoded error->solution mappings.

What this strategy KNOWS (configuration, not learning):
- What ports exist (ORIGIN_PORTS, DESTINATION_PORTS)
- How to parse tasks into structured format
- How to call MCP tools

What this strategy does NOT KNOW (must be learned):
- Which ports are blocked
- Which alternatives work for which errors
- What error codes mean

═══════════════════════════════════════════════════════════════════════════════
NOTE: TIER 1 (Programmatic Rule Lookup) has been DISABLED.
═══════════════════════════════════════════════════════════════════════════════
Learning now happens exclusively through Tier 2 (LLM reasoning):
1. LLM receives learned rules as context
2. LLM reasons about which solution to apply
3. LLM suggestion is passed via `preferred_solution` parameter
4. `execute_action` applies the LLM's suggestion

The `apply_learned_rules` method is preserved but NOT CALLED from precept_agent.py.
Internal learning caches are maintained for potential future use.
"""

import re
from typing import Any, Dict, List, Tuple

from ..config import LogisticsConfig
from ..rule_parser import DynamicRuleParser
from .base import (
    ActionResult,
    BaselineDomainStrategy,
    BlackSwanCategory,
    DomainStrategy,
    ParsedTask,
)

# AutoGen imports (optional)
try:
    from autogen_core.tools import FunctionTool

    AUTOGEN_AVAILABLE = True
except ImportError:
    AUTOGEN_AVAILABLE = False
    FunctionTool = None


class LogisticsDomainStrategy(DomainStrategy):
    """
    Logistics domain strategy for shipping/port black swan scenarios.

    🚨 CRITICAL: This strategy does NOT have hardcoded error->solution mappings.

    What this strategy KNOWS (configuration, not learning):
    - What ports exist (ORIGIN_PORTS, DESTINATION_PORTS)
    - How to parse tasks into structured format
    - How to call MCP tools

    What this strategy does NOT KNOW (must be learned):
    - Which ports are blocked
    - Which alternatives work for which errors
    - What error codes mean

    ═══════════════════════════════════════════════════════════════════════════
    TIER 2 ONLY (LLM Reasoning):
    ═══════════════════════════════════════════════════════════════════════════
    With Tier 1 disabled, learning happens via LLM reasoning:
    1. LLM receives learned rules + task context
    2. LLM suggests solution via `preferred_solution` parameter
    3. `execute_action` applies LLM's suggestion with priority
    """

    # ═══════════════════════════════════════════════════════════════════════════
    # CONFIGURATION - Import from single source of truth
    # ═══════════════════════════════════════════════════════════════════════════
    # Known ports (vocabulary, NOT which are blocked!)
    ORIGIN_PORTS = LogisticsConfig.ORIGIN_PORTS
    DESTINATION_PORTS = LogisticsConfig.DESTINATION_PORTS
    DOCUMENTATION_TYPES = LogisticsConfig.DOCUMENTATION_TYPES

    # NO ERROR_PATTERNS here - that would be cheating!
    # The agent must LEARN which ports are blocked.

    def __init__(self, max_retries: int = None):
        """
        Initialize the logistics domain strategy.

        Args:
            max_retries: Maximum number of retries allowed. If None, uses DEFAULT_MAX_RETRIES.
                        - 1 = near first-try only (1 initial + 1 retry = 2 attempts)
                        - 2 = balanced (1 initial + 2 retries = 3 attempts) [default]
                        - 4 = lenient (1 initial + 4 retries = 5 attempts)
        """
        super().__init__(max_retries=max_retries)

        # Dynamic rule parser - only knows the vocabulary (port names)
        # Does NOT know which are blocked or which are alternatives
        self.rule_parser = DynamicRuleParser(
            known_entities=self.ORIGIN_PORTS + self.DOCUMENTATION_TYPES
        )

        # Runtime learned knowledge (empty at start!)
        self._learned_alternatives: Dict[str, str] = {}

        # KEY LEARNING: Route → Working origin port mapping
        # Real-world: Port closures, customs blocks require rerouting
        # Example: "rotterdam→boston" → blocked, use "antwerp→boston"
        self._learned_route_ports: Dict[
            str, str
        ] = {}  # "destination" → "working_origin"

        # CUSTOMS LEARNING: Destination → Required documentation
        # Real-world: Different destinations have different customs requirements
        self._learned_customs_docs: Dict[
            str, str
        ] = {}  # "destination" → "documentation_type"

    @property
    def category(self) -> BlackSwanCategory:
        return BlackSwanCategory.LOGISTICS

    @property
    def domain_name(self) -> str:
        return "logistics"

    def get_system_prompt(self, learned_rules: List[str] = None) -> str:
        base_prompt = f"""You are a logistics routing agent with PRECEPT+COMPASS learning capabilities.

AVAILABLE ACTIONS:
- book_shipment(origin, destination): Book a shipment
- check_port(port): Check port availability

AVAILABLE ORIGIN PORTS: {", ".join(self.ORIGIN_PORTS)}
AVAILABLE DESTINATIONS: {", ".join(self.DESTINATION_PORTS)}

YOUR WORKFLOW:
1. Call get_learned_rules() first
2. Apply any learned rules proactively
3. Execute the booking
4. On failure, record_error and try alternatives
5. Learn from the experience

COMPASS ADVANTAGES:
- ML-based complexity analysis
- Smart rollout allocation
- Early stopping when confident"""

        if learned_rules:
            rules_section = "\n\n═══ LEARNED RULES (Apply these FIRST!) ═══\n"
            for i, rule in enumerate(learned_rules, 1):
                rules_section += f"{i}. {rule}\n"
            base_prompt = rules_section + "\n" + base_prompt

        return base_prompt

    def get_available_actions(self) -> List[str]:
        return ["book_shipment", "check_port", "clear_customs"]

    def get_available_entities(self) -> List[str]:
        return self.ORIGIN_PORTS + self.DESTINATION_PORTS

    def get_options_for_task(self, parsed_task: ParsedTask) -> List[str]:
        """
        Return options relevant to this specific task type.

        - Customs tasks → documentation types
        - Booking tasks → origin ports (ALL ports - no restrictions)

        NOTE: We do NOT restrict options for multi-condition scenarios.
        All agents (PRECEPT, ExpeL, Full Reflexion) should have equal
        access to the same set of possible solutions. Restricting options
        would give PRECEPT an unfair advantage by reducing its search space.
        """
        is_customs = parsed_task.parameters.get("is_customs", False)
        if is_customs:
            return self.DOCUMENTATION_TYPES.copy()
        else:
            # Return ALL origin ports - same options for all agents
            return self.ORIGIN_PORTS.copy()

    def parse_task(self, task: str) -> ParsedTask:
        """Parse logistics task into components."""
        import re

        task_lower = (
            task.lower()
            .replace("new york", "new_york")
            .replace("los angeles", "los_angeles")
        )

        # ═══════════════════════════════════════════════════════════════════
        # MULTI-CONDITION EXTRACTION: Parse [Conditions: X + Y + Z]
        # ═══════════════════════════════════════════════════════════════════
        condition_key = None
        conditions = []
        condition_match = re.search(r"\[Conditions:\s*([^\]]+)\]", task, re.IGNORECASE)
        if condition_match:
            # Extract conditions like "R-482 + P-220" -> ["R-482", "P-220"]
            cond_str = condition_match.group(1)
            conditions = [c.strip() for c in cond_str.split("+")]
            # Generate deterministic key (sorted, joined with +)
            condition_key = "+".join(sorted(conditions))

        # Detect if this is a customs task
        is_customs_task = any(
            keyword in task_lower
            for keyword in [
                "customs",
                "clearance",
                "documentation",
                "import",
                "declaration",
            ]
        )

        # Detect destination
        destination = None
        for dest in self.DESTINATION_PORTS:
            dest_check = dest.replace("_", " ")
            if dest in task_lower or dest_check in task_lower:
                destination = dest
                break

        # For customs tasks, also check city names
        if not destination and is_customs_task:
            city_map = {
                "new york": "new_york",
                "los angeles": "los_angeles",
                "chicago": "chicago",
                "miami": "miami",
                "seattle": "seattle",
                "boston": "boston",
            }
            for city, dest_key in city_map.items():
                if city in task_lower:
                    destination = dest_key
                    break

        # Detect origin (default to rotterdam from task)
        origin = "rotterdam"  # Default from most tasks
        for port in self.ORIGIN_PORTS:
            if port in task_lower:
                origin = port
                break

        # Determine if US destination
        is_us = destination in [
            "boston",
            "new_york",
            "los_angeles",
            "chicago",
            "miami",
            "seattle",
        ]

        # Determine action type
        if is_customs_task:
            action = "clear_customs"
            entity = "customs"
        else:
            action = "book_shipment"
            entity = "shipment"

        return ParsedTask(
            raw_task=task,
            action=action,
            entity=entity,
            source=origin,
            target=destination,
            parameters={
                "is_us_destination": is_us,
                "is_customs": is_customs_task,
                "documentation": "standard",  # Default, will be updated by learning
                "condition_key": condition_key,  # Multi-condition key for rule storage
                "conditions": conditions,  # Individual conditions
            },
        )

    def apply_learned_rules(
        self,
        parsed_task: ParsedTask,
        rules: List[str],
    ) -> Tuple[ParsedTask, bool, str]:
        """
        Apply learned rules programmatically (TIER 1 - CURRENTLY DISABLED).

        ⚠️ NOTE: This method is NOT currently called from precept_agent.py.
        Tier 1 has been disabled in favor of Tier 2 (LLM reasoning only).
        This method is preserved for potential future re-enablement.

        When enabled, this method:
        - Applies learned rules proactively before execution
        - Uses local cache for fast lookups
        - Parses server rules dynamically

        IMPORTANT: Rules are TASK-TYPE SPECIFIC!
        - Customs tasks only check customs rules (documentation)
        - Booking tasks only check port rules (origins)
        This prevents cross-contamination of learned patterns.
        """
        destination = parsed_task.target
        origin = parsed_task.source
        is_us = parsed_task.parameters.get("is_us_destination", False)
        is_customs = parsed_task.parameters.get("is_customs", False)

        # ═══════════════════════════════════════════════════════════════════
        # CUSTOMS TASKS: Only check customs documentation rules
        # ═══════════════════════════════════════════════════════════════════
        if is_customs:
            if destination and destination in self._learned_customs_docs:
                doc_type = self._learned_customs_docs[destination]
                parsed_task.parameters["documentation"] = doc_type
                return parsed_task, True, f"Learned:customs:{destination}→{doc_type}"
            # For customs: Don't fall through to port rules!
            return parsed_task, False, "Exploration:customs"

        # ═══════════════════════════════════════════════════════════════════
        # BOOKING TASKS: Check port routing rules
        # ═══════════════════════════════════════════════════════════════════
        # Check local learned cache first (route → working_port)
        route_key = f"{origin}→{destination}"
        if route_key in self._learned_route_ports:
            working_origin = self._learned_route_ports[route_key]
            parsed_task.source = working_origin
            return parsed_task, True, f"Learned:{route_key}→{working_origin}"

        # Also check destination-based rules (e.g., "US → use antwerp")
        if destination and destination in self._learned_route_ports:
            working_origin = self._learned_route_ports[destination]
            parsed_task.source = working_origin
            return parsed_task, True, f"Learned:{destination}→{working_origin}"

        # Fall back to server rules
        if not rules:
            return parsed_task, False, "Exploration"

        # DYNAMICALLY parse rules from text
        parsed_rules = self.rule_parser.parse_rules(rules)

        if not parsed_rules:
            return parsed_task, False, "Exploration"

        # Find an applicable rule
        applicable_rule = self.rule_parser.find_applicable_rule(
            parsed_rules,
            parsed_task.source,
            is_us,
        )

        if applicable_rule:
            # Apply the learned rule
            parsed_task.source = applicable_rule.alternative

            # Cache for future use
            self._learned_route_ports[route_key] = applicable_rule.alternative
            if destination:
                self._learned_route_ports[destination] = applicable_rule.alternative

            rule_name = f"{applicable_rule.error_code}→{applicable_rule.alternative}"
            if applicable_rule.condition:
                rule_name += f" ({applicable_rule.condition})"

            return parsed_task, True, rule_name

        return parsed_task, False, "Exploration"

    async def execute_action(
        self,
        mcp_client: Any,
        parsed_task: ParsedTask,
    ) -> ActionResult:
        """
        Execute the booking or customs action.

        TIER 2 (LLM REASONING) - PRIMARY PATH:
        ══════════════════════════════════════
        With Tier 1 disabled, LLM reasoning is the primary learning mechanism.
        The LLM suggests solutions via `preferred_solution` parameter.

        Priority order:
        1. preferred_solution (from LLM reasoning - Tier 2)
        2. Domain-specific parameters (documentation, source)
        3. Default values

        Learning: Successful actions are cached for potential Tier 1 re-enablement.
        """
        is_customs = parsed_task.parameters.get("is_customs", False)

        # Check for LLM-suggested solution (Tier 2)
        preferred = parsed_task.parameters.get("preferred_solution")

        # ═══════════════════════════════════════════════════════════════════
        # VALIDATION: Ensure LLM suggestion is a valid option
        # Prevents error codes (R-482, SH-701) from being used as solutions
        # ═══════════════════════════════════════════════════════════════════
        if preferred:
            preferred_lower = preferred.lower().replace(" ", "_")
            if is_customs:
                # Validate it's a known documentation type
                if preferred_lower not in [d.lower() for d in self.DOCUMENTATION_TYPES]:
                    preferred = None  # Invalid suggestion, ignore it
            else:
                # Validate it's a known port name
                if preferred_lower not in [p.lower() for p in self.ORIGIN_PORTS]:
                    preferred = None  # Invalid suggestion, ignore it

        if is_customs:
            # Execute customs clearance
            # Priority: LLM suggestion > learned documentation > default
            documentation = (
                preferred or parsed_task.parameters.get("documentation") or "standard"
            )
            result = await mcp_client.clear_customs(
                parsed_task.target,
                documentation,
            )

            if "CLEARED" in result:
                # 🎓 LEARN: Remember this documentation works for this destination
                destination = parsed_task.target
                if destination:
                    self._learned_customs_docs[destination] = documentation

                # NOTE: We don't record solutions here on first-try success.
                # Rule recording happens in handle_error() with GENERIC error codes
                # (e.g., CUSTOMS-HS-002 → enhanced) for cross-entity transfer.
                # Recording entity-specific codes (CUSTOMS_BOSTON) would prevent
                # rules from applying to different destinations.

                # ═══════════════════════════════════════════════════════════════════
                # FIX: Strategy format should be parseable - use colons, not arrows
                # ═══════════════════════════════════════════════════════════════════
                strategy = f"customs:{documentation}"
                if preferred:
                    strategy = f"LLM:{strategy}"

                return ActionResult(
                    success=True,
                    response=result,
                    strategy_used=strategy,
                )

            # Extract error code from customs response
            # BUGFIX: Always provide an error_code, even when the response
            # doesn't contain a structured CUSTOMS-XX-NNN code (e.g., MCP
            # validation errors). Without this, the framework-level error
            # recovery pipeline was silently bypassed.
            error_match = re.search(r"(CUSTOMS-[A-Z]+-\d{3})", result)
            error_code = error_match.group(1) if error_match else "CUSTOMS-FAIL"

            return ActionResult(
                success=False,
                response=result,
                error_code=error_code,
            )

        # Execute booking action
        # Priority: LLM suggestion > parsed source
        origin = preferred or parsed_task.source
        condition_key = parsed_task.parameters.get("condition_key")

        # ═══════════════════════════════════════════════════════════════════
        # BLACK SWAN CSP ENFORCEMENT:
        # For multi-condition scenarios, use hash-based enforcement where
        # Solution = f(hash(composite_key)) - ONLY ONE solution works!
        # ═══════════════════════════════════════════════════════════════════
        if condition_key:
            # ═══════════════════════════════════════════════════════════════
            # BUGFIX: Guard against None destination which causes Pydantic
            # validation error at the MCP tool level. Test scenarios
            # generated by generate_test_from_learned_keys can use
            # destinations not in DESTINATION_PORTS (e.g., "Paris"),
            # causing parsed_task.target=None. Without this guard,
            # EVERY pivot also fails (same None destination), so even
            # the correct solution (e.g., antwerp for R-482) never
            # succeeds. Fall back to book_shipment which tolerates
            # any destination string.
            # ═══════════════════════════════════════════════════════════════
            destination = parsed_task.target
            if destination:
                # Multi-condition: use hash-based enforcement
                result = await mcp_client.call_tool(
                    "execute_logistics_multi_condition",
                    {"condition_key": condition_key, "origin": origin, "destination": destination}
                )
            else:
                # Destination not parseable - fall back to lenient path
                result = await mcp_client.book_shipment(
                    origin,
                    parsed_task.target or "unknown",
                )
        else:
            # Single-condition: use base tool
            result = await mcp_client.book_shipment(
                origin,
                parsed_task.target,
            )

        if "CONFIRMED" in result:
            # 🎓 LEARN: Remember this route works (using the origin that worked)
            destination = parsed_task.target
            route_key = f"{origin}→{destination}"
            self._learned_route_ports[route_key] = origin  # Learn the working origin
            if destination:
                self._learned_route_ports[destination] = origin

            # NOTE: We don't record solutions here on first-try success.
            # Rule recording happens in handle_error() with GENERIC error codes
            # (e.g., R-482 → antwerp) for cross-entity transfer.
            # Recording entity-specific codes (ROUTE_BOSTON) would prevent
            # rules from applying to different routes with the same blocked port.

            # ═══════════════════════════════════════════════════════════════════
            # FIX: Strategy should ONLY contain the ORIGIN (working solution)
            # NOT "origin→destination" which causes extraction bugs.
            # The destination is irrelevant for learned rules - only the
            # working origin port matters for future scenarios.
            # ═══════════════════════════════════════════════════════════════════
            strategy = f"origin:{origin}"
            if preferred:
                strategy = f"LLM:{strategy}"

            return ActionResult(
                success=True,
                response=result,
                strategy_used=strategy,
            )

        # Extract error code from response (supports both single and multi-letter codes)
        # Matches: R-482, H-903, SH-701, LA-550, A-701
        # Also matches: ROUTE-312 (from execute_logistics_multi_condition)
        # BUGFIX: Always provide an error_code, even when the response doesn't
        # contain a structured XX-NNN code (e.g., MCP validation errors, tool
        # parameter rejections). The fallback "ROUTE-FAIL" ensures the error
        # recovery pipeline is always triggered.
        # BUGFIX: Use word boundary (\b) to prevent matching "TE-312" from
        # within "ROUTE-312". The old regex [A-Z]{1,2}-\d{3} would match
        # the LAST 2 chars of "ROUTE" + digits, producing wrong error codes.
        error_match = re.search(r"\b([A-Z]{1,5}-\d{3})\b", result)
        error_code = error_match.group(1) if error_match else "ROUTE-FAIL"

        return ActionResult(
            success=False,
            response=result,
            error_code=error_code,
        )

    async def handle_error(
        self,
        mcp_client: Any,
        error_code: str,
        parsed_task: ParsedTask,
        context: Dict[str, Any],
    ) -> ActionResult:
        """
        Handle error and attempt recovery.

        FAIR COMPARISON: Limited to MAX_RETRIES (same budget as baseline)
        The difference: After learning, PRECEPT knows which port/documentation works!

        Recovery strategy:
        1. Record the error for learning
        2. Try remaining options (within retry budget)
        3. Each failure adds to learning
        """
        is_customs = parsed_task.parameters.get("is_customs", False)

        # Record error for learning (this is how PRECEPT learns!)
        await mcp_client.record_error(
            error_code,
            f"{'Customs' if is_customs else 'Booking'} {parsed_task.source}→{parsed_task.target}",
        )

        # Track retries for fair comparison
        retries_made = context.get("retries_made", 0)

        if is_customs:
            # Handle customs error - try different documentation types
            tried_docs = context.get("tried_docs", {"standard"})
            remaining_docs = [
                d for d in self.DOCUMENTATION_TYPES if d not in tried_docs
            ]

            # Check if we have a LEARNED solution for this destination
            # NO HARDCODED MAPPINGS - use what was learned from previous experiences
            if parsed_task.target in self._learned_customs_docs:
                learned_doc = self._learned_customs_docs[parsed_task.target]
                if learned_doc in remaining_docs:
                    remaining_docs.remove(learned_doc)
                    remaining_docs.insert(0, learned_doc)

            for doc_type in remaining_docs:
                if retries_made >= self.MAX_RETRIES:
                    break

                tried_docs.add(doc_type)
                retries_made += 1
                result = await mcp_client.clear_customs(parsed_task.target, doc_type)

                if "CLEARED" in result:
                    # 🎓 LEARN: This documentation works for this destination!
                    destination = parsed_task.target
                    self._learned_customs_docs[destination] = doc_type

                    # ═══════════════════════════════════════════════════════════
                    # ROUTE-LEVEL RULE STORAGE: Include destination
                    # ═══════════════════════════════════════════════════════════
                    condition_key = (parsed_task.parameters or {}).get("condition_key")
                    base_key = condition_key if condition_key else error_code

                    # Route-specific rule: error|destination → doc_type
                    if destination:
                        route_rule_key = f"{base_key}|{destination}"
                    else:
                        route_rule_key = base_key

                    await mcp_client.record_solution(
                        error_code=route_rule_key,  # Route-specific customs rule
                        solution=doc_type,
                        context=f"Customs clearance for {destination} (conditions: {condition_key or error_code})",
                    )

                    # Also store fallback rule without destination
                    if destination:
                        await mcp_client.record_solution(
                            error_code=base_key,
                            solution=doc_type,
                            context=f"Customs clearance fallback (conditions: {condition_key or error_code})",
                        )

                    return ActionResult(
                        success=True,
                        response=result,
                        strategy_used=f"customs:{doc_type} (retry {retries_made}/{self.MAX_RETRIES})",
                    )

            return ActionResult(
                success=False,
                response=f"All customs retries exhausted ({retries_made}/{self.MAX_RETRIES})",
                strategy_used="Failed",
            )

        # Handle booking error - try different ports
        tried_ports = context.get("tried_ports", {parsed_task.source})
        remaining_ports = [p for p in self.ORIGIN_PORTS if p not in tried_ports]

        # PRECEPT's TRUE ADVANTAGE: Use learned knowledge!
        # 1. Known working ports FIRST (learned from previous tasks)
        # 2. Unknown ports in RANDOM order (same as baseline - fair comparison)
        import random

        destination = parsed_task.target
        route_key = f"{parsed_task.source}→{destination}"
        known_working = [
            p
            for p in remaining_ports
            if route_key in self._learned_route_ports
            and self._learned_route_ports[route_key] == p
        ]
        unknown = [p for p in remaining_ports if p not in known_working]
        random.shuffle(unknown)  # Random order like baseline!
        remaining_ports = known_working + unknown

        # Get condition_key for multi-condition enforcement
        condition_key = (parsed_task.parameters or {}).get("condition_key")

        # FAIR: Only try up to MAX_RETRIES (same as baseline's retry budget)
        for alt_origin in remaining_ports:
            if retries_made >= self.MAX_RETRIES:
                break  # Same limit as baseline!

            tried_ports.add(alt_origin)
            retries_made += 1

            # ═══════════════════════════════════════════════════════════
            # BLACK SWAN CSP: Use multi-condition tool for hash enforcement
            # ═══════════════════════════════════════════════════════════
            if condition_key:
                result = await mcp_client.call_tool(
                    "execute_logistics_multi_condition",
                    {"condition_key": condition_key, "origin": alt_origin, "destination": parsed_task.target}
                )
            else:
                result = await mcp_client.book_shipment(alt_origin, parsed_task.target)

            if "CONFIRMED" in result:
                # 🎓 LEARN: This route works!
                route_key = f"{parsed_task.source}→{destination}"
                self._learned_route_ports[route_key] = alt_origin
                if destination:
                    self._learned_route_ports[destination] = alt_origin

                # ═══════════════════════════════════════════════════════════════
                # ROUTE-LEVEL RULE STORAGE: Include destination for specificity
                # ═══════════════════════════════════════════════════════════════
                # Real-world: Same error might need different solutions for
                # different destinations. E.g., R-482→boston might use antwerp,
                # but R-482→singapore might use hamburg.
                #
                # Rule key format: "{condition_key}|{destination}" or "{error_code}|{destination}"
                # This enables route-specific learning across runs.
                # ═══════════════════════════════════════════════════════════════
                condition_key = (parsed_task.parameters or {}).get("condition_key")
                base_key = condition_key if condition_key else error_code

                # Include destination in rule key for route-level learning
                if destination:
                    route_rule_key = f"{base_key}|{destination}"
                else:
                    route_rule_key = base_key

                await mcp_client.record_solution(
                    error_code=route_rule_key,  # Route-specific: R-482|boston or SAFE+FAST|seattle
                    solution=alt_origin,
                    context=f"Route: {parsed_task.source}→{destination} (conditions: {condition_key or error_code})",
                )

                # Also store the base rule (without destination) for fallback retrieval
                if destination:
                    await mcp_client.record_solution(
                        error_code=base_key,  # Generic: R-482 or SAFE+FAST
                        solution=alt_origin,
                        context=f"Fallback rule (conditions: {condition_key or error_code})",
                    )

                return ActionResult(
                    success=True,
                    response=result,
                    strategy_used=f"Fallback to {alt_origin} (retry {retries_made}/{self.MAX_RETRIES})",
                )

            # Check for new error
            error_match = re.search(r"([A-Z]{1,2}-\d{3})", result)
            new_error_code = error_match.group(1) if error_match else None

            if new_error_code and new_error_code != error_code:
                # Record this error too (more learning!)
                await mcp_client.record_error(
                    new_error_code,
                    f"Booking {alt_origin}→{parsed_task.target}",
                )

        return ActionResult(
            success=False,
            response=f"All retries exhausted ({retries_made}/{self.MAX_RETRIES})",
            strategy_used="Failed",
        )

    def _get_domain_tools(self, mcp_client: Any) -> List[Any]:
        """Return logistics-specific tools."""
        if not AUTOGEN_AVAILABLE:
            return []

        tools = []

        async def book_shipment(origin: str, destination: str) -> str:
            """Book a shipment from origin to destination."""
            return await mcp_client.book_shipment(origin, destination)

        tools.append(FunctionTool(book_shipment, description="Book a shipment."))

        async def check_port(port: str) -> str:
            """Check port availability."""
            return await mcp_client.check_port(port)

        tools.append(FunctionTool(check_port, description="Check port status."))

        async def clear_customs(
            destination: str, documentation: str = "standard"
        ) -> str:
            """Clear customs for a shipment with specified documentation."""
            return await mcp_client.clear_customs(destination, documentation)

        tools.append(
            FunctionTool(clear_customs, description="Clear customs for shipment.")
        )

        return tools


class LogisticsBaselineStrategy(BaselineDomainStrategy):
    """
    Logistics baseline strategy - NO LEARNING.

    Behavior:
    - Always tries Rotterdam first (from task)
    - On failure, tries RANDOM alternatives
    - Does NOT know which ports work
    - Does NOT learn from failures
    """

    # ═══════════════════════════════════════════════════════════════════════════
    # CONFIGURATION - Import from single source of truth (same as PRECEPT)
    # ═══════════════════════════════════════════════════════════════════════════
    ORIGIN_PORTS = LogisticsConfig.ORIGIN_PORTS
    DOCUMENTATION_TYPES = LogisticsConfig.DOCUMENTATION_TYPES

    def __init__(self, max_retries: int = None):
        """
        Initialize the logistics baseline strategy.

        Args:
            max_retries: Maximum number of retries allowed. If None, uses DEFAULT_MAX_RETRIES.
        """
        super().__init__(max_retries=max_retries)

    @property
    def domain_name(self) -> str:
        return "logistics"

    def get_available_options(self) -> List[str]:
        # Return both ports and documentation types
        return self.ORIGIN_PORTS.copy() + self.DOCUMENTATION_TYPES.copy()

    def get_options_for_task(self, parsed_task: ParsedTask) -> List[str]:
        """Return options relevant to this specific task type."""
        is_customs = parsed_task.parameters.get("is_customs", False)
        if is_customs:
            # Customs tasks need documentation types
            return self.DOCUMENTATION_TYPES.copy()
        else:
            # Booking tasks need ports
            return self.ORIGIN_PORTS.copy()

    def parse_task(self, task: str) -> ParsedTask:
        """Parse logistics task."""
        import re

        task_lower = (
            task.lower()
            .replace("new york", "new_york")
            .replace("los angeles", "los_angeles")
            .replace("long beach", "long_beach")
        )

        # ═══════════════════════════════════════════════════════════════════
        # MULTI-CONDITION EXTRACTION: Parse [Conditions: X + Y + Z]
        # ═══════════════════════════════════════════════════════════════════
        condition_key = None
        conditions = []
        condition_match = re.search(r"\[Conditions:\s*([^\]]+)\]", task, re.IGNORECASE)
        if condition_match:
            cond_str = condition_match.group(1)
            conditions = [c.strip() for c in cond_str.split("+")]
            condition_key = "+".join(sorted(conditions))

        # Detect if this is a customs task
        is_customs_task = any(
            keyword in task_lower
            for keyword in [
                "customs",
                "clearance",
                "documentation",
                "import",
                "declaration",
            ]
        )

        # ═══════════════════════════════════════════════════════════════════
        # BUGFIX: Use LogisticsConfig.DESTINATION_PORTS (same source of truth
        # as PRECEPT's strategy) instead of a hardcoded list. The old hardcoded
        # list was missing singapore, london, hamburg — causing parsed_task.target
        # to be None for those destinations. This led to a Pydantic validation
        # error when calling execute_logistics_multi_condition with
        # destination=None, making ALL baseline multi-condition calls fail.
        # ═══════════════════════════════════════════════════════════════════
        destination = None
        for dest in LogisticsConfig.DESTINATION_PORTS:
            dest_check = dest.replace("_", " ")
            if dest in task_lower or dest_check in task_lower:
                destination = dest
                break

        return ParsedTask(
            raw_task=task,
            action="clear_customs" if is_customs_task else "book_shipment",
            entity="customs" if is_customs_task else "shipment",
            source="rotterdam",  # Default from task
            target=destination,
            parameters={
                "is_customs": is_customs_task,
                "condition_key": condition_key,
                "conditions": conditions,
            },
        )

    def get_default_option(self, parsed_task: ParsedTask) -> str:
        """Always start with default option."""
        is_customs = parsed_task.parameters.get("is_customs", False)
        if is_customs:
            return "standard"
        return "rotterdam"

    async def execute_action(
        self,
        mcp_client: Any,
        option: str,
        parsed_task: ParsedTask,
    ) -> Tuple[bool, str]:
        """Execute booking or customs with given option.

        BLACK SWAN CSP: For multi-condition scenarios, uses hash-based enforcement
        where Solution = f(hash(composite_key)). This is FAIR - baselines face the
        same strict enforcement as PRECEPT.
        """
        is_customs = parsed_task.parameters.get("is_customs", False)
        condition_key = parsed_task.parameters.get("condition_key")

        if is_customs:
            result = await mcp_client.clear_customs(parsed_task.target, option)
            success = "CLEARED" in result
        else:
            # ═══════════════════════════════════════════════════════════════════
            # BLACK SWAN CSP ENFORCEMENT:
            # For multi-condition scenarios, use the multi-condition tool which
            # enforces Solution = f(hash(composite_key)) - ONLY ONE solution works!
            # This is FAIR - baselines face the same strict enforcement as PRECEPT.
            # ═══════════════════════════════════════════════════════════════════
            if condition_key:
                # BUGFIX: Guard against None destination which causes Pydantic
                # validation error at the MCP tool level. If destination wasn't
                # parsed from the task text, fall back to book_shipment.
                destination = parsed_task.target
                if destination:
                    result = await mcp_client.call_tool(
                        "execute_logistics_multi_condition",
                        {"condition_key": condition_key, "origin": option, "destination": destination}
                    )
                else:
                    result = await mcp_client.book_shipment(option, "unknown")
            else:
                result = await mcp_client.book_shipment(option, parsed_task.target)
            success = "CONFIRMED" in result

        return success, result
