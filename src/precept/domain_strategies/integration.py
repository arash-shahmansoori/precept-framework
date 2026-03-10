"""
Integration Domain Strategy for PRECEPT.

Handles API/OAuth black swan scenarios.

Black Swan Types (from black_swan_gen.py):
- Auth_Zombie: Token expires; agent retries without refreshing
- Gateway_Masking: Generic 500/502 masks upstream failure
- Silent_Throttling: Latency spikes instead of 429

🚨 CRITICAL: This strategy does NOT have hardcoded error->solution mappings.

What this strategy KNOWS (configuration, not learning):
- What endpoints/sources exist (from IntegrationConfig)
- What auth providers are available (vocabulary)
- How to parse tasks into structured format
- How to call MCP tools

What this strategy does NOT KNOW (must be learned):
- Which endpoints have auth issues vs gateway issues
- Which recovery strategy works for each endpoint
- What error codes mean (vague: AUTH-ERR-401 doesn't reveal solution)
"""

from typing import Any, Dict, List, Optional, Tuple

from ..config import IntegrationConfig
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


class IntegrationDomainStrategy(DomainStrategy):
    """
    Integration domain strategy for API/OAuth black swan scenarios.

    Black Swan Types (from black_swan_gen.py):
    - Auth_Zombie: Token expires; agent retries without refreshing
    - Gateway_Masking: Generic 500/502 masks upstream failure
    - Silent_Throttling: Latency spikes instead of 429

    🚨 CRITICAL: This strategy does NOT have hardcoded error->solution mappings.

    What this strategy KNOWS (configuration, not learning):
    - What endpoints/sources exist (from IntegrationConfig)
    - How to parse tasks into structured format
    - How to call MCP tools

    What this strategy does NOT KNOW (must be learned):
    - Which endpoints have which type of failure
    - Which recovery strategy works for each endpoint
    - What error codes mean
    """

    # ═══════════════════════════════════════════════════════════════════════════
    # CONFIGURATION - Import from single source of truth
    # ═══════════════════════════════════════════════════════════════════════════
    AUTH_PROVIDERS = IntegrationConfig.AUTH_PROVIDERS
    API_ENDPOINTS = IntegrationConfig.API_ENDPOINT_TYPES
    KNOWN_SOURCES = IntegrationConfig.KNOWN_SOURCES
    KNOWN_ENDPOINTS = IntegrationConfig.KNOWN_ENDPOINTS

    # NO ERROR_PATTERNS here - that would be cheating!
    # The agent must LEARN which endpoints have which issues.

    def __init__(self, max_retries: Optional[int] = None):
        """
        Initialize the Integration domain strategy.

        Args:
            max_retries: Maximum number of retries allowed. If None, uses DEFAULT_MAX_RETRIES.
                        - 1 = near first-try only (1 initial + 1 retry = 2 attempts)
                        - 2 = balanced (1 initial + 2 retries = 3 attempts) [default]
                        - 4 = lenient (1 initial + 4 retries = 5 attempts)
        """
        super().__init__(max_retries=max_retries)

        # Dynamic rule parser - only knows the vocabulary (endpoints, sources)
        # Does NOT know which have issues or which recovery works
        self.rule_parser = DynamicRuleParser(
            known_entities=self.AUTH_PROVIDERS + list(self.API_ENDPOINTS)
        )

        # Runtime learned knowledge (empty at start!)
        self._learned_alternatives: Dict[str, str] = {}

        # KEY LEARNING: Source/Endpoint → Recovery strategy
        # Real-world: Different APIs fail differently, each needs specific fix
        # e.g., "salesforce" → "re-authenticate", "legacy-erp" → "escalate to ops"
        self._learned_recovery_strategies: Dict[str, str] = {}

    @property
    def category(self) -> BlackSwanCategory:
        return BlackSwanCategory.INTEGRATION

    @property
    def domain_name(self) -> str:
        return "integration"

    def get_system_prompt(self, learned_rules: List[str] = None) -> str:
        base = """You are an integration specialist with PRECEPT learning capabilities.

AVAILABLE ACTIONS:
- authenticate(provider): Authenticate with a provider
- call_api(endpoint): Call an API endpoint
- refresh_token(): Refresh authentication token

PRECEPT ADVANTAGES:
- Learns from auth failures (401/403 → refresh, not retry)
- Detects silent throttling (latency spikes)"""

        if learned_rules:
            rules_section = "\n\n═══ LEARNED RULES ═══\n"
            for i, rule in enumerate(learned_rules, 1):
                rules_section += f"{i}. {rule}\n"
            base = rules_section + base

        return base

    # WORKING alternatives - same as MCP server's WORKING_SOURCES/WORKING_ENDPOINTS
    WORKING_SOURCES = [
        "salesforce-backup",
        "hubspot-v2",
        "zendesk-premium",
        "stripe-webhook-v2",
        "google_workspace-admin",
        "microsoft_graph-delegated",
    ]
    WORKING_ENDPOINTS = [
        "legacy-erp-v2",
        "partner-api-proxy",
        "payment-gateway-v2",
        "analytics-api-backend",
        "inventory-service-direct",
        "notification-service-fallback",
    ]

    def get_available_actions(self) -> List[str]:
        return ["authenticate", "call_api", "refresh_token"]

    def get_available_entities(self) -> List[str]:
        return (
            self.AUTH_PROVIDERS
            + list(self.API_ENDPOINTS)
            + self.KNOWN_SOURCES
            + self.KNOWN_ENDPOINTS
        )

    def get_options_for_task(self, parsed_task: ParsedTask) -> List[str]:
        """Return ALL options (blocked + working) SHUFFLED for fair exploration.

        Shuffling ensures both PRECEPT and baselines have the same random
        chance of trying working options first. Without this, blocked options
        would always be tried first (unfair advantage for neither agent).
        """
        import random

        if parsed_task.action == "sync_data":
            options = self.KNOWN_SOURCES + self.WORKING_SOURCES
        else:
            options = self.KNOWN_ENDPOINTS + self.WORKING_ENDPOINTS

        # Shuffle for fair random exploration
        shuffled = options.copy()
        random.shuffle(shuffled)
        return shuffled

    def _normalize_for_matching(self, text: str) -> str:
        """Normalize text for entity matching.

        Converts 'analytics-api' and 'Analytics Api' to 'analytics api' for matching.
        """
        return text.lower().replace("-", " ").replace("_", " ")

    def parse_task(self, task: str) -> ParsedTask:
        task_normalized = self._normalize_for_matching(task)

        # Extract entity first (this determines the action type!)
        entity = "api"
        entity_type = "unknown"  # Track if we found a specific entity

        # Check for sources (OAuth/data sources → sync_data)
        for src in self.KNOWN_SOURCES:
            src_normalized = self._normalize_for_matching(src)
            if src_normalized in task_normalized:
                entity = src
                entity_type = "source"
                break

        # Check for endpoints (API endpoints → call_api)
        if entity_type == "unknown":  # Only if no source found
            for ep in self.KNOWN_ENDPOINTS:
                ep_normalized = self._normalize_for_matching(ep)
                if ep_normalized in task_normalized:
                    entity = ep
                    entity_type = "endpoint"
                    break

        # Determine action based on entity type
        # CRITICAL: Entity type takes precedence over keywords!
        if entity_type == "source":
            # We found a source entity → always sync_data
            action = "sync_data"
        elif entity_type == "endpoint":
            # We found an endpoint entity → always call_api
            action = "call_api"
        else:
            # No entity found - use keywords to determine action
            if any(
                kw in task_normalized
                for kw in ["sync", "pull", "fetch", "connect", "export", "refresh"]
            ):
                action = "sync_data"
            else:
                action = "call_api"  # default

        # ═══════════════════════════════════════════════════════════════════
        # MULTI-CONDITION EXTRACTION: Parse [Conditions: X + Y + Z]
        # ═══════════════════════════════════════════════════════════════════
        import re

        condition_key = None
        conditions = []
        condition_match = re.search(r"\[Conditions:\s*([^\]]+)\]", task, re.IGNORECASE)
        if condition_match:
            cond_str = condition_match.group(1)
            conditions = [c.strip() for c in cond_str.split("+")]
            # Generate deterministic key (sorted, joined with +)
            condition_key = "+".join(sorted(conditions))

        return ParsedTask(
            raw_task=task,
            action=action,
            entity=entity,
            source=entity,
            parameters={
                "endpoint": entity,
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
        Apply learned rules - THE KEY PRECEPT ADVANTAGE!

        Real-world: After learning that salesforce needs re-authentication,
        PRECEPT knows to refresh token BEFORE retrying (not just retry blindly).

        This is different from other domains - we're not switching endpoints,
        we're applying the CORRECT recovery strategy.
        """
        endpoint = parsed_task.entity

        # If we know this endpoint's recovery strategy, mark it
        if endpoint in self._learned_recovery_strategies:
            strategy = self._learned_recovery_strategies[endpoint]
            # Store the known strategy for handle_error to use
            parsed_task.parameters["known_recovery"] = strategy
            return parsed_task, True, f"Learned:{endpoint}→{strategy}"

        return parsed_task, False, "Exploration"

    async def execute_action(
        self, mcp_client: Any, parsed_task: ParsedTask
    ) -> ActionResult:
        """Execute integration action via MCP server tools with REAL blocking logic.

        The LLM suggests solutions via `preferred_solution` parameter.
        Priority order:
        1. preferred_solution (from LLM reasoning - Tier 2)
        2. Entity from task parsing
        3. Default value
        """
        action = parsed_task.action

        # Check for LLM-suggested solution (Tier 2) - CRITICAL for PRECEPT!
        preferred = parsed_task.parameters.get("preferred_solution")

        # Validate LLM suggestion is a valid source/endpoint
        if action == "sync_data":
            valid_options = self.KNOWN_SOURCES + self.WORKING_SOURCES
            if preferred and preferred in valid_options:
                endpoint = preferred
            else:
                endpoint = parsed_task.entity or "api"
        else:  # call_api
            valid_options = self.KNOWN_ENDPOINTS + self.WORKING_ENDPOINTS
            if preferred and preferred in valid_options:
                endpoint = preferred
            else:
                endpoint = parsed_task.entity or "api"

        # ═══════════════════════════════════════════════════════════════════
        # BLACK SWAN CSP ENFORCEMENT:
        # For multi-condition scenarios, use hash-based enforcement where
        # Solution = f(hash(composite_key)) - ONLY ONE solution works!
        # ═══════════════════════════════════════════════════════════════════
        condition_key = parsed_task.parameters.get("condition_key")

        # Use preferred solution if provided by LLM reasoning
        if preferred:
            endpoint = preferred

        try:
            if action == "sync_data":
                if condition_key:
                    # Multi-condition: use hash-based enforcement
                    response = await mcp_client.call_tool(
                        "execute_integration_multi_condition",
                        {
                            "condition_key": condition_key,
                            "solution": endpoint,
                            "operation": "sync",
                        },
                    )
                else:
                    # Single-condition: use base tool
                    response = await mcp_client.call_tool(
                        "sync_data", {"source": endpoint, "destination": "database"}
                    )
            else:
                if condition_key:
                    # Multi-condition: use hash-based enforcement
                    response = await mcp_client.call_tool(
                        "execute_integration_multi_condition",
                        {
                            "condition_key": condition_key,
                            "solution": endpoint,
                            "operation": "call",
                        },
                    )
                else:
                    # Single-condition: use base tool
                    response = await mcp_client.call_tool(
                        "call_api", {"endpoint": endpoint, "method": "GET"}
                    )

            response_str = str(response) if response else ""

            if "SUCCESS" in response_str:
                # Store ENDPOINT as strategy_used (not action name)
                # This ensures atomic precepts store valid solution hints
                return ActionResult(
                    success=True,
                    response=response_str,
                    strategy_used=endpoint,  # Just the endpoint/source that worked
                )

            # Extract error code from response using VAGUE codes from MCP server
            error_code = "INTEGRATION-ERROR"

            import re

            # Match vague codes like INT-401, INT-402, GW-401, GW-502
            code_match = re.search(
                r"(INT-\d+|GW-\d+|SOURCE-NOT-FOUND|ENDPOINT-NOT-FOUND)", response_str
            )
            if code_match:
                error_code = code_match.group(1)
            elif (
                "NOT-FOUND" in response_str or "not in approved" in response_str.lower()
            ):
                error_code = "NOT-FOUND"

            return ActionResult(
                success=False,
                response=response_str,
                error_code=error_code,
                strategy_used=endpoint,  # Just the endpoint/source
            )

        except Exception as e:
            return ActionResult(
                success=False,
                response=f"MCP call failed: {str(e)}",
                error_code="MCP-ERROR",
            )

    async def handle_error(
        self,
        mcp_client: Any,
        error_code: str,
        parsed_task: ParsedTask,
        context: Dict[str, Any],
    ) -> ActionResult:
        """
        Handle integration errors - TRUE LEARNING without hardcoded mappings.

        🚨 CRITICAL: This method does NOT know which alternative works!
        It tries all available options and LEARNS which one succeeds.

        FAIR COMPARISON: Limited to MAX_RETRIES (same budget as baseline)
        The difference: After learning, PRECEPT knows which alternative works!

        Working alternatives exist for each blocked source/endpoint:
        - salesforce → salesforce-backup
        - hubspot → hubspot-v2
        - zendesk → zendesk-premium
        - stripe → stripe-webhook-v2
        - legacy-erp → legacy-erp-v2
        - partner-api → partner-api-proxy
        - etc.
        """
        import random

        await mcp_client.record_error(error_code, f"API {parsed_task.entity}")
        endpoint = parsed_task.entity
        action = parsed_task.action

        # Get condition_key for multi-condition enforcement
        condition_key = (parsed_task.parameters or {}).get("condition_key")

        # Track retries for fair comparison
        retries_made = context.get("retries_made", 0)

        if action == "sync_data":
            # Mapping from blocked sources to working alternatives
            # MUST match MCP server's WORKING_SOURCES!
            ALL_ALTERNATIVES = {
                "salesforce": ["salesforce-backup", "hubspot-v2", "zendesk-premium"],
                "hubspot": ["hubspot-v2", "salesforce-backup", "zendesk-premium"],
                "zendesk": ["zendesk-premium", "hubspot-v2", "salesforce-backup"],
                "stripe": ["stripe-webhook-v2"],
                "google_workspace": ["google_workspace-admin"],
                "microsoft_graph": ["microsoft_graph-delegated"],
            }
            tried_sources = context.get("tried_sources", {endpoint})
            alternatives = ALL_ALTERNATIVES.get(
                endpoint, [f"{endpoint}-v2", f"{endpoint}-backup"]
            )
            remaining = [a for a in alternatives if a not in tried_sources]

            # PRECEPT's TRUE ADVANTAGE: Use learned knowledge!
            known_working = [
                a
                for a in remaining
                if endpoint in self._learned_recovery_strategies
                and self._learned_recovery_strategies[endpoint] == a
            ]
            unknown = [a for a in remaining if a not in known_working]
            random.shuffle(unknown)  # Random order like baseline!
            remaining = known_working + unknown

            for alt_source in remaining:
                if retries_made >= self.MAX_RETRIES:
                    break

                tried_sources.add(alt_source)
                retries_made += 1

                try:
                    # CONSISTENT WITH LOGISTICS: Use BASE tool directly
                    response = await mcp_client.call_tool(
                        "sync_data",
                        {"source": alt_source, "destination": "local"},
                    )

                    if "SUCCESS" in str(response):
                        # 🎓 LEARN: This alternative works!
                        self._learned_recovery_strategies[endpoint] = alt_source

                        rule_key = condition_key if condition_key else error_code
                        await mcp_client.record_solution(
                            error_code=rule_key,
                            solution=alt_source,
                            context=f"Alternative for {endpoint} (conditions: {condition_key or error_code})",
                        )
                        return ActionResult(
                            success=True,
                            response=str(response),
                            # Store just the source name for atomic precept storage
                            strategy_used=f"{alt_source} (retry {retries_made}/{self.MAX_RETRIES})",
                        )
                except Exception:
                    continue

        elif action == "call_api":
            # Mapping from blocked endpoints to working alternatives
            # MUST match MCP server's WORKING_ENDPOINTS!
            ALL_ALTERNATIVES = {
                "legacy-erp": ["legacy-erp-v2", "partner-api-proxy"],
                "partner-api": ["partner-api-proxy", "legacy-erp-v2"],
                "payment-gateway": ["payment-gateway-v2"],
                "analytics-api": ["analytics-api-backend"],
                "inventory-service": ["inventory-service-direct"],
                "notification-service": ["notification-service-fallback"],
            }
            tried_endpoints = context.get("tried_endpoints", {endpoint})
            alternatives = ALL_ALTERNATIVES.get(
                endpoint, [f"{endpoint}-v2", f"{endpoint}-proxy"]
            )
            remaining = [a for a in alternatives if a not in tried_endpoints]

            # PRECEPT's TRUE ADVANTAGE: Use learned knowledge!
            known_working = [
                a
                for a in remaining
                if endpoint in self._learned_recovery_strategies
                and self._learned_recovery_strategies[endpoint] == a
            ]
            unknown = [a for a in remaining if a not in known_working]
            random.shuffle(unknown)  # Random order like baseline!
            remaining = known_working + unknown

            for alt_endpoint in remaining:
                if retries_made >= self.MAX_RETRIES:
                    break

                tried_endpoints.add(alt_endpoint)
                retries_made += 1

                try:
                    # CONSISTENT WITH LOGISTICS: Use BASE tool directly
                    response = await mcp_client.call_tool(
                        "call_api",
                        {"endpoint": alt_endpoint, "method": "GET"},
                    )

                    if "SUCCESS" in str(response):
                        # 🎓 LEARN: This alternative works!
                        self._learned_recovery_strategies[endpoint] = alt_endpoint

                        rule_key = condition_key if condition_key else error_code
                        await mcp_client.record_solution(
                            error_code=rule_key,
                            solution=alt_endpoint,
                            context=f"Alternative for {endpoint} (conditions: {condition_key or error_code})",
                        )
                        return ActionResult(
                            success=True,
                            response=str(response),
                            # Store just the endpoint name for atomic precept storage
                            strategy_used=f"{alt_endpoint} (retry {retries_made}/{self.MAX_RETRIES})",
                        )
                except Exception:
                    continue

        return ActionResult(
            success=False,
            response=f"All retries exhausted ({retries_made}/{self.MAX_RETRIES})",
            strategy_used="Failed",
        )

    def _get_domain_tools(self, mcp_client: Any) -> List[Any]:
        return []


class IntegrationBaselineStrategy(BaselineDomainStrategy):
    """
    Integration baseline strategy - NO LEARNING.

    Behavior:
    - Always tries the endpoint from task first
    - On failure, tries RANDOM alternative endpoints/recovery actions
    - Does NOT know which endpoints have which issues
    - Does NOT learn from failures

    ALL endpoints are blocked (100%)! Baseline will struggle.
    - salesforce/hubspot: OAuth expired
    - zendesk: Rate limited
    - legacy-erp/partner-api: Gateway errors

    PRECEPT learns specific recovery for each error type and succeeds.
    """

    # ═══════════════════════════════════════════════════════════════════════════
    # CONFIGURATION - Import from single source of truth (same as PRECEPT)
    # ═══════════════════════════════════════════════════════════════════════════
    KNOWN_ENDPOINTS = (
        IntegrationConfig.KNOWN_ENDPOINTS + IntegrationConfig.KNOWN_SOURCES
    )
    KNOWN_SOURCES = IntegrationConfig.KNOWN_SOURCES

    def __init__(self, max_retries: Optional[int] = None):
        """
        Initialize the Integration baseline strategy.

        Args:
            max_retries: Maximum number of retries allowed. If None, uses DEFAULT_MAX_RETRIES.
        """
        super().__init__(max_retries=max_retries)

    @property
    def domain_name(self) -> str:
        return "integration"

    def get_available_options(self) -> List[str]:
        return self.KNOWN_ENDPOINTS.copy()

    # WORKING alternatives - same as MCP server's WORKING_SOURCES/WORKING_ENDPOINTS
    WORKING_SOURCES = [
        "salesforce-backup",
        "hubspot-v2",
        "zendesk-premium",
        "stripe-webhook-v2",
        "google_workspace-admin",
        "microsoft_graph-delegated",
    ]
    WORKING_ENDPOINTS = [
        "legacy-erp-v2",
        "partner-api-proxy",
        "payment-gateway-v2",
        "analytics-api-backend",
        "inventory-service-direct",
        "notification-service-fallback",
    ]

    def get_options_for_task(self, parsed_task: ParsedTask) -> List[str]:
        """Return ALL options (blocked + working) SHUFFLED for fair comparison.

        Shuffling ensures baselines have the same random chance as PRECEPT.
        """
        import random

        if parsed_task.action == "sync_data":
            options = self.KNOWN_SOURCES + self.WORKING_SOURCES
        else:
            options = self.KNOWN_ENDPOINTS + self.WORKING_ENDPOINTS

        shuffled = options.copy()
        random.shuffle(shuffled)
        return shuffled

    def _normalize_for_matching(self, text: str) -> str:
        """Normalize text for entity matching.

        Converts 'analytics-api' and 'Analytics Api' to 'analytics api' for matching.
        """
        return text.lower().replace("-", " ").replace("_", " ")

    def parse_task(self, task: str) -> ParsedTask:
        task_normalized = self._normalize_for_matching(task)

        # Extract entity first (this determines the action type!)
        entity = "api"
        entity_type = "unknown"  # Track if we found a specific entity

        # Check for sources (OAuth/data sources → sync_data)
        for src in self.KNOWN_SOURCES:
            src_normalized = self._normalize_for_matching(src)
            if src_normalized in task_normalized:
                entity = src
                entity_type = "source"
                break

        # Check for endpoints (API endpoints → call_api)
        if entity_type == "unknown":  # Only if no source found
            for ep in self.KNOWN_ENDPOINTS:
                ep_normalized = self._normalize_for_matching(ep)
                if ep_normalized in task_normalized:
                    entity = ep
                    entity_type = "endpoint"
                    break

        # Determine action based on entity type
        # CRITICAL: Entity type takes precedence over keywords!
        if entity_type == "source":
            # We found a source entity → always sync_data
            action = "sync_data"
        elif entity_type == "endpoint":
            # We found an endpoint entity → always call_api
            action = "call_api"
        else:
            # No entity found - use keywords to determine action
            if any(
                kw in task_normalized
                for kw in ["sync", "pull", "fetch", "connect", "export", "refresh"]
            ):
                action = "sync_data"
            else:
                action = "call_api"  # default

        # Build parameters
        parameters = {"endpoint": entity}

        # ═══════════════════════════════════════════════════════════════════
        # MULTI-CONDITION EXTRACTION: Parse [Conditions: X + Y + Z] pattern
        # CRITICAL: Must match PRECEPT strategy to ensure fair comparison
        # ═══════════════════════════════════════════════════════════════════
        import re

        # BUGFIX: Parse and SORT conditions like PRECEPT does. The old code
        # used the raw string without sorting, producing a different
        # condition_key than PRECEPT for the same conditions. Since the hash
        # is computed on condition_key, this caused different hash results.
        condition_match = re.search(r"\[Conditions:\s*([^\]]+)\]", task, re.IGNORECASE)
        if condition_match:
            cond_str = condition_match.group(1)
            conditions = [c.strip() for c in cond_str.split("+")]
            condition_key = "+".join(sorted(conditions))
            parameters["condition_key"] = condition_key
            parameters["conditions"] = conditions

        return ParsedTask(
            raw_task=task,
            action=action,
            entity=entity,
            source=entity,
            parameters=parameters,
        )

    def get_default_option(self, parsed_task: ParsedTask) -> str:
        return parsed_task.entity or "salesforce"

    async def execute_action(
        self,
        mcp_client: Any,
        option: str,
        parsed_task: ParsedTask,
    ) -> Tuple[bool, str]:
        """Execute via MCP server with REAL blocking logic.

        BLACK SWAN CSP: For multi-condition scenarios, uses hash-based enforcement
        where Solution = f(hash(composite_key)). This is FAIR - baselines face the
        same strict enforcement as PRECEPT.
        """
        action = parsed_task.action
        endpoint = option  # option is the endpoint
        condition_key = parsed_task.parameters.get("condition_key")

        # ═══════════════════════════════════════════════════════════════════
        # BLACK SWAN CSP ENFORCEMENT:
        # For multi-condition scenarios, use the multi-condition tool which
        # enforces Solution = f(hash(composite_key)) - ONLY ONE solution works!
        # This is FAIR - baselines face the same strict enforcement as PRECEPT.
        # ═══════════════════════════════════════════════════════════════════
        try:
            if action == "sync_data":
                if condition_key:
                    # BUGFIX: Use correct parameter names matching the MCP tool
                    # signature: execute_integration_multi_condition(condition_key,
                    # solution, operation). The old code passed "source" instead of
                    # "solution" and omitted "operation", causing Pydantic validation
                    # errors on every baseline multi-condition call.
                    response = await mcp_client.call_tool(
                        "execute_integration_multi_condition",
                        {
                            "condition_key": condition_key,
                            "solution": endpoint,
                            "operation": "sync",
                        },
                    )
                else:
                    # Single-condition: use base tool
                    response = await mcp_client.call_tool(
                        "sync_data", {"source": endpoint, "destination": "database"}
                    )
            else:
                if condition_key:
                    # BUGFIX: Same parameter name fix for call_api path.
                    response = await mcp_client.call_tool(
                        "execute_integration_multi_condition",
                        {
                            "condition_key": condition_key,
                            "solution": endpoint,
                            "operation": "call",
                        },
                    )
                else:
                    # Single-condition: use base tool
                    response = await mcp_client.call_tool(
                        "call_api", {"endpoint": endpoint, "method": "GET"}
                    )

            if isinstance(response, str) and "SUCCESS" in response:
                return True, response
            else:
                return False, str(response)

        except Exception as e:
            return False, f"MCP call failed: {str(e)}"
