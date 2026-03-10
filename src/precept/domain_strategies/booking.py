"""
Booking Domain Strategy for PRECEPT.

Handles reservation black swan scenarios.

Black Swan Types (from black_swan_gen.py):
- Phantom_Inventory: HTTP 200 OK but body is error
- Gateway_Timeout: Payment hangs upstream

🚨 CRITICAL: This strategy does NOT have hardcoded error->solution mappings.

What this strategy KNOWS (configuration, not learning):
- What flights exist (ALL_FLIGHTS)
- What payment gateways exist (PAYMENT_GATEWAYS)
- How to parse tasks into structured format
- How to call MCP tools

What this strategy does NOT KNOW (must be learned):
- Which flights have phantom inventory
- Which alternatives work for which errors
- What error codes mean

Configuration is centralized in BookingConfig for maintainability.
"""

from typing import Any, Dict, List, Optional, Tuple

from ..config import BookingConfig
from ..rule_parser import DynamicRuleParser
from .base import (
    ActionResult,
    BaselineDomainStrategy,
    BlackSwanCategory,
    DomainStrategy,
    ParsedTask,
    ProbeResult,
)

# AutoGen imports (optional)
try:
    from autogen_core.tools import FunctionTool

    AUTOGEN_AVAILABLE = True
except ImportError:
    AUTOGEN_AVAILABLE = False
    FunctionTool = None


class BookingDomainStrategy(DomainStrategy):
    """
    Booking domain strategy for reservation black swan scenarios.

    Black Swan Types (from black_swan_gen.py):
    - Phantom_Inventory: HTTP 200 OK but body is error
    - Gateway_Timeout: Payment hangs upstream

    🚨 CRITICAL: This strategy does NOT have hardcoded error->solution mappings.

    What this strategy KNOWS (configuration, not learning):
    - What flights exist (from BookingConfig)
    - What payment gateways exist (from BookingConfig)
    - How to parse tasks into structured format
    - How to call MCP tools

    What this strategy does NOT KNOW (must be learned):
    - Which flights have phantom inventory
    - Which alternatives work for which errors
    - What error codes mean
    """

    # ═══════════════════════════════════════════════════════════════════════════
    # CONFIGURATION - Import from single source of truth
    # NOTE: We do NOT import WORKING_FLIGHTS - that would be cheating!
    # The agent must LEARN which flights work through experience.
    # ═══════════════════════════════════════════════════════════════════════════
    PAYMENT_GATEWAYS = BookingConfig.PAYMENT_GATEWAYS
    BOOKING_SYSTEMS = BookingConfig.BOOKING_SYSTEMS
    ALL_FLIGHTS = BookingConfig.ALL_FLIGHTS
    KNOWN_FLIGHTS = BookingConfig.ALL_FLIGHTS  # Alias for backward compatibility
    DEFAULT_FLIGHT = (
        BookingConfig.DEFAULT_FLIGHT
    )  # Blocked flight (no unfair advantage)
    # REMOVED: WORKING_FLIGHTS - this was cheating (ground truth knowledge)

    def __init__(self, max_retries: Optional[int] = None):
        """
        Initialize the booking domain strategy.

        Args:
            max_retries: Maximum number of retries allowed. If None, uses DEFAULT_MAX_RETRIES.
        """
        super().__init__(max_retries=max_retries)

        self.rule_parser = DynamicRuleParser(
            known_entities=self.PAYMENT_GATEWAYS
            + self.BOOKING_SYSTEMS
            + self.ALL_FLIGHTS
        )
        self._learned_alternatives: Dict[str, str] = {}
        # KEY LEARNING: Blocked flight → Working alternative
        # Real-world: Phantom inventory (HTTP 200 but no seats) requires trying alternatives
        self._learned_working_flights: Dict[str, str] = {}  # route → working_flight
        self._learned_blocked_flights: set = (
            set()
        )  # flights that have phantom inventory

    @property
    def category(self) -> BlackSwanCategory:
        return BlackSwanCategory.BOOKING

    @property
    def domain_name(self) -> str:
        return "booking"

    def get_system_prompt(self, learned_rules: List[str] = None) -> str:
        base = """You are a booking agent with PRECEPT learning capabilities.

AVAILABLE ACTIONS:
- make_reservation(system): Make a reservation
- process_payment(gateway): Process payment

PRECEPT ADVANTAGES:
- Learns from phantom inventory (200 OK lies)
- Remembers gateway timeout patterns"""

        if learned_rules:
            rules_section = "\n\n═══ LEARNED RULES ═══\n"
            for i, rule in enumerate(learned_rules, 1):
                rules_section += f"{i}. {rule}\n"
            base = rules_section + base

        return base

    def get_available_actions(self) -> List[str]:
        return ["make_reservation", "process_payment"]

    def get_available_entities(self) -> List[str]:
        return self.PAYMENT_GATEWAYS + self.BOOKING_SYSTEMS

    def get_options_for_task(self, parsed_task: ParsedTask) -> List[str]:
        """Return flight options SHUFFLED for fair exploration.

        Shuffling ensures both PRECEPT and baselines have the same random
        chance of trying working flights first.
        """
        import random

        options = self.ALL_FLIGHTS.copy()
        random.shuffle(options)
        return options

    def parse_task(self, task: str) -> ParsedTask:
        import re

        task_lower = task.lower()
        task_upper = task.upper()

        # Detect action type
        action = "book_flight"
        if "payment" in task_lower or "pay" in task_lower:
            action = "process_payment"

        # Extract flight ID (blocked flights: AA-999, UA-666)
        flight_id = self.DEFAULT_FLIGHT  # Default (blocked - no unfair advantage)
        for flt in self.KNOWN_FLIGHTS:
            if flt in task_upper or flt.lower() in task_lower:
                flight_id = flt
                break

        # Extract gateway
        gateway = "card"  # Default
        for gw in self.PAYMENT_GATEWAYS:
            if gw in task_lower:
                gateway = gw
                break

        # ═══════════════════════════════════════════════════════════════════
        # MULTI-CONDITION EXTRACTION: Parse [Conditions: X + Y + Z]
        # ═══════════════════════════════════════════════════════════════════
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
            entity=flight_id,
            source=gateway,
            parameters={
                "gateway": gateway,
                "flight_id": flight_id,
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

        Real-world: After learning that AA-999 has phantom inventory,
        PRECEPT skips it and tries a learned working flight FIRST.

        🔧 FIX: Now checks MCP server's persisted rules (format: "BK-XXX → flight_id")
        and combines with local learned knowledge for best performance.
        """
        import logging

        logger = logging.getLogger("precept.booking")
        flight_id = parsed_task.entity

        # 🔧 FIX: First, extract working flights from MCP server's persisted rules
        # Rules format: "BK-XXX → DL-123" or "WORKING_FLIGHT → DL-123"
        mcp_working_flights: set = set()
        for rule in rules:
            if "→" in rule:
                parts = rule.split("→")
                if len(parts) == 2:
                    working_flight = parts[1].strip()
                    if working_flight in self.ALL_FLIGHTS:
                        mcp_working_flights.add(working_flight)
                        # Also update local knowledge
                        self._learned_working_flights[working_flight] = True

        if mcp_working_flights:
            logger.info(
                f"📚 Loaded {len(mcp_working_flights)} working flights from MCP rules: {mcp_working_flights}"
            )

        # Combine MCP rules with local knowledge
        all_working = set(self._learned_working_flights.keys()) | mcp_working_flights

        # If current flight is blocked, use a known working alternative
        if flight_id in self._learned_blocked_flights:
            if all_working:
                working_flight = list(all_working)[0]  # Use first known working flight
                parsed_task.entity = working_flight
                parsed_task.parameters["flight_id"] = working_flight
                logger.info(
                    f"🎯 Applied rule: Skip blocked {flight_id} → use {working_flight}"
                )
                return parsed_task, True, f"Learned:Skip {flight_id}→{working_flight}"

        # If we have learned working flights and the current flight is untested,
        # suggest a learned working flight as preferred solution (but still explore)
        # NOTE: We don't check WORKING_FLIGHTS - that would be cheating!
        # We only use knowledge learned through experience (all_working).
        if flight_id not in all_working and all_working:
            # Suggest a flight we LEARNED works (not from hardcoded ground truth)
            working_flight = list(all_working)[0]
            parsed_task.parameters["preferred_solution"] = working_flight
            logger.info(
                f"💡 Suggesting learned working flight: {working_flight} (original: {flight_id})"
            )
            return parsed_task, True, f"Learned:Suggest {working_flight}"

        return parsed_task, False, "Exploration"

    async def execute_action(
        self, mcp_client: Any, parsed_task: ParsedTask
    ) -> ActionResult:
        """Execute booking action via MCP server tools with REAL blocking logic.

        The LLM suggests solutions via `preferred_solution` parameter.
        Priority order:
        1. preferred_solution (from LLM reasoning - Tier 2)
        2. Entity from task parsing
        3. Default value (DEFAULT_FLIGHT - blocked, no unfair advantage)

        🔧 FIX: Now records working flights to MCP for cross-task learning!
        """
        import logging
        import re

        logger = logging.getLogger("precept.booking")

        # Check for LLM-suggested solution (Tier 2)
        preferred = parsed_task.parameters.get("preferred_solution")

        # Validate LLM suggestion is a valid flight
        if preferred and preferred in self.ALL_FLIGHTS:
            flight_id = preferred
        else:
            flight_id = parsed_task.entity or self.DEFAULT_FLIGHT

        passenger = parsed_task.parameters.get("passenger", "John Doe")
        condition_key = parsed_task.parameters.get("condition_key")

        # ═══════════════════════════════════════════════════════════════════
        # BLACK SWAN CSP ENFORCEMENT:
        # For multi-condition scenarios, use hash-based enforcement where
        # Solution = f(hash(composite_key)) - ONLY ONE solution works!
        # ═══════════════════════════════════════════════════════════════════
        try:
            if condition_key:
                # Multi-condition: use hash-based enforcement
                response = await mcp_client.call_tool(
                    "execute_booking_multi_condition",
                    {"condition_key": condition_key, "flight_id": flight_id, "passenger_name": passenger}
                )
            else:
                # Single-condition: use base tool
                response = await mcp_client.call_tool(
                    "book_flight", {"flight_id": flight_id, "passenger_name": passenger}
                )

            response_str = str(response) if response else ""

            # HTTP 200 LIE - check response body, not just status!
            if "SUCCESS" in response_str and "FAILED" not in response_str:
                # 🎓 LEARN: Remember this flight works!
                self._learned_working_flights[flight_id] = True

                # 🔧 FIX: Record working flight to MCP for persistence!
                # This enables cross-task learning: if task asked for blocked flight X
                # but we used working flight Y, record "WORKING_FLIGHT → Y"
                try:
                    await mcp_client.record_solution(
                        error_code="WORKING_FLIGHT",
                        solution=flight_id,
                        context=f"Flight {flight_id} confirmed working",
                    )
                    logger.info(f"📝 Recorded working flight: {flight_id}")
                except Exception as e:
                    logger.warning(f"Failed to record working flight: {e}")

                # Store FLIGHT ID as strategy_used (not status word)
                # This ensures atomic precepts store valid solution hints
                return ActionResult(
                    success=True,
                    response=response_str,
                    strategy_used=flight_id,  # Just the flight ID that worked
                )

            # Extract VAGUE error code from response (BK-XXX format)
            # NOTE: Strip trailing punctuation like periods that may be part of the sentence
            error_code = "BOOKING-ERROR"
            match = re.search(r"Error code: ([A-Z]{2,}-\d{3})", response_str)
            if match:
                error_code = match.group(1)

            return ActionResult(
                success=False,
                response=response_str,
                error_code=error_code,
                strategy_used=flight_id,  # Just the flight ID
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
        Handle booking errors - TRUE LEARNING without hardcoded mappings.

        🚨 CRITICAL: This method does NOT know which flight works!
        It tries all available flights and LEARNS which one succeeds.

        FAIR COMPARISON: Limited to MAX_RETRIES (same budget as baseline)
        The difference: After learning, PRECEPT knows which flight works!

        🔧 FIX: Now properly records error→solution mappings to MCP!
        """
        import logging
        import random

        logger = logging.getLogger("precept.booking")

        await mcp_client.record_error(error_code, f"Booking {parsed_task.entity}")

        # 🎓 LEARN: Mark this flight as blocked (phantom inventory)
        failed_flight = parsed_task.entity
        self._learned_blocked_flights.add(failed_flight)

        # Track retries for fair comparison
        retries_made = context.get("retries_made", 0)
        tried_flights = context.get("tried_flights", set())
        tried_flights.add(failed_flight)

        # For booking errors (BK-XXX), need to retry with different flights
        # Uses prefix matching for VAGUE error codes - NO hardcoded old codes!
        if error_code and error_code.startswith("BK-"):
            # Get untried flights
            alt_flights = [f for f in self.KNOWN_FLIGHTS if f not in tried_flights]

            # PRECEPT's TRUE ADVANTAGE: Use learned knowledge!
            # 1. Known working flights FIRST (learned from previous tasks)
            # 2. Unknown flights in RANDOM order (same as baseline)
            known_working = [
                f for f in alt_flights if f in self._learned_working_flights
            ]
            unknown = [f for f in alt_flights if f not in self._learned_working_flights]
            random.shuffle(unknown)  # Random order like baseline - fair comparison!
            alt_flights = known_working + unknown  # Learned flights first, then random

            logger.info(
                f"🔍 handle_error: {error_code} | Known working: {len(known_working)} | Unknown: {len(unknown)} | Retries left: {self.MAX_RETRIES - retries_made}"
            )

            for alt_flight in alt_flights:
                if retries_made >= self.MAX_RETRIES:
                    logger.warning(
                        f"❌ Retries exhausted ({retries_made}/{self.MAX_RETRIES}) for {error_code}"
                    )
                    break  # Same limit as baseline!

                tried_flights.add(alt_flight)
                retries_made += 1

                try:
                    response = await mcp_client.call_tool(
                        "book_flight",
                        {"flight_id": alt_flight, "passenger_name": "John Doe"},
                    )
                    if "SUCCESS" in str(response) and "FAILED" not in str(response):
                        # 🎓 LEARN: This flight works!
                        self._learned_working_flights[alt_flight] = True

                        # 🔧 FIX: Record solution to MCP for persistence
                        # This creates the actual learned rule: error_code → working_flight
                        try:
                            await mcp_client.record_solution(
                                error_code=error_code,
                                solution=alt_flight,
                                context=f"Working flight for blocked {failed_flight}",
                            )
                            logger.info(
                                f"✅ RULE LEARNED: {error_code} → {alt_flight} (retry {retries_made}/{self.MAX_RETRIES})"
                            )
                        except Exception as e:
                            logger.error(f"❌ Failed to record solution: {e}")

                        # 🔧 FIX: Also store as domain mapping for redundancy
                        try:
                            await mcp_client.call_tool(
                                "store_domain_mapping",
                                {
                                    "domain": "booking",
                                    "mapping_type": "error_solutions",
                                    "key": error_code,
                                    "value": alt_flight,
                                },
                            )
                            logger.debug(
                                f"📁 Domain mapping stored: booking/{error_code} → {alt_flight}"
                            )
                        except Exception as e:
                            logger.debug(
                                f"Domain mapping storage failed (non-critical): {e}"
                            )

                        return ActionResult(
                            success=True,
                            response=str(response),
                            # Store just the flight ID for atomic precept storage
                            strategy_used=f"{alt_flight} (retry {retries_made}/{self.MAX_RETRIES})",
                        )
                    else:
                        # This flight also failed - mark it blocked
                        self._learned_blocked_flights.add(alt_flight)
                        logger.debug(f"Flight {alt_flight} also blocked")
                except Exception as e:
                    logger.warning(f"Exception trying {alt_flight}: {e}")
                    continue

        return ActionResult(
            success=False,
            response=f"Booking failed - retries exhausted ({retries_made}/{self.MAX_RETRIES})",
        )

    def _get_domain_tools(self, mcp_client: Any) -> List[Any]:
        return []

    # =========================================================================
    # COMPASS EPISTEMIC PROBE PROTOCOL (DYNAMIC DISCOVERY)
    # =========================================================================
    # Probes are DISCOVERED at runtime from the MCP server, not hardcoded here.
    # The base class discover_probes() method queries the MCP server.
    #
    # The MCP server's _DIAGNOSTIC_PROBES registry defines available probes.
    # COMPASS learns which probes are useful for which errors through experience.
    #
    # Learning flow:
    # 1. Agent starts → discover_probes() queries MCP server
    # 2. MCP returns: {check_inventory, check_gds_status, check_fare}
    # 3. BK-401 error occurs → COMPASS doesn't know which probe to use
    # 4. Tries check_inventory (exploration) → discovers PHANTOM_INVENTORY
    # 5. LEARNS: "BK-4xx errors → check_inventory helps"
    # 6. Next BK-4xx → immediately uses check_inventory (exploitation)

    # NOTE: The _probe_* methods below are FALLBACK implementations.
    # When discovery works, the base class creates generic executors.
    # These are used when discovery fails or for testing.

    async def _probe_inventory(
        self, mcp_client: Any, context: Dict[str, Any]
    ) -> ProbeResult:
        """
        Probe flight inventory via alternative source.

        This probe calls the MCP server's `check_inventory` tool which:
        1. Queries BookingConfig for blocked flights (simulating GDS lookup)
        2. Returns structured response: PHANTOM_INVENTORY, AVAILABLE, or UNKNOWN
        3. Provides actionable information for COMPASS decision-making

        The probe DISCOVERS the constraint - it doesn't KNOW which flights work.
        """
        import logging

        logger = logging.getLogger("precept.booking.probe")

        # Extract flight ID from context (handle both dict and object forms)
        parsed_task = context.get("parsed_task", {})
        if isinstance(parsed_task, dict):
            flight_id = (
                context.get("entity")
                or parsed_task.get("flight_id")
                or parsed_task.get("entity")
            )
        else:
            flight_id = context.get("entity")

        if not flight_id:
            return ProbeResult(
                success=False,
                raw_output="No flight ID in context for inventory probe",
                should_retry=True,
            )

        try:
            # Call the MCP server's check_inventory tool
            response = await mcp_client.call_tool(
                "check_inventory", {"flight_id": flight_id}
            )
            response_str = str(response) if response else ""
            logger.debug(f"Inventory probe response: {response_str[:100]}")

            # Parse structured response from MCP tool
            response_upper = response_str.upper()

            if "PHANTOM_INVENTORY" in response_upper or "NO_SEATS" in response_upper:
                # Discovered: This flight has phantom inventory (PHYSICS constraint)
                self._learned_blocked_flights.add(flight_id)
                logger.info(f"🔍 Probe discovered PHANTOM_INVENTORY for {flight_id}")

                # Find learned working alternatives
                alternatives = [
                    f
                    for f in self.ALL_FLIGHTS
                    if f not in self._learned_blocked_flights
                    and f in self._learned_working_flights
                ]
                alternative = alternatives[0] if alternatives else None

                return ProbeResult(
                    success=True,
                    constraint_discovered="PHANTOM_INVENTORY",
                    constraint_tier="physics",  # Can't book what doesn't exist
                    negotiated_alternative=alternative,
                    raw_output=response_str,
                    should_retry=False,  # Don't retry same flight
                )

            elif "AVAILABLE" in response_upper:
                # Probe confirms inventory exists - error was transient
                logger.debug(f"Probe shows {flight_id} has AVAILABLE inventory")
                return ProbeResult(
                    success=True,
                    raw_output=response_str,
                    should_retry=True,  # Safe to retry
                )

            elif "UNKNOWN" in response_upper:
                # Probe inconclusive - not in GDS cache
                logger.debug(f"Probe inconclusive for {flight_id}")
                return ProbeResult(
                    success=True,
                    raw_output=response_str,
                    should_retry=True,
                )

            else:
                # Unexpected response format
                return ProbeResult(
                    success=True,
                    raw_output=response_str,
                    should_retry=True,
                )

        except Exception as e:
            logger.warning(f"Inventory probe exception: {e}")
            # Fall back to learned knowledge if available
            if flight_id in self._learned_blocked_flights:
                alternatives = [
                    f
                    for f in self.ALL_FLIGHTS
                    if f not in self._learned_blocked_flights
                    and f in self._learned_working_flights
                ]
                alternative = alternatives[0] if alternatives else None
                return ProbeResult(
                    success=True,
                    constraint_discovered="PHANTOM_INVENTORY",
                    constraint_tier="physics",
                    negotiated_alternative=alternative,
                    raw_output=f"From learned knowledge: {flight_id} is blocked",
                    should_retry=False,
                )

            return ProbeResult(
                success=False,
                raw_output=f"Probe failed: {str(e)}",
                should_retry=True,
            )

    async def _probe_gds_sync(
        self, mcp_client: Any, context: Dict[str, Any]
    ) -> ProbeResult:
        """
        Check if GDS systems are synchronized.

        This probe calls the MCP server's `check_gds_status` tool which:
        - Compares availability across Sabre, Amadeus, Travelport
        - Returns SYNCED or OUT_OF_SYNC status
        - Provides last sync timestamp

        GDS desync is a POLICY constraint (transient, wait and retry).
        """
        import logging

        logger = logging.getLogger("precept.booking.probe")

        try:
            response = await mcp_client.call_tool("check_gds_status", {})
            response_str = str(response) if response else ""
            logger.debug(f"GDS sync probe response: {response_str[:100]}")

            response_upper = response_str.upper()

            if "OUT_OF_SYNC" in response_upper or "STALE" in response_upper:
                logger.info("🔍 Probe discovered GDS_STALE condition")
                return ProbeResult(
                    success=True,
                    constraint_discovered="GDS_STALE",
                    constraint_tier="policy",  # Policy: wait for sync
                    raw_output=response_str,
                    should_retry=True,  # Transient - retry after short delay
                )

            if "SYNCED" in response_upper:
                logger.debug("GDS systems are synchronized")
                return ProbeResult(
                    success=True,
                    raw_output=response_str,
                    should_retry=True,
                )

            return ProbeResult(
                success=True,
                raw_output=response_str,
                should_retry=True,
            )

        except Exception as e:
            logger.debug(f"GDS sync probe exception: {e}")
            return ProbeResult(
                success=False,
                raw_output=f"GDS probe unavailable: {str(e)}",
                should_retry=True,
            )

    async def _probe_fare(
        self, mcp_client: Any, context: Dict[str, Any]
    ) -> ProbeResult:
        """
        Check if fare is still valid and bookable.

        This probe calls the MCP server's `check_fare` tool which:
        - Re-prices the itinerary
        - Checks fare rules and restrictions
        - Warns if inventory is phantom despite valid fare

        FARE_EXPIRED is a POLICY constraint (re-search required).
        """
        import logging

        logger = logging.getLogger("precept.booking.probe")

        # Extract flight ID from context
        parsed_task = context.get("parsed_task", {})
        if isinstance(parsed_task, dict):
            flight_id = (
                context.get("entity")
                or parsed_task.get("flight_id")
                or parsed_task.get("entity")
            )
        else:
            flight_id = context.get("entity")

        if not flight_id:
            return ProbeResult(
                success=False,
                raw_output="No flight ID in context for fare probe",
                should_retry=True,
            )

        try:
            response = await mcp_client.call_tool(
                "check_fare", {"flight_id": flight_id}
            )
            response_str = str(response) if response else ""
            logger.debug(f"Fare probe response: {response_str[:100]}")

            response_upper = response_str.upper()

            if "EXPIRED" in response_upper:
                logger.info(f"🔍 Probe discovered FARE_EXPIRED for {flight_id}")
                return ProbeResult(
                    success=True,
                    constraint_discovered="FARE_EXPIRED",
                    constraint_tier="policy",  # Policy: re-search required
                    raw_output=response_str,
                    should_retry=False,  # Can't proceed with expired fare
                )

            if "VALID" in response_upper:
                # Fare is valid, but check for phantom inventory warning
                if "PHANTOM" in response_upper:
                    logger.info(f"Fare valid but inventory phantom for {flight_id}")
                    # Don't discover constraint here - let inventory probe handle it
                    return ProbeResult(
                        success=True,
                        raw_output=response_str,
                        should_retry=True,  # Fare OK, inventory issue separate
                    )
                logger.debug(f"Fare valid for {flight_id}")
                return ProbeResult(
                    success=True,
                    raw_output=response_str,
                    should_retry=True,
                )

            return ProbeResult(
                success=True,
                raw_output=response_str,
                should_retry=True,
            )

        except Exception as e:
            logger.debug(f"Fare probe exception: {e}")
            return ProbeResult(
                success=False,
                raw_output=f"Fare probe unavailable: {str(e)}",
                should_retry=True,
            )


class BookingBaselineStrategy(BaselineDomainStrategy):
    """
    Booking baseline strategy - NO LEARNING.

    Behavior:
    - Always tries the flight from task first
    - On failure, tries RANDOM alternatives
    - Does NOT know which flights work
    - Does NOT learn from failures
    """

    # ═══════════════════════════════════════════════════════════════════════════
    # CONFIGURATION - Import from single source of truth (same as PRECEPT)
    # ═══════════════════════════════════════════════════════════════════════════
    KNOWN_FLIGHTS = BookingConfig.ALL_FLIGHTS
    PAYMENT_GATEWAYS = BookingConfig.PAYMENT_GATEWAYS
    DEFAULT_FLIGHT = (
        BookingConfig.DEFAULT_FLIGHT
    )  # Blocked flight (no unfair advantage)

    def __init__(self, max_retries: Optional[int] = None):
        """
        Initialize the booking baseline strategy.

        Args:
            max_retries: Maximum number of retries allowed. If None, uses DEFAULT_MAX_RETRIES.
        """
        super().__init__(max_retries=max_retries)

    @property
    def domain_name(self) -> str:
        return "booking"

    def get_available_options(self) -> List[str]:
        """Return flight options SHUFFLED for fair exploration."""
        import random

        options = self.KNOWN_FLIGHTS.copy()
        random.shuffle(options)
        return options

    def parse_task(self, task: str) -> ParsedTask:
        import re

        task_lower = task.lower()
        task_upper = task.upper()

        # BUGFIX: Match both uppercase and lowercase like PRECEPT's parse_task.
        # The old code only checked uppercase, missing flight IDs in lowercase.
        flight_id = self.DEFAULT_FLIGHT  # Default (blocked - no unfair advantage)
        for flt in self.KNOWN_FLIGHTS:
            if flt in task_upper or flt.lower() in task_lower:
                flight_id = flt
                break

        # ═══════════════════════════════════════════════════════════════════
        # MULTI-CONDITION EXTRACTION: Parse [Conditions: X + Y + Z]
        # CRITICAL: Must match PRECEPT strategy for fair comparison
        # ═══════════════════════════════════════════════════════════════════
        condition_key = None
        conditions = []
        condition_match = re.search(r"\[Conditions:\s*([^\]]+)\]", task, re.IGNORECASE)
        if condition_match:
            cond_str = condition_match.group(1)
            conditions = [c.strip() for c in cond_str.split("+")]
            condition_key = "+".join(sorted(conditions))  # SORTED like PRECEPT!

        return ParsedTask(
            raw_task=task,
            action="book_flight",
            entity=flight_id,
            source="card",
            parameters={
                "flight_id": flight_id,
                "condition_key": condition_key,
                "conditions": conditions,
            },
        )

    def get_default_option(self, parsed_task: ParsedTask) -> str:
        return parsed_task.entity or self.DEFAULT_FLIGHT

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
        flight_id = option  # option is the flight ID for booking
        condition_key = parsed_task.parameters.get("condition_key")

        try:
            if condition_key:
                # BUGFIX: Guard against None flight_id which causes Pydantic
                # validation error at the MCP tool level (same as logistics fix).
                if flight_id:
                    # Multi-condition: use hash-based enforcement
                    response = await mcp_client.call_tool(
                        "execute_booking_multi_condition",
                        {"condition_key": condition_key, "flight_id": flight_id, "passenger_name": "John Doe"}
                    )
                else:
                    response = await mcp_client.call_tool(
                        "book_flight", {"flight_id": self.DEFAULT_FLIGHT, "passenger_name": "John Doe"}
                    )
            else:
                # Single-condition: use base tool
                response = await mcp_client.call_tool(
                    "book_flight", {"flight_id": flight_id or self.DEFAULT_FLIGHT, "passenger_name": "John Doe"}
                )

            response_str = str(response) if response else ""
            # HTTP 200 LIE - check for actual failure in body
            if "SUCCESS" in response_str and "FAILED" not in response_str:
                return True, response_str
            else:
                return False, response_str

        except Exception as e:
            return False, f"MCP call failed: {str(e)}"
