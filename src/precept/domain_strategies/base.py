"""
Base classes for Domain Strategy Pattern.

This module provides abstract base classes for domain-specific strategies
that handle different black swan categories.

Architecture:
- DomainStrategy: For PRECEPT agents with learning capabilities
- BaselineDomainStrategy: For baseline agents with NO learning (fair comparison)

═══════════════════════════════════════════════════════════════════════════════
NOTE: TIER 1 (Programmatic Rule Lookup) has been DISABLED in precept_agent.py.
═══════════════════════════════════════════════════════════════════════════════
The `apply_learned_rules` method in each strategy is no longer called.
Learning now happens exclusively through Tier 2 (LLM reasoning):
1. LLM receives learned rules + task context
2. LLM suggests solution via `preferred_solution` parameter
3. `execute_action` applies LLM's suggestion with priority

The `apply_learned_rules` methods are preserved for potential future re-enablement.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple

# AutoGen imports (optional - only needed if creating tools)
try:
    from autogen_core.tools import FunctionTool

    AUTOGEN_AVAILABLE = True
except ImportError:
    AUTOGEN_AVAILABLE = False
    FunctionTool = None


class BlackSwanCategory(Enum):
    """Supported black swan categories from black_swan_gen.py."""

    LOGISTICS = "Logistics"
    CODING = "Coding"
    DEVOPS = "DevOps"
    FINANCE = "Finance"
    BOOKING = "Booking"
    INTEGRATION = "Integration"


@dataclass
class ParsedTask:
    """Generic parsed task representation."""

    raw_task: str
    action: str  # e.g., "book", "install", "deploy"
    entity: str  # e.g., "shipment", "package", "stack"
    source: Optional[str] = None  # e.g., origin port, source repo
    target: Optional[str] = None  # e.g., destination, target env
    parameters: Optional[Dict[str, Any]] = None


@dataclass
class ActionResult:
    """Result of executing an action."""

    success: bool
    response: str
    error_code: Optional[str] = None
    fallback_action: Optional[Dict[str, Any]] = None
    strategy_used: str = ""


# =============================================================================
# COMPASS EPISTEMIC PROBE PROTOCOL
# =============================================================================
# Domain strategies can optionally provide diagnostic probes.
# This enables COMPASS to trigger epistemic detours when errors are vague.


@dataclass
class ProbeSpec:
    """
    Specification for a diagnostic probe.

    Domain strategies return these to tell COMPASS what probes are AVAILABLE.

    IMPORTANT: error_patterns is now OPTIONAL and should be LEARNED, not hardcoded.
    - If empty, COMPASS will learn when this probe is useful
    - If provided, treated as initial hints (can be overridden by learning)

    The learning flow:
    1. Error occurs → COMPASS doesn't know which probe to use
    2. Tries probes (random or heuristic order)
    3. Tracks which probes revealed useful constraints
    4. Builds learned mapping: error_signature → useful_probes

    Attributes:
        probe_id: Unique identifier for this probe type
        description: Human-readable description of what this probe investigates
        execute: Async function that executes the probe and returns result
        error_patterns: OPTIONAL initial hints (learned over time, not hardcoded)
        reveals: What information this probe CAN reveal (for documentation)
        cost: Relative cost of executing this probe (for prioritization)
    """

    probe_id: str
    description: str
    execute: Callable[..., Any]  # async (mcp_client, context) -> ProbeResult
    error_patterns: List[str] = field(default_factory=list)  # Now OPTIONAL hints
    reveals: str = ""
    cost: float = 1.0  # Low cost = try first when exploring


@dataclass
class ProbeResult:
    """
    Result of executing a diagnostic probe.

    Attributes:
        success: Whether the probe executed successfully
        constraint_discovered: ID of constraint discovered (if any)
        constraint_tier: "physics", "policy", "instruction" (if constraint found)
        negotiated_alternative: Suggested alternative if blocked
        raw_output: Raw output from probe execution
        should_retry: Whether the original action should be retried
    """

    success: bool
    constraint_discovered: Optional[str] = None
    constraint_tier: Optional[str] = None  # "physics", "policy", "instruction"
    negotiated_alternative: Optional[str] = None
    raw_output: str = ""
    should_retry: bool = True


class DomainStrategy(ABC):
    """
    Abstract base class for domain-specific strategies.

    Implement this for each black swan category to make
    PRECEPTAgent work with that domain.

    Features:
    - Learning from errors via dynamic rule parsing
    - Domain-specific error handling and recovery
    - LLM reasoning integration via `preferred_solution` parameter

    ═══════════════════════════════════════════════════════════════════════════
    TIER 2 ONLY (LLM Reasoning):
    ═══════════════════════════════════════════════════════════════════════════
    With Tier 1 disabled, `apply_learned_rules` is NOT called by precept_agent.py.
    Learning happens via LLM reasoning:
    1. LLM receives learned rules as context
    2. LLM suggests solution via `preferred_solution` parameter
    3. `execute_action` applies LLM's suggestion with priority

    FAIR COMPARISON: Same retry budget as baseline (MAX_RETRIES = 1)
    - Both agents get 2 total attempts (1 initial + 1 retry)
    - PRECEPT uses LLM reasoning to choose the RIGHT solution on first try
    - Baseline must guess correctly with limited retries

    Usage:
        class MyDomainStrategy(DomainStrategy):
            @property
            def category(self) -> BlackSwanCategory:
                return BlackSwanCategory.LOGISTICS

            # ... implement other abstract methods
    """

    # FAIR COMPARISON: Same retry budget as baseline
    # 4 retries = sufficient exploration for black swan scenarios
    # PRECEPT applies learned rules proactively, baseline must guess
    DEFAULT_MAX_RETRIES = 4  # Total attempts = 1 initial + 4 retries = 5

    def __init__(self, max_retries: int = None):
        """
        Initialize the domain strategy.

        Args:
            max_retries: Maximum number of retries allowed. If None, uses DEFAULT_MAX_RETRIES.
                        - 1 = near first-try only (minimal exploration)
                        - 4 = standard (aligned with AgentConfig.max_retries) [default]
        """
        self.max_retries = (
            max_retries if max_retries is not None else self.DEFAULT_MAX_RETRIES
        )

    @property
    def MAX_RETRIES(self) -> int:
        """Property for backward compatibility with existing code using MAX_RETRIES."""
        return self.max_retries

    @property
    @abstractmethod
    def category(self) -> BlackSwanCategory:
        """Return the black swan category this strategy handles."""
        pass

    @property
    @abstractmethod
    def domain_name(self) -> str:
        """Return human-readable domain name."""
        pass

    @abstractmethod
    def get_system_prompt(self, learned_rules: List[str] = None) -> str:
        """
        Generate system prompt for this domain.

        Args:
            learned_rules: Optional list of learned rules to include

        Returns:
            System prompt string
        """
        pass

    @abstractmethod
    def get_available_actions(self) -> List[str]:
        """Return list of available actions in this domain."""
        pass

    @abstractmethod
    def get_available_entities(self) -> List[str]:
        """Return list of known entities (ports, packages, etc.)."""
        pass

    def get_options_for_task(self, parsed_task: "ParsedTask") -> List[str]:
        """
        Return options relevant to this specific task type.

        Override this method to return task-specific options:
        - Booking tasks → port names
        - Customs tasks → documentation types

        By default, returns all available entities.
        """
        return self.get_available_entities()

    @abstractmethod
    def parse_task(self, task: str) -> ParsedTask:
        """
        Parse a task string into structured format.

        Args:
            task: Raw task string

        Returns:
            ParsedTask with extracted components
        """
        pass

    @abstractmethod
    def apply_learned_rules(
        self,
        parsed_task: ParsedTask,
        rules: List[str],
    ) -> Tuple[ParsedTask, bool, str]:
        """
        Apply learned rules to transform task before execution (TIER 1 - DISABLED).

        ⚠️ NOTE: This method is currently NOT called from precept_agent.py.
        Tier 1 has been disabled in favor of Tier 2 (LLM reasoning only).
        This method is preserved for potential future re-enablement.

        Args:
            parsed_task: The parsed task
            rules: List of learned rules

        Returns:
            (modified_task, was_rule_applied, rule_name)
        """
        pass

    @abstractmethod
    async def execute_action(
        self,
        mcp_client: Any,
        parsed_task: ParsedTask,
    ) -> ActionResult:
        """
        Execute the primary action for this task.

        Args:
            mcp_client: The MCP client to use
            parsed_task: The parsed task

        Returns:
            ActionResult with success/failure and any error codes
        """
        pass

    @abstractmethod
    async def handle_error(
        self,
        mcp_client: Any,
        error_code: str,
        parsed_task: ParsedTask,
        context: Dict[str, Any],
    ) -> ActionResult:
        """
        Handle an error and attempt recovery.

        Args:
            mcp_client: The MCP client
            error_code: The error code received
            parsed_task: The original parsed task
            context: Additional context

        Returns:
            ActionResult from the recovery attempt
        """
        pass

    def create_autogen_tools(self, mcp_client: Any) -> List[Any]:
        """
        Create AutoGen FunctionTools for this domain.

        Override to add domain-specific tools.
        Default implementation provides common PRECEPT tools.

        Args:
            mcp_client: The MCP client to wrap

        Returns:
            List of FunctionTool objects
        """
        if not AUTOGEN_AVAILABLE:
            return []

        tools = []

        # Common PRECEPT tools (available for all domains)
        async def get_learned_rules() -> str:
            """Get all learned rules. Call this FIRST."""
            return await mcp_client.get_learned_rules()

        tools.append(
            FunctionTool(
                get_learned_rules, description="Get learned rules. CALL FIRST!"
            )
        )

        async def record_error(error_code: str, context: str) -> str:
            """Record an error for learning."""
            return await mcp_client.record_error(error_code, context)

        tools.append(
            FunctionTool(record_error, description="Record error for learning.")
        )

        async def retrieve_memories(query: str) -> str:
            """Search past experiences."""
            return await mcp_client.retrieve_memories(query)

        tools.append(
            FunctionTool(retrieve_memories, description="Search past experiences.")
        )

        # Add domain-specific tools (subclasses override _get_domain_tools)
        tools.extend(self._get_domain_tools(mcp_client))

        return tools

    @abstractmethod
    def _get_domain_tools(self, mcp_client: Any) -> List[Any]:
        """Return domain-specific tools. Override in subclasses."""
        pass

    # =========================================================================
    # COMPASS EPISTEMIC PROBE PROTOCOL (Dynamic Discovery)
    # =========================================================================
    # Probes are DISCOVERED at runtime from the MCP server, not hardcoded.
    # The agent learns which probes are useful for which errors through experience.

    # Cache for discovered probes (populated by discover_probes)
    _discovered_probes: List["ProbeSpec"] = []
    _probes_discovered: bool = False

    async def discover_probes(self, mcp_client: Any) -> List[ProbeSpec]:
        """
        Discover available diagnostic probes from the MCP server.

        This is the KEY to truly dynamic learning:
        1. Agent doesn't know a priori what probes exist
        2. Queries MCP server for available diagnostics
        3. Creates ProbeSpec objects dynamically
        4. COMPASS learns which probes help for which errors

        Args:
            mcp_client: MCP client to query for available probes

        Returns:
            List of discovered ProbeSpec objects
        """
        import json
        import logging

        logger = logging.getLogger(f"precept.{self.domain_name}")

        try:
            # Query MCP server for available probes
            response = await mcp_client.call_tool(
                "discover_probes", {"domain": self.domain_name}
            )
            response_str = str(response) if response else "{}"

            # Parse probe registry
            probe_registry = json.loads(response_str)
            domain_probes = probe_registry.get(self.domain_name, {})

            discovered = []
            for probe_id, probe_info in domain_probes.items():
                # Create ProbeSpec dynamically
                probe = ProbeSpec(
                    probe_id=probe_id,
                    description=probe_info.get("description", ""),
                    execute=self._create_probe_executor(probe_id, probe_info),
                    error_patterns=[],  # LEARNED, not hardcoded
                    reveals=", ".join(probe_info.get("reveals", [])),
                    cost=probe_info.get("cost", 1.0),
                )
                discovered.append(probe)

            self._discovered_probes = discovered
            self._probes_discovered = True

            logger.info(
                f"🔍 Discovered {len(discovered)} probes for {self.domain_name}: "
                f"{[p.probe_id for p in discovered]}"
            )
            return discovered

        except Exception as e:
            logger.warning(f"Probe discovery failed: {e}")
            # Fall back to static probes if discovery fails
            return self.get_probes()

    def _create_probe_executor(
        self, probe_id: str, probe_info: Dict[str, Any]
    ) -> Callable:
        """
        Create a probe executor function for a discovered probe.

        This creates a generic executor that calls the MCP tool.
        Domain strategies can override for custom behavior.
        """

        async def executor(mcp_client: Any, context: Dict[str, Any]) -> ProbeResult:
            """Generic probe executor that calls MCP tool."""
            import logging

            logger = logging.getLogger(f"precept.{self.domain_name}.probe")

            try:
                # Build parameters from context
                params = {}
                for param_name, param_type in probe_info.get("parameters", {}).items():
                    # Try to get parameter from context
                    if param_name in context:
                        params[param_name] = context[param_name]
                    elif param_name == "flight_id":
                        # Common parameter extraction
                        params[param_name] = context.get("entity", "")
                    elif param_name == "port_name":
                        params[param_name] = context.get("entity", "")

                # Call the MCP tool
                response = await mcp_client.call_tool(probe_id, params)
                response_str = str(response) if response else ""
                response_upper = response_str.upper()

                # Check for constraint discovery based on 'reveals' patterns
                for reveal in probe_info.get("reveals", []):
                    if reveal in response_upper:
                        # Determine constraint tier
                        tier = "physics"  # Default
                        if "STALE" in reveal or "EXPIRED" in reveal:
                            tier = "policy"

                        return ProbeResult(
                            success=True,
                            constraint_discovered=reveal,
                            constraint_tier=tier,
                            raw_output=response_str,
                            should_retry=reveal
                            not in ["PHANTOM_INVENTORY", "FARE_EXPIRED"],
                        )

                # No constraint discovered
                return ProbeResult(
                    success=True,
                    raw_output=response_str,
                    should_retry=True,
                )

            except Exception as e:
                logger.warning(f"Probe {probe_id} failed: {e}")
                return ProbeResult(
                    success=False,
                    raw_output=f"Probe failed: {str(e)}",
                    should_retry=True,
                )

        return executor

    def get_probes(self) -> List[ProbeSpec]:
        """
        Return available diagnostic probes for this domain.

        DEPRECATED: Use discover_probes() for dynamic discovery.

        This method is kept for backward compatibility and as fallback
        when discovery fails. Override to provide static probes.
        """
        # Return discovered probes if available
        if self._probes_discovered:
            return self._discovered_probes
        return []

    def should_probe(self, error_code: str, error_message: str) -> Optional[ProbeSpec]:
        """
        Determine if a probe should be triggered for this error.

        NOTE: With dynamic learning, this is deprecated.
        COMPASS controller handles probe selection based on learned patterns.

        Returns the first available probe (for exploration) or None.
        """
        probes = self.get_probes()
        if probes:
            # Return first probe for exploration (COMPASS will learn which is useful)
            return probes[0]
        return None

    async def execute_probe(
        self,
        probe: ProbeSpec,
        mcp_client: Any,
        context: Dict[str, Any],
    ) -> ProbeResult:
        """
        Execute a diagnostic probe and return results.

        Args:
            probe: The probe specification to execute
            mcp_client: MCP client for domain operations
            context: Context including parsed_task, error info, etc.

        Returns:
            ProbeResult with discovered constraints and alternatives
        """
        try:
            result = await probe.execute(mcp_client, context)
            return result
        except Exception as e:
            return ProbeResult(
                success=False,
                raw_output=f"Probe failed: {str(e)}",
                should_retry=True,  # Probe failed, try normal retry
            )


class BaselineDomainStrategy(ABC):
    """
    Abstract base class for baseline strategies.

    Unlike DomainStrategy, baselines:
    - Do NOT learn from errors
    - Do NOT apply learned rules
    - Use RANDOM fallback (or fixed order)
    - May fail if unlucky

    This ensures FAIR comparison between PRECEPT (with learning) and
    baseline (without learning) agents.
    """

    # Same default as DomainStrategy for fair comparison
    DEFAULT_MAX_RETRIES = 4

    def __init__(self, max_retries: int = None):
        """
        Initialize the baseline strategy.

        Args:
            max_retries: Maximum number of retries allowed. If None, uses DEFAULT_MAX_RETRIES.
                        - 1 = near first-try only (minimal exploration)
                        - 4 = standard (aligned with AgentConfig.max_attempts) [default]
        """
        self.max_retries = (
            max_retries if max_retries is not None else self.DEFAULT_MAX_RETRIES
        )

    @property
    def MAX_RETRIES(self) -> int:
        """Property for backward compatibility with existing code using MAX_RETRIES."""
        return self.max_retries

    @property
    @abstractmethod
    def domain_name(self) -> str:
        """Return domain name."""
        pass

    @abstractmethod
    def get_available_options(self) -> List[str]:
        """Return list of options to try (ports, packages, etc.)."""
        pass

    def get_options_for_task(self, parsed_task: ParsedTask) -> List[str]:
        """
        Return options relevant to this specific task type.

        Override in subclass for task-specific options.
        Default: returns all available options.
        """
        return self.get_available_options()

    @abstractmethod
    def parse_task(self, task: str) -> ParsedTask:
        """Parse task into components."""
        pass

    @abstractmethod
    def get_default_option(self, parsed_task: ParsedTask) -> str:
        """Get the default first option to try."""
        pass

    @abstractmethod
    async def execute_action(
        self,
        mcp_client: Any,
        option: str,
        parsed_task: ParsedTask,
    ) -> Tuple[bool, str]:
        """
        Execute action with the given option.

        Returns:
            (success, response)
        """
        pass
