"""
COMPASS Controller for PRECEPT Framework.

COMPASS = Context-Optimized Metric Planning & Adaptive Strategy System

This is the "System 2" Executive Function that implements:
1. Hierarchical Constraint Prioritization (Physics > Policy > Instruction)
2. Geometric Pruning (eliminate P(success)=0 branches)
3. Temporal Abstraction (maintain strategic context)
4. Epistemic Detour (probe when uncertain)

PRODUCTION-READY DESIGN:
- No hardcoded patterns - all configuration is injectable
- Probes are provided by domain strategies (each domain knows its diagnostics)
- Constraint discovery is learned, not hardcoded
- Full configurability via COMPASSConfig dataclass

Usage:
    from precept.compass_controller import COMPASSController, COMPASSConfig

    config = COMPASSConfig(
        enable_fast_path=True,
        enable_epistemic_probing=True,
        trivial_confidence_threshold=0.85,
    )
    compass = COMPASSController(config=config)

    # Inject domain strategy for probes
    compass.set_domain_strategy(booking_strategy)

    # Check if action is allowed
    decision = compass.evaluate_action(task, parsed_task, complexity)

    if decision.action == COMPASSAction.PROBE:
        # Execute probe provided by domain strategy
        result = await compass.execute_probe(mcp_client, context)
"""

import logging
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Any, Dict, Optional, Tuple

from .csp_constraint_manager import (
    Constraint,
    ConstraintTier,
    CSPConstraintManager,
    ExecutionFeedback,
)

if TYPE_CHECKING:
    from .domain_strategies.base import DomainStrategy, ProbeResult, ProbeSpec

logger = logging.getLogger(__name__)


# =============================================================================
# COMPASS CONFIGURATION
# =============================================================================


@dataclass
class COMPASSConfig:
    """
    Configuration for COMPASS Controller.

    All parameters are configurable for experiments and production tuning.
    No hardcoded defaults - everything is explicit.

    Attributes:
        enable_fast_path: Route trivial tasks to fast path (skip LLM)
        enable_epistemic_probing: Use domain probes for vague errors
        enable_constraint_hierarchy: Apply Physics > Policy > Instruction
        trivial_confidence_threshold: Min confidence for fast path (0.0-1.0)
        min_probe_interval_seconds: Rate limit for probes (avoid spam)
        max_probes_per_task: Maximum probes before falling back to retry
        learn_from_probes: Store probe results in PRECEPT memory
    """

    enable_fast_path: bool = True
    enable_epistemic_probing: bool = True
    enable_constraint_hierarchy: bool = True
    trivial_confidence_threshold: float = 0.8
    min_probe_interval_seconds: float = 1.0
    max_probes_per_task: int = 3
    learn_from_probes: bool = True

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "enable_fast_path": self.enable_fast_path,
            "enable_epistemic_probing": self.enable_epistemic_probing,
            "enable_constraint_hierarchy": self.enable_constraint_hierarchy,
            "trivial_confidence_threshold": self.trivial_confidence_threshold,
            "min_probe_interval_seconds": self.min_probe_interval_seconds,
            "max_probes_per_task": self.max_probes_per_task,
            "learn_from_probes": self.learn_from_probes,
        }


# =============================================================================
# COMPASS DECISION TYPES
# =============================================================================


class COMPASSAction(Enum):
    """Actions COMPASS can recommend."""

    PROCEED = "proceed"  # Execute the action
    BLOCK = "block"  # Physics/Policy blocks this action
    PROBE = "probe"  # Need epistemic detour first
    PIVOT = "pivot"  # Use alternative approach
    FAST_PATH = "fast_path"  # Trivial task, use fast path


@dataclass
class COMPASSDecision:
    """
    A decision from the COMPASS controller.

    Attributes:
        action: What to do (PROCEED, BLOCK, PROBE, PIVOT, FAST_PATH)
        reason: Why this decision was made
        blocking_constraint: If BLOCK, the constraint that blocks
        negotiated_alternative: If PIVOT, the suggested alternative
        probe_spec: If PROBE, the probe to execute (from domain strategy)
        constraint_context: Context to inject into LLM
        complexity_level: trivial/medium/complex
        should_skip_llm: Whether to skip LLM reasoning
    """

    action: COMPASSAction
    reason: str
    blocking_constraint: Optional[str] = None
    negotiated_alternative: Optional[str] = None
    probe_spec: Optional["ProbeSpec"] = None
    constraint_context: str = ""
    complexity_level: str = "medium"
    should_skip_llm: bool = False


# =============================================================================
# LEARNED CONSTRAINT PATTERNS
# =============================================================================


@dataclass
class LearnedPattern:
    """
    A pattern learned from experience.

    Instead of hardcoding "BK-" → probe inventory, we learn:
    "When error contains 'phantom', probing inventory helps"
    """

    pattern: str  # The pattern that triggers this
    action: str  # "probe", "block", "pivot"
    probe_id: Optional[str] = None  # If action=probe, which probe
    alternative: Optional[str] = None  # If action=pivot, the alternative
    success_count: int = 0  # How often this helped
    failure_count: int = 0  # How often this failed
    confidence: float = 0.5  # success_count / (success + failure)

    def update(self, succeeded: bool) -> None:
        """Update pattern confidence based on outcome."""
        if succeeded:
            self.success_count += 1
        else:
            self.failure_count += 1
        total = self.success_count + self.failure_count
        self.confidence = self.success_count / total if total > 0 else 0.5


# =============================================================================
# COMPASS CONTROLLER
# =============================================================================


class COMPASSController:
    """
    The COMPASS Executive Controller.

    Acts as "System 2" - the strategic layer that decides:
    1. WHETHER to execute (constraint checking)
    2. WHAT strategy to use (complexity-based routing)
    3. WHEN to probe (epistemic detour trigger)
    4. HOW to negotiate (alternative suggestions)

    PRODUCTION DESIGN:
    - Probes come from domain strategies (not hardcoded)
    - Patterns are learned from experience (not hardcoded)
    - All configuration is injectable
    - Statistics track actual benefit
    """

    def __init__(
        self,
        config: Optional[COMPASSConfig] = None,
    ):
        """
        Initialize the COMPASS controller.

        Args:
            config: Optional configuration. Uses defaults if not provided.
        """
        self.config = config or COMPASSConfig()

        # Components
        self.csp_manager = CSPConstraintManager()

        # Domain strategy (provides probes) - injected later
        self._domain_strategy: Optional["DomainStrategy"] = None

        # Learned patterns (from experience, not hardcoded)
        self._learned_patterns: Dict[str, LearnedPattern] = {}

        # Dynamic probe learning state
        # Tracks which probes we've tried for each error type (for exploration)
        self._probes_tried_for_error: Dict[str, set] = {}

        # Probe discovery state
        self._probes_discovered: bool = False

        # State
        self._probes_this_task: int = 0
        self._last_probe_time: float = 0.0
        self.user_instruction: Optional[str] = None

        # Statistics for evaluation
        self.stats = {
            "fast_path_used": 0,
            "full_path_used": 0,
            "constraints_blocked": 0,
            "probes_discovered": 0,  # Track dynamic discovery
            "probes_triggered": 0,
            "probes_successful": 0,
            "probes_failed": 0,
            "pivots_negotiated": 0,
            "physics_overrides": 0,
            "patterns_learned": 0,
        }

    # =========================================================================
    # CONFIGURATION & DEPENDENCY INJECTION
    # =========================================================================

    def set_domain_strategy(self, strategy: "DomainStrategy") -> None:
        """
        Inject domain strategy for probe capabilities.

        Args:
            strategy: The domain strategy that provides probes
        """
        self._domain_strategy = strategy
        self._probes_discovered = False  # Reset discovery state
        logger.debug(f"COMPASS: Domain strategy set to {strategy.domain_name}")

    async def discover_probes(self, mcp_client: Any) -> None:
        """
        Discover available probes from the MCP server.

        This is called once to populate the probe catalog.
        The agent doesn't know a priori what probes exist - it discovers them.

        Args:
            mcp_client: MCP client to query for available probes
        """
        if not self._domain_strategy:
            logger.warning("COMPASS: No domain strategy set, cannot discover probes")
            return

        if self._probes_discovered:
            return  # Already discovered

        try:
            probes = await self._domain_strategy.discover_probes(mcp_client)
            self._probes_discovered = True
            self.stats["probes_discovered"] = len(probes)
            logger.info(
                f"🔍 COMPASS discovered {len(probes)} probes: "
                f"{[p.probe_id for p in probes]}"
            )
        except Exception as e:
            logger.warning(f"COMPASS probe discovery failed: {e}")

    def set_user_instruction(self, instruction: str) -> None:
        """Set the current user instruction (for conflict resolution)."""
        self.user_instruction = instruction

    def get_config(self) -> COMPASSConfig:
        """Get current configuration."""
        return self.config

    # =========================================================================
    # MAIN DECISION POINTS
    # =========================================================================

    def evaluate_action(
        self,
        task: str,
        parsed_task: Any,
        complexity: Dict[str, Any],
    ) -> COMPASSDecision:
        """
        Evaluate an action BEFORE execution.

        This is the main COMPASS decision point.

        Returns:
            COMPASSDecision with action and context
        """
        # Get constraint context for LLM
        constraint_context = self.csp_manager.get_llm_context()

        # ═══════════════════════════════════════════════════════════════════
        # STEP 1: Check Hierarchical Constraints (Physics > Policy > Instruction)
        # ═══════════════════════════════════════════════════════════════════
        if self.config.enable_constraint_hierarchy:
            blocking = self._check_blocking_constraints(parsed_task)
            if blocking:
                constraint_id, tier, negotiated = blocking

                if tier == ConstraintTier.PHYSICS:
                    self.stats["physics_overrides"] += 1
                    self.stats["constraints_blocked"] += 1

                    if negotiated:
                        self.stats["pivots_negotiated"] += 1
                        return COMPASSDecision(
                            action=COMPASSAction.PIVOT,
                            reason=f"PHYSICS constraint '{constraint_id}' blocks action",
                            blocking_constraint=constraint_id,
                            negotiated_alternative=negotiated,
                            constraint_context=constraint_context,
                            complexity_level=complexity.get("level", "medium"),
                        )
                    else:
                        return COMPASSDecision(
                            action=COMPASSAction.BLOCK,
                            reason=f"PHYSICS constraint '{constraint_id}' - no alternative",
                            blocking_constraint=constraint_id,
                            constraint_context=constraint_context,
                            complexity_level=complexity.get("level", "medium"),
                        )

        # ═══════════════════════════════════════════════════════════════════
        # STEP 2: Complexity-Based Routing (Fast Path for Trivial Tasks)
        # ═══════════════════════════════════════════════════════════════════
        if self.config.enable_fast_path:
            complexity_level = complexity.get("level", "medium")
            confidence = complexity.get("confidence", 0.5)

            if (
                complexity_level == "trivial"
                and confidence >= self.config.trivial_confidence_threshold
            ):
                self.stats["fast_path_used"] += 1
                return COMPASSDecision(
                    action=COMPASSAction.FAST_PATH,
                    reason="Trivial task with high confidence - using fast path",
                    constraint_context=constraint_context,
                    complexity_level="trivial",
                    should_skip_llm=True,
                )

        # ═══════════════════════════════════════════════════════════════════
        # STEP 3: Default - Proceed with Full PRECEPT Pipeline
        # ═══════════════════════════════════════════════════════════════════
        self.stats["full_path_used"] += 1
        return COMPASSDecision(
            action=COMPASSAction.PROCEED,
            reason="No blocking constraints, proceeding with full PRECEPT pipeline",
            constraint_context=constraint_context,
            complexity_level=complexity.get("level", "medium"),
        )

    def evaluate_error(
        self,
        error_code: str,
        error_message: str,
        context: Dict[str, Any] = None,
    ) -> COMPASSDecision:
        """
        Evaluate an error to decide next action.

        This implements:
        - Epistemic detour (probe vague errors using domain probes)
        - Constraint discovery (update constraint map from feedback)
        - Negotiation (suggest alternatives)
        """
        context = context or {}

        # ═══════════════════════════════════════════════════════════════════
        # STEP 1: Should we probe? (Epistemic Detour)
        # ═══════════════════════════════════════════════════════════════════
        if self.config.enable_epistemic_probing:
            probe_decision = self._check_should_probe(error_code, error_message)
            if probe_decision:
                return probe_decision

        # ═══════════════════════════════════════════════════════════════════
        # STEP 2: Discover Constraint from Error
        # ═══════════════════════════════════════════════════════════════════
        feedback = ExecutionFeedback(
            return_code=1,
            stdout="",
            stderr=error_message,
            duration=0.0,
            tool_name="action",
        )
        discovered = self.csp_manager.intercept_feedback(feedback)

        if discovered:
            constraint = discovered[0]

            # Check if this is a physics constraint
            if constraint.tier == ConstraintTier.PHYSICS:
                # Try to negotiate an alternative
                negotiated = self._negotiate_alternative(
                    constraint, self.user_instruction
                )

                if negotiated:
                    self.stats["pivots_negotiated"] += 1
                    return COMPASSDecision(
                        action=COMPASSAction.PIVOT,
                        reason=f"Discovered PHYSICS constraint: {constraint.id}",
                        blocking_constraint=constraint.id,
                        negotiated_alternative=negotiated,
                        constraint_context=self.csp_manager.get_llm_context(),
                    )
                else:
                    self.stats["constraints_blocked"] += 1
                    return COMPASSDecision(
                        action=COMPASSAction.BLOCK,
                        reason=f"PHYSICS constraint discovered: {constraint.id}",
                        blocking_constraint=constraint.id,
                        constraint_context=self.csp_manager.get_llm_context(),
                    )

        # ═══════════════════════════════════════════════════════════════════
        # STEP 3: No Special Handling - Proceed with Retry Loop
        # ═══════════════════════════════════════════════════════════════════
        return COMPASSDecision(
            action=COMPASSAction.PROCEED,
            reason="Error classified as recoverable, proceeding with retry loop",
            constraint_context=self.csp_manager.get_llm_context(),
        )

    async def execute_probe(
        self,
        mcp_client: Any,
        probe_spec: "ProbeSpec",
        context: Dict[str, Any],
    ) -> "ProbeResult":
        """
        Execute a diagnostic probe from the domain strategy.

        Args:
            mcp_client: MCP client for domain operations
            probe_spec: The probe specification to execute
            context: Context including parsed_task, error info, etc.

        Returns:
            ProbeResult with discovered constraints and alternatives
        """
        from .domain_strategies.base import ProbeResult

        if not self._domain_strategy:
            logger.warning("COMPASS: No domain strategy set, cannot execute probe")
            return ProbeResult(
                success=False,
                raw_output="No domain strategy configured",
                should_retry=True,
            )

        try:
            result = await self._domain_strategy.execute_probe(
                probe_spec, mcp_client, context
            )

            # Extract error code from context for learning
            error_code = context.get("error_code", "")

            if result.success:
                self.stats["probes_successful"] += 1

                # ═══════════════════════════════════════════════════════════
                # DYNAMIC PROBE LEARNING: Learn error→probe mapping
                # ═══════════════════════════════════════════════════════════
                if self.config.learn_from_probes and result.constraint_discovered:
                    # This probe revealed useful information for this error!
                    # Learn the mapping: error_code → probe_id
                    self.learn_pattern(
                        error_pattern=self._get_error_key(error_code),
                        action="probe",
                        succeeded=True,
                        probe_id=probe_spec.probe_id,
                    )
                    logger.info(
                        f"📚 LEARNED: {error_code} → {probe_spec.probe_id} reveals "
                        f"{result.constraint_discovered}"
                    )
                    self._learn_constraint_from_probe(result)
                elif self.config.learn_from_probes:
                    # Probe succeeded but didn't reveal constraint
                    # This probe might not be useful for this error
                    self.learn_pattern(
                        error_pattern=self._get_error_key(error_code),
                        action="probe",
                        succeeded=False,
                        probe_id=probe_spec.probe_id,
                    )
            else:
                self.stats["probes_failed"] += 1

            return result

        except Exception as e:
            logger.error(f"COMPASS: Probe execution failed: {e}")
            self.stats["probes_failed"] += 1
            return ProbeResult(
                success=False,
                raw_output=f"Probe execution error: {str(e)}",
                should_retry=True,
            )

    def process_probe_result(self, result: "ProbeResult") -> COMPASSDecision:
        """
        Process the result of an epistemic probe.

        NOTE: We deliberately DO NOT add probe-discovered constraints to the
        CSP manager's blocking list. This is because:
        1. Probe constraints are entity-specific (e.g., flight AA-999 has phantom inventory)
        2. Adding generic constraints like "PHANTOM_INVENTORY" would block ALL future tasks
        3. The probe knowledge should INFORM the retry loop, not PRE-EMPTIVELY block

        The domain strategy handles entity-specific blocking via _learned_blocked_flights.
        COMPASS provides the constraint context to the LLM for informed decision-making.
        """
        if result.constraint_discovered:
            tier = self._parse_constraint_tier(result.constraint_tier)

            if tier == ConstraintTier.PHYSICS:
                self.stats["physics_overrides"] += 1
                logger.info(
                    f"COMPASS: Discovered constraint {result.constraint_discovered} (tier: physics)"
                )

                if result.negotiated_alternative:
                    self.stats["pivots_negotiated"] += 1
                    return COMPASSDecision(
                        action=COMPASSAction.PIVOT,
                        reason=f"Probe revealed PHYSICS: {result.constraint_discovered}",
                        blocking_constraint=result.constraint_discovered,
                        negotiated_alternative=result.negotiated_alternative,
                        constraint_context=f"Constraint: {result.constraint_discovered} (physics)",
                    )
                else:
                    # No alternative known yet - let retry loop explore
                    logger.info(
                        f"COMPASS: Constraint {result.constraint_discovered} discovered, "
                        "proceeding with exploration to find alternatives"
                    )
                    return COMPASSDecision(
                        action=COMPASSAction.PROCEED,
                        reason=f"Probe revealed PHYSICS: {result.constraint_discovered} - explore alternatives",
                        blocking_constraint=result.constraint_discovered,
                        constraint_context=f"Constraint: {result.constraint_discovered} (physics) - try alternatives",
                    )

        # Transient issue or no constraint found - proceed with retry
        return COMPASSDecision(
            action=COMPASSAction.PROCEED,
            reason="Probe completed - proceeding with retry",
            constraint_context=self.csp_manager.get_llm_context(),
        )

    # =========================================================================
    # LEARNING FROM EXPERIENCE
    # =========================================================================

    def learn_pattern(
        self,
        error_pattern: str,
        action: str,
        succeeded: bool,
        probe_id: Optional[str] = None,
        alternative: Optional[str] = None,
    ) -> None:
        """
        Learn a pattern from experience.

        Args:
            error_pattern: The error pattern (e.g., "phantom", "timeout")
            action: What action was taken ("probe", "pivot", "block")
            succeeded: Whether the action led to success
            probe_id: If action=probe, which probe was used
            alternative: If action=pivot, what alternative was used
        """
        if error_pattern not in self._learned_patterns:
            self._learned_patterns[error_pattern] = LearnedPattern(
                pattern=error_pattern,
                action=action,
                probe_id=probe_id,
                alternative=alternative,
            )
            self.stats["patterns_learned"] += 1

        self._learned_patterns[error_pattern].update(succeeded)

    def get_learned_pattern(
        self, error_code: str, error_message: str
    ) -> Optional[LearnedPattern]:
        """
        Get a learned pattern that matches this error.

        Returns the highest-confidence matching pattern, or None.
        """
        combined = f"{error_code} {error_message}".lower()
        best_match: Optional[LearnedPattern] = None
        best_confidence = 0.0

        for pattern, learned in self._learned_patterns.items():
            if pattern in combined and learned.confidence > best_confidence:
                best_match = learned
                best_confidence = learned.confidence

        return best_match

    # =========================================================================
    # ACCESSORS
    # =========================================================================

    def get_constraint_context(self) -> str:
        """Get current constraint context for LLM injection."""
        return self.csp_manager.get_llm_context()

    def get_stats(self) -> Dict[str, int]:
        """Get COMPASS statistics."""
        return {
            **self.stats,
            **self.csp_manager.stats,
            "patterns_count": len(self._learned_patterns),
        }

    def reset_task(self) -> None:
        """Reset per-task state. Call at start of each task."""
        self._probes_this_task = 0
        self.user_instruction = None

    def reset(self) -> None:
        """Full reset (for testing). Clears all learned patterns."""
        self.reset_task()
        self._learned_patterns.clear()
        self._probes_tried_for_error.clear()
        self.stats = {key: 0 for key in self.stats}

    # =========================================================================
    # PRIVATE METHODS
    # =========================================================================

    def _check_should_probe(
        self, error_code: str, error_message: str
    ) -> Optional[COMPASSDecision]:
        """
        Check if we should trigger an epistemic probe.

        DYNAMIC PROBE LEARNING:
        1. EXPLOIT: If we've learned which probe works for this error → use it
        2. EXPLORE: If this is a new error → try probes to discover what helps
        3. LEARN: Track which probes reveal useful constraints for which errors

        This is the key to avoiding hardcoded error_patterns.
        """
        import time

        if not self._domain_strategy:
            return None

        # Rate limiting
        if self._probes_this_task >= self.config.max_probes_per_task:
            logger.debug("COMPASS: Max probes reached for this task")
            return None

        now = time.time()
        if now - self._last_probe_time < self.config.min_probe_interval_seconds:
            logger.debug("COMPASS: Probe rate limited")
            return None

        available_probes = self._domain_strategy.get_probes()
        if not available_probes:
            return None

        # ═══════════════════════════════════════════════════════════════════
        # PHASE 1: EXPLOIT - Use learned knowledge if available
        # ═══════════════════════════════════════════════════════════════════
        learned = self.get_learned_pattern(error_code, error_message)
        if learned and learned.confidence > 0.6 and learned.action == "probe":
            # We've learned this error benefits from a specific probe
            for probe in available_probes:
                if probe.probe_id == learned.probe_id:
                    self._probes_this_task += 1
                    self._last_probe_time = now
                    self.stats["probes_triggered"] += 1
                    logger.info(
                        f"🎯 EXPLOIT: Using learned probe {probe.probe_id} "
                        f"for {error_code} (confidence: {learned.confidence:.2f})"
                    )
                    return COMPASSDecision(
                        action=COMPASSAction.PROBE,
                        reason=f"Learned: {probe.probe_id} helps for {error_code}",
                        probe_spec=probe,
                        constraint_context=self.csp_manager.get_llm_context(),
                    )

        # ═══════════════════════════════════════════════════════════════════
        # PHASE 2: EXPLORE - Try probes to learn what helps
        # ═══════════════════════════════════════════════════════════════════
        # For new errors, we need to explore which probe reveals useful info.
        # Strategy: Try probes in order of cost (low cost first)

        # Check if this error type is "vague" enough to warrant probing
        # Vague errors have generic codes that don't reveal the root cause
        error_lower = f"{error_code} {error_message}".lower()
        is_vague_error = any(
            indicator in error_lower
            for indicator in [
                "failed",
                "error",
                "unavailable",
                "timeout",
                "unable",
                "bk-",  # Booking domain
                "lg-",  # Logistics domain
                "pkg-",  # Coding domain
            ]
        )

        if is_vague_error:
            # Get probes we haven't tried for this error yet
            error_key = self._get_error_key(error_code)
            tried_probes = self._probes_tried_for_error.get(error_key, set())

            # Sort by cost (try cheap probes first)
            untried_probes = [
                p for p in available_probes if p.probe_id not in tried_probes
            ]
            untried_probes.sort(key=lambda p: p.cost)

            if untried_probes:
                probe = untried_probes[0]
                self._probes_this_task += 1
                self._last_probe_time = now
                self.stats["probes_triggered"] += 1

                # Track that we're trying this probe for this error
                if error_key not in self._probes_tried_for_error:
                    self._probes_tried_for_error[error_key] = set()
                self._probes_tried_for_error[error_key].add(probe.probe_id)

                logger.info(
                    f"🔍 EXPLORE: Trying probe {probe.probe_id} for new error {error_code}"
                )
                return COMPASSDecision(
                    action=COMPASSAction.PROBE,
                    reason=f"Exploring: {probe.probe_id} for {error_code}",
                    probe_spec=probe,
                    constraint_context=self.csp_manager.get_llm_context(),
                )

        return None

    def _get_error_key(self, error_code: str) -> str:
        """Extract error key for learning (e.g., 'BK-401' → 'BK-4xx')."""
        # Normalize error codes to learn patterns, not exact codes
        # BK-401, BK-402 → both map to 'BK-4' pattern
        if "-" in error_code:
            prefix, suffix = error_code.split("-", 1)
            if suffix and suffix[0].isdigit():
                return f"{prefix}-{suffix[0]}"
        return error_code.lower()

    def _check_blocking_constraints(
        self, parsed_task: Any
    ) -> Optional[Tuple[str, ConstraintTier, Optional[str]]]:
        """
        Check if any active constraint blocks this action.

        Returns:
            Tuple of (constraint_id, tier, negotiated_alternative) or None
        """
        active = self.csp_manager.refine_interceptor.get_active_constraints()

        for constraint_id in active:
            constraint = self.csp_manager.constraints.get(constraint_id)
            if constraint and constraint.tier == ConstraintTier.PHYSICS:
                # Physics constraint - check if it blocks this action
                action_type = getattr(parsed_task, "action", None)
                if action_type:
                    negotiated = self._negotiate_alternative(constraint, action_type)
                    return constraint_id, constraint.tier, negotiated

        return None

    def _negotiate_alternative(
        self, constraint: Constraint, user_goal: Optional[str]
    ) -> Optional[str]:
        """Negotiate an alternative when a constraint blocks the goal."""
        if not user_goal:
            return None

        # Check if constraint has solution patterns
        if constraint.solution_patterns:
            return constraint.solution_patterns[0]

        # Check learned patterns for alternatives
        for pattern in self._learned_patterns.values():
            if pattern.alternative and pattern.confidence > 0.6:
                return pattern.alternative

        return None

    def _learn_constraint_from_probe(self, result: "ProbeResult") -> None:
        """Learn from a successful probe result."""
        if result.constraint_discovered:
            # This could be extended to update PRECEPT's memory
            logger.info(
                f"COMPASS: Learned constraint {result.constraint_discovered} "
                f"(tier: {result.constraint_tier})"
            )

    def _parse_constraint_tier(self, tier_str: Optional[str]) -> ConstraintTier:
        """Parse constraint tier string to enum."""
        if not tier_str:
            return ConstraintTier.INSTRUCTION

        tier_lower = tier_str.lower()
        if "physics" in tier_lower:
            return ConstraintTier.PHYSICS
        elif "policy" in tier_lower:
            return ConstraintTier.POLICY
        else:
            return ConstraintTier.INSTRUCTION


# =============================================================================
# FACTORY FUNCTION
# =============================================================================


def create_compass_controller(
    config: Optional[COMPASSConfig] = None,
    domain_strategy: Optional["DomainStrategy"] = None,
) -> COMPASSController:
    """
    Create a COMPASS controller with the specified configuration.

    Args:
        config: Optional configuration. Uses defaults if not provided.
        domain_strategy: Optional domain strategy for probes.

    Returns:
        Configured COMPASSController
    """
    controller = COMPASSController(config=config)
    if domain_strategy:
        controller.set_domain_strategy(domain_strategy)
    return controller
