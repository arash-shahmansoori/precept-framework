"""
PRECEPT Constraint Classification Module.

Implements DETERMINISTIC PRUNING by:
1. Classifying errors as Hard/Soft/Transient constraints
2. Building a FORBIDDEN list that grows with each failure
3. Injecting constraint context into LLM prompts
4. Mathematically eliminating failed paths from search space

This converts "Probabilistic Persistence" (hoping retry works)
into "Deterministic Pruning" (stopping because it cannot work).

Usage:
    from precept.constraints import RefineInterceptor, ConstraintType

    interceptor = RefineInterceptor()
    constraint = interceptor.add_constraint("Rotterdam", "R-482", "Port blocked")

    if interceptor.is_forbidden("Rotterdam"):
        print("Rotterdam is BANNED - don't retry!")

    # Get injection for LLM prompt
    forbidden_section = interceptor.get_forbidden_injection()
"""

import re
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Dict, List, Optional, Set

if TYPE_CHECKING:
    from .config import ConstraintConfig


def _get_default_constraint_config():
    """Lazy import to avoid circular dependencies."""
    from .config import ConstraintConfig
    return ConstraintConfig()


# =============================================================================
# CONSTRAINT TYPES
# =============================================================================


class ConstraintType(Enum):
    """Classification of error constraints for deterministic pruning."""

    HARD = "hard"  # Permanent - NEVER retry (e.g., port strike, network down)
    SOFT = "soft"  # Might work with different parameters
    TRANSIENT = "transient"  # Might work if retried (rare - network glitch)


# =============================================================================
# CONSTRAINT DATA CLASS
# =============================================================================


@dataclass
class Constraint:
    """A constraint that permanently prunes a solution from the search space."""

    solution: str  # The solution that failed
    error_code: str  # The error code encountered
    constraint_type: ConstraintType  # Hard, Soft, or Transient
    reason: str  # Why this is forbidden
    timestamp: float = field(default_factory=time.time)


# =============================================================================
# PURE FUNCTIONS FOR CONSTRAINT CLASSIFICATION
# =============================================================================


def classify_error(
    error_code: str,
    error_message: str,
    config: Optional["ConstraintConfig"] = None,
) -> ConstraintType:
    """
    Classify an error as Hard, Soft, or Transient constraint.

    TASK-AGNOSTIC: Classification is based purely on semantic analysis
    of the error message, NOT on domain-specific error codes.

    Args:
        error_code: The error code string
        error_message: The error message text
        config: Optional constraint configuration

    Returns:
        ConstraintType indicating how to handle this error
    """
    if config is None:
        config = _get_default_constraint_config()

    message_lower = error_message.lower()

    # Check for transient indicators first
    if any(indicator in message_lower for indicator in config.transient_indicators):
        return ConstraintType.TRANSIENT

    # Check for hard constraint indicators
    if any(indicator in message_lower for indicator in config.hard_indicators):
        return ConstraintType.HARD

    # Default to SOFT constraint
    return ConstraintType.SOFT


def create_constraint(
    solution: str,
    error_code: str,
    error_message: str,
    config: Optional["ConstraintConfig"] = None,
) -> Constraint:
    """
    Create a Constraint from an error.

    This is a pure function that doesn't mutate any state.

    Args:
        solution: The solution that failed
        error_code: The error code
        error_message: The error message
        config: Optional constraint configuration

    Returns:
        A new Constraint object
    """
    constraint_type = classify_error(error_code, error_message, config)
    reason = f"{error_code}: {error_message[:80]}"

    return Constraint(
        solution=solution.lower(),
        error_code=error_code,
        constraint_type=constraint_type,
        reason=reason,
    )


def format_forbidden_injection(constraints: List[Constraint]) -> str:
    """
    Generate the FORBIDDEN context injection for LLM prompt.

    This is the "Kill Switch" - it modifies the context window to
    make certain solutions IMPOSSIBLE to suggest.

    Args:
        constraints: List of constraints to format

    Returns:
        Formatted string for prompt injection
    """
    if not constraints:
        return ""

    lines = ["\n🚫 FORBIDDEN OPTIONS (Probability = 0.0 - DO NOT SUGGEST):"]

    for constraint in constraints:
        forbidden_marker = (
            "❌ BANNED"
            if constraint.constraint_type == ConstraintType.HARD
            else "⚠️ FAILED"
        )
        lines.append(
            f"  {forbidden_marker}: '{constraint.solution}' - {constraint.reason}"
        )

    lines.append(
        "\n⛔ SYSTEM CONSTRAINT: Suggesting any FORBIDDEN option is STRICTLY PROHIBITED."
    )
    lines.append("   You MUST suggest a DIFFERENT solution not in the forbidden list.")

    return "\n".join(lines)


def get_remaining_options(all_options: List[str], forbidden: Set[str]) -> List[str]:
    """
    Get remaining valid options after pruning FORBIDDEN ones.

    This implements SEARCH SPACE PRUNING - failed branches are cut off.

    Args:
        all_options: All available options
        forbidden: Set of forbidden option names (lowercase)

    Returns:
        List of options not in the forbidden set
    """
    return [opt for opt in all_options if opt.lower() not in forbidden]


def suggest_diagnostic_probe(error_code: str, error_message: str) -> Optional[str]:
    """
    Suggest a diagnostic probe instead of a dumb retry.

    TASK-AGNOSTIC EPISTEMIC PROBING:
    Instead of "try again", understand WHY it failed.
    Probes are generic diagnostic actions, not domain-specific.

    Args:
        error_code: The error code
        error_message: The error message

    Returns:
        Suggested diagnostic probe, or None
    """
    probe_map = {
        # Connectivity issues
        r"connection|network|refused|unreachable|dns": "Verify connectivity to target",
        r"timeout|timed out|slow|latency": "Check endpoint responsiveness",
        # Authentication/Authorization
        r"auth|token|expired|credentials|unauthorized": "Verify authentication state",
        r"permission|denied|forbidden|access": "Check authorization/permissions",
        # Resource existence
        r"not found|404|missing|doesn't exist|unknown": "Verify resource exists",
        # Capacity/State
        r"capacity|full|overload|exhausted|busy": "Check resource availability",
        r"locked|in use|busy|conflict": "Check resource state/locks",
        # Configuration
        r"invalid|malformed|format|syntax": "Validate input format",
        r"version|deprecated|obsolete": "Check version compatibility",
    }

    combined = f"{error_code} {error_message}".lower()
    for pattern, probe in probe_map.items():
        if re.search(pattern, combined):
            return probe

    return None


# =============================================================================
# REFINE INTERCEPTOR CLASS
# =============================================================================


@dataclass
class RefineInterceptor:
    """
    The Refine Layer - Semantic firewall between Environment and Agent.

    Implements DETERMINISTIC PRUNING by:
    1. Classifying errors as Hard/Soft/Transient constraints
    2. Building a FORBIDDEN list that grows with each failure
    3. Injecting constraint context into LLM prompts
    4. Mathematically eliminating failed paths from search space

    This converts "Probabilistic Persistence" (hoping retry works)
    into "Deterministic Pruning" (stopping because it cannot work).

    Retry Behavior Flags:
    - soft_constraints_retriable: If True, only HARD errors are pruned;
      SOFT errors (e.g., "server busy") can still be retried.
    """

    # Configuration
    config: "ConstraintConfig" = field(default_factory=_get_default_constraint_config)

    # Accumulated constraints (grows during task execution)
    constraints: List[Constraint] = field(default_factory=list)

    # FORBIDDEN solutions (pruned from search space - probability = 0.0)
    forbidden: Set[str] = field(default_factory=set)

    # Diagnostic probes performed
    probes_performed: List[str] = field(default_factory=list)

    # Retry behavior: If True, SOFT constraints are NOT added to forbidden
    # This allows retrying options that failed with "soft" errors
    soft_constraints_retriable: bool = True

    # Statistics
    stats: Dict[str, int] = field(
        default_factory=lambda: {
            "total_constraints": 0,
            "hard_constraints": 0,
            "soft_constraints": 0,
            "dumb_retries_prevented": 0,
            "diagnostic_probes": 0,
        }
    )

    def classify_error(self, error_code: str, error_message: str) -> ConstraintType:
        """
        Classify an error as Hard, Soft, or Transient constraint.

        Delegates to the pure function for testability.
        """
        return classify_error(error_code, error_message, self.config)

    def add_constraint(
        self,
        solution: str,
        error_code: str,
        error_message: str,
    ) -> Constraint:
        """
        Add a constraint after a failure. This PRUNES the solution from search space.

        TASK-AGNOSTIC: Works with any error code from any environment.
        The probability of suggesting this solution becomes 0.0.
        """
        constraint = create_constraint(
            solution=solution,
            error_code=error_code,
            error_message=error_message,
            config=self.config,
        )

        self.constraints.append(constraint)

        # Update statistics
        self.stats["total_constraints"] += 1
        if constraint.constraint_type == ConstraintType.HARD:
            self.stats["hard_constraints"] += 1
        elif constraint.constraint_type == ConstraintType.SOFT:
            self.stats["soft_constraints"] += 1

        # Determine which constraints to add to FORBIDDEN set
        # - TRANSIENT errors: Never forbidden (can retry same solution)
        # - SOFT errors: Forbidden only if soft_constraints_retriable is False
        # - HARD errors: Always forbidden
        if constraint.constraint_type == ConstraintType.HARD:
            self.forbidden.add(solution.lower())
        elif constraint.constraint_type == ConstraintType.SOFT:
            if not self.soft_constraints_retriable:
                self.forbidden.add(solution.lower())
            # When soft_constraints_retriable=True, SOFT errors are NOT forbidden
            # This allows the agent to retry the same solution later

        return constraint

    def is_forbidden(self, solution: str) -> bool:
        """Check if a solution is in the FORBIDDEN set (probability = 0.0)."""
        return solution.lower() in self.forbidden

    def get_forbidden_injection(self) -> str:
        """
        Generate the FORBIDDEN context injection for LLM prompt.

        This is the "Kill Switch" - it modifies the context window to
        make certain solutions IMPOSSIBLE to suggest.
        """
        return format_forbidden_injection(self.constraints)

    def get_remaining_options(self, all_options: List[str]) -> List[str]:
        """
        Get remaining valid options after pruning FORBIDDEN ones.

        This implements SEARCH SPACE PRUNING - failed branches are cut off.
        """
        return get_remaining_options(all_options, self.forbidden)

    def suggest_diagnostic_probe(
        self, error_code: str, error_message: str
    ) -> Optional[str]:
        """
        Suggest a diagnostic probe instead of a dumb retry.
        """
        probe = suggest_diagnostic_probe(error_code, error_message)
        if probe:
            self.probes_performed.append(probe)
            self.stats["diagnostic_probes"] += 1
        return probe

    def record_prevented_retry(self) -> None:
        """Record that a dumb retry was prevented."""
        self.stats["dumb_retries_prevented"] += 1

    def reset(self) -> None:
        """Reset interceptor for a new task."""
        self.constraints.clear()
        self.forbidden.clear()
        self.probes_performed.clear()

    def get_stats(self) -> Dict[str, int]:
        """Get current statistics."""
        return self.stats.copy()


# =============================================================================
# FACTORY FUNCTIONS
# =============================================================================


def create_refine_interceptor(
    config: Optional["ConstraintConfig"] = None,
    soft_constraints_retriable: bool = True,
) -> RefineInterceptor:
    """
    Factory function to create a RefineInterceptor.

    Args:
        config: Optional constraint configuration
        soft_constraints_retriable: If True, SOFT errors are not added to forbidden set,
            allowing the agent to retry options that failed with soft errors.
            Default: True (more exhaustive search like Full Reflexion)

    Returns:
        Configured RefineInterceptor instance
    """
    if config is None:
        config = _get_default_constraint_config()
    return RefineInterceptor(
        config=config,
        soft_constraints_retriable=soft_constraints_retriable,
    )


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    # Types
    "ConstraintType",
    "Constraint",
    "RefineInterceptor",
    # Pure functions
    "classify_error",
    "create_constraint",
    "format_forbidden_injection",
    "get_remaining_options",
    "suggest_diagnostic_probe",
    # Factory
    "create_refine_interceptor",
]
