"""
Constraint Configuration for PRECEPT.

Configuration for constraint classification in deterministic pruning.
Defines indicator patterns for classifying errors as hard, soft, or transient.

Usage:
    from precept.config.constraints import ConstraintConfig

    config = ConstraintConfig()
    # Add custom indicators
    config.hard_indicators.append("custom_blocker")
"""

from dataclasses import dataclass, field
from typing import List


@dataclass
class ConstraintConfig:
    """Configuration for constraint classification."""

    # Transient indicators (might work if retried)
    transient_indicators: List[str] = field(
        default_factory=lambda: [
            "timeout",
            "timed out",
            "temporary",
            "retry later",
            "try again",
            "momentarily",
            "intermittent",
            "connection reset",
            "service unavailable",
        ]
    )

    # Hard constraint indicators (permanent for this solution)
    hard_indicators: List[str] = field(
        default_factory=lambda: [
            # Access/Permission
            "blocked",
            "forbidden",
            "denied",
            "not authorized",
            "access denied",
            "permission denied",
            # Existence
            "not found",
            "does not exist",
            "no such",
            "unknown",
            "invalid",
            # Capacity/State
            "suspended",
            "unavailable",
            "closed",
            "offline",
            "shutdown",
            "at capacity",
            "full",
            "exhausted",
            # Incompatibility
            "incompatible",
            "not supported",
            "unsupported",
            "deprecated",
            "rejected",
            "refused",
            # Expiration
            "expired",
            "revoked",
            "cancelled",
        ]
    )


__all__ = ["ConstraintConfig"]
