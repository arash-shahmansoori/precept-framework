"""
Baseline Configuration for PRECEPT.

Configuration for baseline agents used in fair comparison with PRECEPT.

Usage:
    from precept.config.baseline import BaselineConfig

    config = BaselineConfig()
    config = BaselineConfig(max_attempts=5, verbose=True)
    config = BaselineConfig.from_env()
"""

import os
from dataclasses import dataclass


@dataclass
class BaselineConfig:
    """Configuration for baseline agents."""

    # Retry limits (same as PRECEPT for fair comparison)
    max_attempts: int = 3

    # Internal concurrency
    max_internal_workers: int = 3

    # LLM settings
    model: str = "gpt-4o-mini"
    temperature: float = 0.3
    max_tokens: int = 200
    reflection_max_tokens: int = 300
    full_reflexion_max_tokens: int = 350

    # Reflexion-specific
    max_reflections_per_type: int = 20

    # Feature flags
    verbose: bool = False

    @classmethod
    def from_env(cls) -> "BaselineConfig":
        """Load baseline config from environment variables."""
        return cls(
            max_attempts=int(os.getenv("BASELINE_MAX_ATTEMPTS", "3")),
            max_internal_workers=int(os.getenv("BASELINE_INTERNAL_WORKERS", "3")),
            model=os.getenv("BASELINE_MODEL", "gpt-4o-mini"),
            verbose=os.getenv("BASELINE_VERBOSE", "false").lower() == "true",
        )


__all__ = ["BaselineConfig"]
