"""
Experiment Configuration for PRECEPT.

This module contains experiment-specific configuration that is separate from
the core PRECEPT configuration. It's placed in the examples/config directory
because it's specific to experiment runs, not to the PRECEPT framework itself.

ISOLATION FOR PARALLEL EXPERIMENTS:
    Set PRECEPT_DATA_DIR environment variable to isolate experiment data.
    This is CRITICAL for running parallel experiments without data leakage.

Usage:
    from examples.config import ExperimentConfig

    # Create from command-line args
    config = ExperimentConfig.from_args(args)

    # Create directly
    config = ExperimentConfig(
        num_train=6,
        num_test=4,
        domain="logistics",
        seed=42,
    )
"""

import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional

# Project paths (relative to this file - examples/config/experiment.py)
PROJECT_ROOT = Path(__file__).parent.parent.parent

# CRITICAL: Support PRECEPT_DATA_DIR env var for experiment isolation
# This prevents race conditions and data leakage in parallel experiments
_env_data_dir = os.environ.get("PRECEPT_DATA_DIR")
if _env_data_dir:
    DATA_DIR = Path(_env_data_dir)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
else:
    DATA_DIR = PROJECT_ROOT / "data"

STATS_PATH = DATA_DIR / "precept_mcp_stats.json"
SERVER_SCRIPT = PROJECT_ROOT / "src" / "precept" / "precept_mcp_server.py"

# Add src to path for imports
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))


@dataclass
class ExperimentConfig:
    """Configuration for experiment runs."""

    # Train/test split
    num_train: int = 6
    num_test: int = 4

    # Concurrency settings
    concurrent_testing: bool = False
    concurrent_training: bool = False
    max_workers: int = 4
    training_workers: int = 2
    agent_internal_workers: int = 3

    # Reproducibility
    seed: Optional[int] = None

    # Domain
    domain: str = "logistics"

    @classmethod
    def from_args(cls, args) -> "ExperimentConfig":
        """Create config from argparse args."""
        return cls(
            num_train=getattr(args, "train", 6),
            num_test=getattr(args, "test", 4),
            concurrent_testing=getattr(args, "concurrent", False),
            concurrent_training=getattr(args, "concurrent_training", False),
            max_workers=getattr(args, "workers", 4),
            training_workers=getattr(args, "training_workers", 2),
            agent_internal_workers=getattr(args, "agent_workers", 3),
            seed=getattr(args, "seed", None),
            domain=getattr(args, "domain", "logistics"),
        )

    def apply_seed(self) -> None:
        """Apply the random seed if set."""
        if self.seed is not None:
            import random

            random.seed(self.seed)
            print(f"🎲 Random seed set to {self.seed} for reproducible experiments")


# Domain icons for display
DOMAIN_ICONS: Dict[str, str] = {
    "logistics": "🚚",
    "coding": "💻",
    "devops": "☁️",
    "finance": "💰",
    "booking": "🏨",
    "integration": "🔗",
}


def _get_scenario_generators() -> Dict[str, Callable]:
    """Lazy load scenario generators to avoid circular imports."""
    from precept import (
        generate_booking_scenarios,
        generate_coding_scenarios,
        generate_devops_scenarios,
        generate_finance_scenarios,
        generate_integration_scenarios,
        generate_logistics_scenarios,
    )

    return {
        "logistics": generate_logistics_scenarios,
        "coding": generate_coding_scenarios,
        "devops": generate_devops_scenarios,
        "finance": generate_finance_scenarios,
        "booking": generate_booking_scenarios,
        "integration": generate_integration_scenarios,
    }


# Lazy-loaded scenario generators
class _ScenarioGeneratorProxy:
    """Proxy class for lazy loading scenario generators."""

    _generators: Optional[Dict[str, Callable]] = None

    def __getitem__(self, key: str) -> Callable:
        if self._generators is None:
            self._generators = _get_scenario_generators()
        return self._generators[key]

    def get(self, key: str, default=None) -> Optional[Callable]:
        if self._generators is None:
            self._generators = _get_scenario_generators()
        return self._generators.get(key, default)

    def keys(self) -> List[str]:
        if self._generators is None:
            self._generators = _get_scenario_generators()
        return list(self._generators.keys())

    def __contains__(self, key: str) -> bool:
        if self._generators is None:
            self._generators = _get_scenario_generators()
        return key in self._generators


SCENARIO_GENERATORS = _ScenarioGeneratorProxy()


__all__ = [
    "ExperimentConfig",
    "PROJECT_ROOT",
    "DATA_DIR",
    "STATS_PATH",
    "SERVER_SCRIPT",
    "DOMAIN_ICONS",
    "SCENARIO_GENERATORS",
]
