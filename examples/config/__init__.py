"""
Experiment Configuration Package.

Contains all configuration related to running experiments with PRECEPT.
This is separate from the core PRECEPT configuration in src/precept/config/.

Usage:
    from examples.config import ExperimentConfig, DOMAIN_ICONS
    from examples.config import ExperimentDisplay
"""

from .display import DisplayConfig, ExperimentDisplay, ProgressTracker
from .experiment import (
    DATA_DIR,
    DOMAIN_ICONS,
    PROJECT_ROOT,
    SCENARIO_GENERATORS,
    SERVER_SCRIPT,
    STATS_PATH,
    ExperimentConfig,
)

__all__ = [
    "ExperimentConfig",
    "PROJECT_ROOT",
    "DATA_DIR",
    "STATS_PATH",
    "SERVER_SCRIPT",
    "DOMAIN_ICONS",
    "SCENARIO_GENERATORS",
    "ExperimentDisplay",
    "DisplayConfig",
    "ProgressTracker",
]
