"""
Path Configuration for PRECEPT.

Centralizes all file path configurations for data storage, scripts, and resources.

ISOLATION FOR PARALLEL EXPERIMENTS:
    Set PRECEPT_DATA_DIR environment variable to isolate experiment data.
    This is CRITICAL for running parallel experiments without data leakage.

    Example:
        export PRECEPT_DATA_DIR=/tmp/precept_exp_001
        python examples/precept_autogen_mcp_full.py ...

Usage:
    from precept.config.paths import DataPaths, get_project_root

    paths = DataPaths()
    print(paths.learned_rules)  # Path to learned rules JSON
"""

import os
from dataclasses import dataclass, field
from pathlib import Path


def get_project_root() -> Path:
    """Get the project root directory."""
    return Path(__file__).parent.parent.parent.parent


def get_data_dir() -> Path:
    """
    Get the data directory.

    Supports PRECEPT_DATA_DIR environment variable for experiment isolation.
    This is CRITICAL for running parallel experiments to avoid:
    - Race conditions on shared ChromaDB
    - Data leakage between experiments
    - Corrupted learned rules from concurrent writes

    Returns:
        Path to data directory (isolated if PRECEPT_DATA_DIR is set)
    """
    # Check for environment variable override (for parallel experiment isolation)
    env_data_dir = os.environ.get("PRECEPT_DATA_DIR")
    if env_data_dir:
        data_path = Path(env_data_dir)
        data_path.mkdir(parents=True, exist_ok=True)
        return data_path
    return get_project_root() / "data"


def get_server_script() -> Path:
    """Get the default MCP server script path."""
    return Path(__file__).parent.parent / "precept_mcp_server.py"


@dataclass
class DataPaths:
    """Centralized data file paths."""

    data_dir: Path = field(default_factory=get_data_dir)

    @property
    def learned_rules(self) -> Path:
        """Path to learned rules JSON file."""
        return self.data_dir / "precept_learned_rules.json"

    @property
    def experiences(self) -> Path:
        """Path to experiences JSON file."""
        return self.data_dir / "precept_experiences.json"

    @property
    def consolidation(self) -> Path:
        """Path to consolidation JSON file."""
        return self.data_dir / "precept_consolidation.json"

    @property
    def mcp_stats(self) -> Path:
        """Path to MCP stats JSON file."""
        return self.data_dir / "precept_mcp_stats.json"

    @property
    def procedures(self) -> Path:
        """Path to procedures JSON file."""
        return self.data_dir / "precept_procedures.json"

    @property
    def domain_mappings(self) -> Path:
        """Path to domain mappings JSON file."""
        return self.data_dir / "precept_domain_mappings.json"

    @property
    def chroma_db(self) -> Path:
        """Path to ChromaDB directory."""
        return self.data_dir / "chroma_precept"

    @property
    def chroma_static_knowledge(self) -> Path:
        """Path to static knowledge ChromaDB directory."""
        return self.data_dir / "chroma_static_knowledge"

    @property
    def static_knowledge(self) -> Path:
        """Path to static knowledge source directory."""
        return self.data_dir / "static_knowledge"

    @property
    def config_dir(self) -> Path:
        """Path to configuration files directory."""
        return self.data_dir / "config"

    @property
    def factual_extraction_config(self) -> Path:
        """Path to factual extraction configuration JSON."""
        return self.config_dir / "factual_extraction.json"


__all__ = [
    "DataPaths",
    "get_project_root",
    "get_data_dir",
    "get_server_script",
]
