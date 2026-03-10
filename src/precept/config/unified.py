"""
Unified PRECEPT Configuration.

Combines all configuration aspects into a single, injectable config object.

Usage:
    from precept.config import PreceptConfig, get_default_config

    # Get default configuration
    config = get_default_config()

    # Create with overrides
    config = PreceptConfig()
    config.llm.model = "gpt-4"
    config.agent.verbose_llm = True

    # Load from environment
    config = PreceptConfig.from_env()
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

from .agent import AgentConfig
from .baseline import BaselineConfig
from .constraints import ConstraintConfig
from .llm import LLMConfig
from .paths import DataPaths, get_server_script
from .prompts import PromptTemplates


@dataclass
class PreceptConfig:
    """
    Unified PRECEPT configuration.

    Combines all configuration aspects into a single, injectable config object.
    """

    # Sub-configurations
    llm: LLMConfig = field(default_factory=LLMConfig)
    agent: AgentConfig = field(default_factory=AgentConfig)
    baseline: BaselineConfig = field(default_factory=BaselineConfig)
    paths: DataPaths = field(default_factory=DataPaths)
    prompts: PromptTemplates = field(default_factory=PromptTemplates)
    constraints: ConstraintConfig = field(default_factory=ConstraintConfig)

    # Server script
    server_script: Optional[Path] = None

    def __post_init__(self):
        """Set default server script if not provided."""
        if self.server_script is None:
            self.server_script = get_server_script()

    @classmethod
    def from_env(cls) -> "PreceptConfig":
        """Load full configuration from environment variables."""
        return cls(
            llm=LLMConfig.from_env(),
            agent=AgentConfig.from_env(),
            baseline=BaselineConfig.from_env(),
        )

    @classmethod
    def default(cls) -> "PreceptConfig":
        """Get default configuration."""
        return cls()

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            "llm": {
                "model": self.llm.model,
                "temperature": self.llm.temperature,
                "max_tokens": self.llm.max_tokens,
            },
            "agent": {
                "max_attempts": self.agent.max_attempts,
                "max_internal_workers": self.agent.max_internal_workers,
                "consolidation_interval": self.agent.consolidation_interval,
                "enable_llm_reasoning": self.agent.enable_llm_reasoning,
            },
            "baseline": {
                "max_attempts": self.baseline.max_attempts,
                "max_internal_workers": self.baseline.max_internal_workers,
            },
        }


def get_default_config() -> PreceptConfig:
    """Get the default PRECEPT configuration."""
    return PreceptConfig.default()


def get_config_from_env() -> PreceptConfig:
    """Load configuration from environment variables."""
    return PreceptConfig.from_env()


def create_agent_config(
    model: str = "gpt-4o-mini",
    max_attempts: int = 3,
    max_internal_workers: int = 3,
    enable_llm_reasoning: bool = True,
    verbose: bool = False,
    **kwargs,
) -> AgentConfig:
    """Create an agent configuration with specified overrides."""
    return AgentConfig(
        max_attempts=max_attempts,
        max_internal_workers=max_internal_workers,
        enable_llm_reasoning=enable_llm_reasoning,
        verbose_llm=verbose,
        **kwargs,
    )


def create_baseline_config(
    model: str = "gpt-4o-mini",
    max_attempts: int = 3,
    max_internal_workers: int = 3,
    verbose: bool = False,
    **kwargs,
) -> BaselineConfig:
    """Create a baseline configuration with specified overrides."""
    return BaselineConfig(
        model=model,
        max_attempts=max_attempts,
        max_internal_workers=max_internal_workers,
        verbose=verbose,
        **kwargs,
    )


__all__ = [
    "PreceptConfig",
    "get_default_config",
    "get_config_from_env",
    "create_agent_config",
    "create_baseline_config",
]
