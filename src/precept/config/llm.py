"""
LLM Configuration for PRECEPT.

Configuration for LLM clients including model selection, temperature, and token limits.

Usage:
    from precept.config.llm import LLMConfig

    config = LLMConfig()
    config = LLMConfig(model="gpt-4", temperature=0.5)
    config = LLMConfig.from_env()
"""

import os
from dataclasses import dataclass


@dataclass
class LLMConfig:
    """Configuration for LLM clients."""

    # Model selection
    model: str = "gpt-4o-mini"

    # Generation parameters
    temperature: float = 0.3
    max_tokens: int = 200

    # Extended token limits for different use cases
    reasoning_max_tokens: int = 300
    reflection_max_tokens: int = 350

    @classmethod
    def from_env(cls) -> "LLMConfig":
        """Load LLM config from environment variables."""
        return cls(
            model=os.getenv("PRECEPT_MODEL", "gpt-4o-mini"),
            temperature=float(os.getenv("PRECEPT_TEMPERATURE", "0.3")),
            max_tokens=int(os.getenv("PRECEPT_MAX_TOKENS", "200")),
        )


__all__ = ["LLMConfig"]
