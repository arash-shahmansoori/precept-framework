"""
Unit Tests for precept.config.llm module.

Tests LLMConfig dataclass and related functions.
"""

import pytest

from precept.config.llm import LLMConfig


class TestLLMConfig:
    """Tests for LLMConfig dataclass."""

    def test_default_values(self):
        """Test default configuration values."""
        config = LLMConfig()

        assert config.model == "gpt-4o-mini"
        assert config.temperature == 0.3
        assert config.max_tokens == 200

    def test_custom_values(self):
        """Test custom configuration values."""
        config = LLMConfig(
            model="gpt-4o",
            temperature=0.3,
            max_tokens=1000,
        )

        assert config.model == "gpt-4o"
        assert config.temperature == 0.3
        assert config.max_tokens == 1000

    def test_config_is_dataclass(self):
        """Test that config is a proper dataclass."""
        from dataclasses import is_dataclass

        assert is_dataclass(LLMConfig)
