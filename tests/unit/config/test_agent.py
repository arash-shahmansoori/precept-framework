"""
Unit Tests for precept.config.agent module.

Tests AgentConfig dataclass and related functions.
"""

import pytest

from precept.config.agent import AgentConfig


class TestAgentConfig:
    """Tests for AgentConfig dataclass."""

    def test_default_values(self):
        """Test default configuration values."""
        config = AgentConfig()

        assert config.max_retries == 2  # max_retries is used, not max_attempts
        assert config.consolidation_interval == 3
        assert config.compass_evolution_interval == 2
        assert config.failure_threshold == 2
        assert config.enable_llm_reasoning is True
        assert config.force_llm_reasoning is False
        assert config.verbose_llm is False
        assert config.max_internal_workers == 3

    def test_custom_values(self):
        """Test custom configuration values."""
        config = AgentConfig(
            max_retries=5,  # max_retries is used, not max_attempts
            consolidation_interval=10,
            compass_evolution_interval=5,
            failure_threshold=3,
            enable_llm_reasoning=False,
            force_llm_reasoning=True,
            verbose_llm=True,
            max_internal_workers=5,
        )

        assert config.max_retries == 5
        assert config.consolidation_interval == 10
        assert config.enable_llm_reasoning is False
        assert config.force_llm_reasoning is True
        assert config.max_internal_workers == 5

    def test_config_is_dataclass(self):
        """Test that config is a proper dataclass."""
        from dataclasses import is_dataclass

        assert is_dataclass(AgentConfig)

    def test_config_mutability(self):
        """Test that config values can be modified."""
        config = AgentConfig()
        config.max_retries = 10
        config.enable_llm_reasoning = False

        assert config.max_retries == 10
        assert config.enable_llm_reasoning is False
