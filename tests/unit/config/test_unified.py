"""
Unit Tests for precept.config.unified module.

Tests PreceptConfig and configuration factory functions.
"""

from pathlib import Path

import pytest

from precept.config.unified import PreceptConfig, get_default_config


class TestPreceptConfig:
    """Tests for PreceptConfig dataclass."""

    def test_default_values(self):
        """Test default configuration values."""
        config = PreceptConfig()

        assert config.agent is not None
        assert config.baseline is not None
        assert config.llm is not None
        assert config.constraints is not None
        assert config.prompts is not None
        assert config.paths is not None

    def test_nested_configs(self):
        """Test that nested configs are accessible."""
        config = PreceptConfig()

        # Agent config - uses max_retries, not max_attempts
        assert hasattr(config.agent, "max_retries")
        assert hasattr(config.agent, "consolidation_interval")

        # Baseline config - uses max_attempts
        assert hasattr(config.baseline, "model")
        assert hasattr(config.baseline, "max_attempts")

        # LLM config
        assert hasattr(config.llm, "model")
        assert hasattr(config.llm, "temperature")

    def test_server_script_path(self):
        """Test server script path."""
        config = PreceptConfig()

        assert config.server_script is not None
        assert isinstance(config.server_script, Path)
        assert "precept_mcp_server.py" in str(config.server_script)

    def test_modify_nested_config(self):
        """Test modifying nested configuration."""
        config = PreceptConfig()

        config.agent.max_retries = 10
        config.llm.temperature = 0.5

        assert config.agent.max_retries == 10
        assert config.llm.temperature == 0.5


class TestGetDefaultConfig:
    """Tests for get_default_config function."""

    def test_returns_precept_config(self):
        """Test that function returns PreceptConfig."""
        config = get_default_config()
        assert isinstance(config, PreceptConfig)

    def test_default_config_is_valid(self):
        """Test that default config has valid values."""
        config = get_default_config()

        assert config.agent.max_retries > 0  # AgentConfig uses max_retries
        assert config.llm.max_tokens > 0
        assert config.baseline.max_attempts > 0  # BaselineConfig uses max_attempts

    def test_multiple_calls_return_new_instances(self):
        """Test that each call returns a new instance."""
        config1 = get_default_config()
        config2 = get_default_config()

        # Should be different instances
        config1.agent.max_retries = 999
        assert config2.agent.max_retries != 999
