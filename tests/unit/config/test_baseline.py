"""
Unit Tests for precept.config.baseline module.

Tests BaselineConfig dataclass and related functions.
"""

import pytest

from precept.config.baseline import BaselineConfig


class TestBaselineConfig:
    """Tests for BaselineConfig dataclass."""

    def test_default_values(self):
        """Test default configuration values."""
        config = BaselineConfig()

        assert config.model == "gpt-4o-mini"
        assert config.max_attempts == 3
        assert config.temperature == 0.3
        assert config.max_tokens == 200
        assert config.verbose is False
        assert config.max_internal_workers == 3
        assert config.reflection_max_tokens == 300
        assert config.full_reflexion_max_tokens == 350
        assert config.max_reflections_per_type == 20

    def test_custom_values(self):
        """Test custom configuration values."""
        config = BaselineConfig(
            model="gpt-4o",
            max_attempts=5,
            temperature=0.5,
            max_tokens=500,
            verbose=True,
            max_internal_workers=5,
        )

        assert config.model == "gpt-4o"
        assert config.max_attempts == 5
        assert config.temperature == 0.5
        assert config.max_tokens == 500
        assert config.verbose is True
        assert config.max_internal_workers == 5

    def test_config_is_dataclass(self):
        """Test that config is a proper dataclass."""
        from dataclasses import is_dataclass

        assert is_dataclass(BaselineConfig)
