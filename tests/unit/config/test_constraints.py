"""
Unit Tests for precept.config.constraints module.

Tests ConstraintConfig dataclass and related functions.
"""

import pytest

from precept.config.constraints import ConstraintConfig


class TestConstraintConfig:
    """Tests for ConstraintConfig dataclass."""

    def test_default_values(self):
        """Test default configuration values."""
        config = ConstraintConfig()

        assert isinstance(config.transient_indicators, list)
        assert isinstance(config.hard_indicators, list)
        # Note: Soft indicators are not a separate field - constraints are
        # classified as either hard or transient based on these lists

    def test_transient_indicators(self):
        """Test transient constraint indicators."""
        config = ConstraintConfig()

        # Should include temporary issues
        indicators = " ".join(config.transient_indicators).lower()
        assert "timeout" in indicators or "retry" in indicators or "temporary" in indicators

    def test_hard_constraint_indicators(self):
        """Test hard constraint indicators."""
        config = ConstraintConfig()

        # Should include permanent issues
        indicators = " ".join(config.hard_indicators).lower()
        assert "blocked" in indicators or "closed" in indicators or "denied" in indicators

    def test_indicators_not_empty(self):
        """Test that indicators lists are not empty."""
        config = ConstraintConfig()

        # Both lists should have entries
        assert len(config.transient_indicators) > 0
        assert len(config.hard_indicators) > 0

    def test_config_is_dataclass(self):
        """Test that config is a proper dataclass."""
        from dataclasses import is_dataclass

        assert is_dataclass(ConstraintConfig)
