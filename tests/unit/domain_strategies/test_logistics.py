"""
Unit Tests for precept.domain_strategies.logistics module.

Tests LogisticsDomainStrategy and LogisticsBaselineStrategy.
"""

import pytest


class TestLogisticsDomainStrategy:
    """Tests for LogisticsDomainStrategy."""

    def test_strategy_import(self):
        """Test that strategy can be imported."""
        from precept.domain_strategies.logistics import LogisticsDomainStrategy

        assert LogisticsDomainStrategy is not None

    def test_strategy_instantiation(self):
        """Test strategy instantiation."""
        from precept.domain_strategies.logistics import LogisticsDomainStrategy

        strategy = LogisticsDomainStrategy()
        assert strategy.domain_name == "logistics"

    def test_get_available_actions(self):
        """Test getting available actions."""
        from precept.domain_strategies.logistics import LogisticsDomainStrategy

        strategy = LogisticsDomainStrategy()
        actions = strategy.get_available_actions()

        assert isinstance(actions, list)
        assert len(actions) > 0
        assert "book_shipment" in actions or "book" in " ".join(actions).lower()

    def test_parse_task(self):
        """Test parsing a logistics task."""
        from precept.domain_strategies.logistics import LogisticsDomainStrategy

        strategy = LogisticsDomainStrategy()
        task = "Book shipment from Rotterdam to Boston"

        parsed = strategy.parse_task(task)

        assert parsed is not None
        assert hasattr(parsed, "action")
        assert hasattr(parsed, "source") or hasattr(parsed, "entity")

    def test_get_system_prompt(self):
        """Test getting system prompt."""
        from precept.domain_strategies.logistics import LogisticsDomainStrategy

        strategy = LogisticsDomainStrategy()
        prompt = strategy.get_system_prompt()

        assert isinstance(prompt, str)
        assert len(prompt) > 0


class TestLogisticsBaselineStrategy:
    """Tests for LogisticsBaselineStrategy."""

    def test_strategy_import(self):
        """Test that baseline strategy can be imported."""
        from precept.domain_strategies.logistics import LogisticsBaselineStrategy

        assert LogisticsBaselineStrategy is not None

    def test_strategy_instantiation(self):
        """Test baseline strategy instantiation."""
        from precept.domain_strategies.logistics import LogisticsBaselineStrategy

        strategy = LogisticsBaselineStrategy()
        assert strategy.domain_name == "logistics"

    def test_get_options_for_task(self):
        """Test getting options for a task."""
        from precept.domain_strategies.logistics import LogisticsBaselineStrategy

        strategy = LogisticsBaselineStrategy()
        task = "Book shipment from Rotterdam to Boston"
        parsed = strategy.parse_task(task)

        options = strategy.get_options_for_task(parsed)

        assert isinstance(options, list)
        assert len(options) > 0

    def test_parse_task(self):
        """Test parsing a task."""
        from precept.domain_strategies.logistics import LogisticsBaselineStrategy

        strategy = LogisticsBaselineStrategy()
        task = "Book shipment from Rotterdam to Boston"

        parsed = strategy.parse_task(task)

        assert parsed is not None


class TestStrategyRegistry:
    """Tests for strategy registry functions."""

    def test_get_domain_strategy(self):
        """Test getting domain strategy from registry."""
        from precept import get_domain_strategy

        strategy = get_domain_strategy("logistics")
        assert strategy is not None
        assert strategy.domain_name == "logistics"

    def test_get_baseline_strategy(self):
        """Test getting baseline strategy from registry."""
        from precept import get_baseline_strategy

        strategy = get_baseline_strategy("logistics")
        assert strategy is not None
        assert strategy.domain_name == "logistics"

    def test_list_available_domains(self):
        """Test listing available domains."""
        from precept import list_available_domains

        domains = list_available_domains()

        assert isinstance(domains, list)
        assert "logistics" in domains
        assert len(domains) >= 1

    def test_invalid_domain_raises(self):
        """Test that invalid domain raises error."""
        from precept import get_domain_strategy

        with pytest.raises((ValueError, KeyError)):
            get_domain_strategy("nonexistent_domain")
