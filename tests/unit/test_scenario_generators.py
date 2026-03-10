"""
Unit Tests for precept.scenario_generators module.

Tests scenario generation including:
- Logistics scenario generation
- Conflict resolution scenarios
- Coverage guarantees
- Phase assignment (training/test)
"""

import pytest
from typing import Dict, List

from precept.scenario_generators import (
    generate_logistics_scenarios,
)
from precept.scenario_generators.logistics import (
    LogisticsScenarioGenerator,
    LogisticsScenarioConfig,
)


# =============================================================================
# LOGISTICS SCENARIO GENERATOR TESTS
# =============================================================================


class TestLogisticsScenarioGenerator:
    """Tests for LogisticsScenarioGenerator."""

    @pytest.fixture
    def generator(self):
        """Create a generator for testing."""
        return LogisticsScenarioGenerator(num_samples=20, train_ratio=0.6)

    def test_generator_initialization(self, generator):
        """Test generator initializes correctly."""
        assert generator.num_samples == 20
        assert generator.train_ratio == 0.6
        assert generator.config == LogisticsScenarioConfig

    def test_generate_all_returns_scenarios(self, generator):
        """Test generate_all returns scenarios."""
        scenarios = generator.generate_all()

        assert isinstance(scenarios, list)
        assert len(scenarios) > 0

    def test_scenarios_have_required_fields(self, generator):
        """Test all scenarios have required fields."""
        scenarios = generator.generate_all()

        required_fields = ["task", "expected", "black_swan_type", "precept_lesson"]

        for scenario in scenarios:
            for field in required_fields:
                assert field in scenario, f"Missing field: {field}"

    def test_scenarios_have_phase(self, generator):
        """Test scenarios are assigned training/test phase."""
        scenarios = generator.generate_all()

        phases = set(s.get("phase") for s in scenarios)
        assert "training" in phases or "test" in phases

    def test_train_test_split(self, generator):
        """Test train/test split follows ratio."""
        scenarios = generator.generate_all()

        training = [s for s in scenarios if s.get("phase") == "training"]
        test = [s for s in scenarios if s.get("phase") == "test"]

        # Should have both phases
        assert len(training) > 0
        assert len(test) > 0

        # Training should be roughly train_ratio of total
        total = len(training) + len(test)
        actual_ratio = len(training) / total
        assert 0.4 <= actual_ratio <= 0.8  # Allow some flexibility


class TestLogisticsConflictResolutionScenarios:
    """Tests for conflict resolution scenario generation."""

    @pytest.fixture
    def generator(self):
        """Create a generator for testing."""
        return LogisticsScenarioGenerator(num_samples=20, train_ratio=0.6)

    def test_conflict_scenarios_generated(self, generator):
        """Test conflict resolution scenarios are generated."""
        scenarios = generator.generate_conflict_resolution_scenarios(
            num_training=4,
            num_test=4,
        )

        assert len(scenarios) > 0

    def test_conflict_scenarios_have_conflict_type(self, generator):
        """Test conflict scenarios have conflict_type field."""
        scenarios = generator.generate_conflict_resolution_scenarios(
            num_training=4,
            num_test=4,
        )

        conflict_types = set()
        for s in scenarios:
            if "conflict_type" in s:
                conflict_types.add(s["conflict_type"])

        # Should have multiple conflict types
        expected_types = {
            "dynamic_should_override",
            "static_should_win",
            "dynamic_completes",
            "agreement",
        }

        # At least some of the expected types should be present
        assert len(conflict_types.intersection(expected_types)) >= 1

    def test_conflict_training_scenarios(self, generator):
        """Test conflict training scenarios are generated."""
        scenarios = generator.generate_conflict_resolution_scenarios(
            num_training=4,
            num_test=2,
        )

        training = [s for s in scenarios if s.get("phase") == "training"]
        assert len(training) > 0

    def test_conflict_test_scenarios(self, generator):
        """Test conflict test scenarios are generated."""
        scenarios = generator.generate_conflict_resolution_scenarios(
            num_training=2,
            num_test=4,
        )

        test = [s for s in scenarios if s.get("phase") == "test"]
        assert len(test) > 0

    def test_conflict_scenarios_have_static_knowledge_tested(self, generator):
        """Test training conflict scenarios reference static knowledge."""
        scenarios = generator.generate_conflict_resolution_scenarios(
            num_training=4,
            num_test=2,
        )

        training = [s for s in scenarios if s.get("phase") == "training"]

        # At least some should have static_knowledge_tested
        has_static_ref = any(
            "static_knowledge_tested" in s for s in training
        )
        assert has_static_ref


class TestGenerateLogisticsScenarios:
    """Tests for generate_logistics_scenarios function."""

    def test_basic_generation(self):
        """Test basic scenario generation."""
        scenarios = generate_logistics_scenarios(num_samples=10, train_ratio=0.6)

        assert isinstance(scenarios, list)
        assert len(scenarios) >= 10

    def test_include_conflict_resolution_default(self):
        """Test conflict resolution is included by default."""
        scenarios = generate_logistics_scenarios(
            num_samples=15,
            train_ratio=0.6,
            include_conflict_resolution=True,
        )

        # Should have some conflict scenarios
        conflict_scenarios = [
            s for s in scenarios if s.get("conflict_type") is not None
        ]
        assert len(conflict_scenarios) >= 1

    def test_exclude_conflict_resolution(self):
        """Test conflict resolution can be excluded."""
        scenarios = generate_logistics_scenarios(
            num_samples=10,
            train_ratio=0.6,
            include_conflict_resolution=False,
        )

        # Should have no conflict scenarios (or very few from other sources)
        conflict_scenarios = [
            s for s in scenarios if s.get("conflict_type") is not None
        ]
        assert len(conflict_scenarios) == 0


class TestLogisticsScenarioConfig:
    """Tests for LogisticsScenarioConfig."""

    def test_blocked_ports_defined(self):
        """Test blocked ports are defined."""
        assert hasattr(LogisticsScenarioConfig, "BLOCKED_PORTS")
        assert len(LogisticsScenarioConfig.BLOCKED_PORTS) > 0

    def test_destinations_defined(self):
        """Test destinations are defined."""
        assert hasattr(LogisticsScenarioConfig, "DESTINATIONS")
        assert len(LogisticsScenarioConfig.DESTINATIONS) > 0

    def test_cargo_types_defined(self):
        """Test cargo types are defined."""
        assert hasattr(LogisticsScenarioConfig, "CARGO_TYPES")
        assert len(LogisticsScenarioConfig.CARGO_TYPES) > 0

    def test_blocked_ports_have_required_fields(self):
        """Test blocked ports have required configuration."""
        for port, config in LogisticsScenarioConfig.BLOCKED_PORTS.items():
            assert "error_code" in config
            assert "working_alternatives" in config
            assert isinstance(config["working_alternatives"], list)


class TestCoverageGuarantee:
    """Tests for coverage guarantee functionality."""

    def test_coverage_guarantee_covers_all_error_types(self):
        """Test that coverage guarantee mode covers all error types."""
        generator = LogisticsScenarioGenerator(num_samples=20, train_ratio=0.6)
        scenarios = generator.generate_all(ensure_coverage=True)

        training = [s for s in scenarios if s.get("phase") == "training"]

        # Extract error codes from training scenarios
        error_codes = set()
        for s in training:
            expected = s.get("expected", "")
            if "→" in expected:
                code = expected.split("→")[0].strip()
                error_codes.add(code)

        # Should cover multiple error types
        assert len(error_codes) >= 4  # At least 4 different error types


class TestPortClosureScenarios:
    """Tests for port closure scenario generation."""

    @pytest.fixture
    def generator(self):
        """Create a generator for testing."""
        return LogisticsScenarioGenerator(num_samples=10, train_ratio=0.6)

    def test_port_closure_scenarios_generated(self, generator):
        """Test port closure scenarios are generated."""
        scenarios = generator.generate_port_closure_scenarios(
            num_training=3,
            num_test=2,
        )

        assert len(scenarios) > 0

    def test_port_closure_has_correct_type(self, generator):
        """Test port closure scenarios have correct type."""
        scenarios = generator.generate_port_closure_scenarios(
            num_training=3,
            num_test=2,
        )

        for s in scenarios:
            assert "Port_Closure" in s.get("black_swan_type", "")


class TestCustomsScenarios:
    """Tests for customs scenario generation."""

    @pytest.fixture
    def generator(self):
        """Create a generator for testing."""
        return LogisticsScenarioGenerator(num_samples=10, train_ratio=0.6)

    def test_customs_scenarios_generated(self, generator):
        """Test customs scenarios are generated."""
        scenarios = generator.generate_customs_scenarios(
            num_training=3,
            num_test=2,
        )

        assert len(scenarios) > 0

    def test_customs_has_correct_type(self, generator):
        """Test customs scenarios have correct type."""
        scenarios = generator.generate_customs_scenarios(
            num_training=3,
            num_test=2,
        )

        for s in scenarios:
            assert "Customs" in s.get("black_swan_type", "")
