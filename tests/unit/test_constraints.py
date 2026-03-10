"""
Unit Tests for precept.constraints module.

Tests RefineInterceptor and constraint handling for deterministic pruning.
"""

import pytest

from precept.constraints import (
    Constraint,
    ConstraintType,
    RefineInterceptor,
    create_refine_interceptor,
)


class TestConstraintType:
    """Tests for ConstraintType enum."""

    def test_constraint_types_exist(self):
        """Test that all constraint types are defined."""
        assert ConstraintType.HARD is not None
        assert ConstraintType.SOFT is not None
        assert ConstraintType.TRANSIENT is not None

    def test_constraint_type_values(self):
        """Test constraint type values."""
        assert ConstraintType.HARD.value == "hard"
        assert ConstraintType.SOFT.value == "soft"
        assert ConstraintType.TRANSIENT.value == "transient"


class TestConstraint:
    """Tests for Constraint dataclass."""

    def test_create_constraint(self):
        """Test creating a constraint."""
        constraint = Constraint(
            solution="rotterdam",
            error_code="R-482",
            reason="Port blocked",
            constraint_type=ConstraintType.HARD,
        )

        assert constraint.solution == "rotterdam"
        assert constraint.error_code == "R-482"
        assert constraint.constraint_type == ConstraintType.HARD

    def test_constraint_defaults(self):
        """Test constraint default values."""
        constraint = Constraint(
            solution="test",
            error_code="E-001",
            reason="Error",
            constraint_type=ConstraintType.SOFT,
        )

        assert constraint.timestamp is not None
        assert isinstance(constraint.timestamp, float)


class TestRefineInterceptor:
    """Tests for RefineInterceptor class."""

    def test_create_interceptor(self):
        """Test creating an interceptor."""
        interceptor = create_refine_interceptor()
        assert interceptor is not None
        assert isinstance(interceptor, RefineInterceptor)

    def test_add_constraint(self):
        """Test adding a constraint."""
        interceptor = create_refine_interceptor()

        constraint = interceptor.add_constraint(
            solution="rotterdam",
            error_code="R-482",
            error_message="Port blocked",
        )

        assert constraint is not None
        assert constraint.solution == "rotterdam"
        assert interceptor.is_forbidden("rotterdam")

    def test_is_forbidden(self):
        """Test checking if solution is forbidden."""
        interceptor = create_refine_interceptor()

        assert interceptor.is_forbidden("rotterdam") is False

        interceptor.add_constraint(
            solution="rotterdam",
            error_code="R-482",
            error_message="Port blocked",
        )

        assert interceptor.is_forbidden("rotterdam") is True
        assert interceptor.is_forbidden("antwerp") is False

    def test_get_remaining_options(self):
        """Test getting remaining non-forbidden options."""
        interceptor = create_refine_interceptor()
        all_options = ["rotterdam", "hamburg", "antwerp"]

        # Initially all available
        remaining = interceptor.get_remaining_options(all_options)
        assert len(remaining) == 3

        # Add constraint
        interceptor.add_constraint("rotterdam", "R-482", "Blocked")
        remaining = interceptor.get_remaining_options(all_options)
        assert len(remaining) == 2
        assert "rotterdam" not in remaining

    def test_get_forbidden_injection(self):
        """Test getting forbidden section for prompt."""
        interceptor = create_refine_interceptor()

        # No forbidden options
        injection = interceptor.get_forbidden_injection()
        assert injection == ""

        # Add constraints
        interceptor.add_constraint("rotterdam", "R-482", "Blocked")
        interceptor.add_constraint("hamburg", "H-903", "US routes blocked")

        injection = interceptor.get_forbidden_injection()
        assert "rotterdam" in injection.lower()
        assert "hamburg" in injection.lower()
        assert "FORBIDDEN" in injection

    def test_reset(self):
        """Test resetting interceptor state."""
        interceptor = create_refine_interceptor()

        interceptor.add_constraint("rotterdam", "R-482", "Blocked")
        assert interceptor.is_forbidden("rotterdam")

        interceptor.reset()
        assert interceptor.is_forbidden("rotterdam") is False

    def test_record_prevented_retry(self):
        """Test recording prevented retries."""
        interceptor = create_refine_interceptor()

        initial_stats = interceptor.get_stats()
        assert initial_stats["dumb_retries_prevented"] == 0

        interceptor.record_prevented_retry()
        interceptor.record_prevented_retry()

        stats = interceptor.get_stats()
        assert stats["dumb_retries_prevented"] == 2

    def test_get_stats(self):
        """Test getting interceptor statistics."""
        interceptor = create_refine_interceptor()

        stats = interceptor.get_stats()
        assert "total_constraints" in stats
        assert "hard_constraints" in stats
        assert "soft_constraints" in stats
        assert "dumb_retries_prevented" in stats
        assert "diagnostic_probes" in stats

    def test_constraint_classification(self):
        """Test that constraints are classified correctly."""
        interceptor = create_refine_interceptor()

        # Hard constraint (blocked)
        interceptor.add_constraint("rotterdam", "R-482", "Port blocked permanently")
        stats = interceptor.get_stats()
        assert stats["hard_constraints"] >= 1

    def test_suggest_diagnostic_probe(self):
        """Test diagnostic probe suggestions."""
        interceptor = create_refine_interceptor()

        # Should suggest probe for certain error types
        probe = interceptor.suggest_diagnostic_probe("R-482", "Port blocked")

        # Probe can be None or a string
        if probe:
            assert isinstance(probe, str)


class TestConstraintIntegration:
    """Integration tests for constraint system."""

    def test_full_pruning_workflow(self):
        """Test a complete pruning workflow."""
        interceptor = create_refine_interceptor()
        all_options = ["rotterdam", "hamburg", "antwerp", "ningbo"]

        # Initial state
        assert len(interceptor.get_remaining_options(all_options)) == 4

        # First failure
        interceptor.add_constraint("rotterdam", "R-482", "Port blocked")
        remaining = interceptor.get_remaining_options(all_options)
        assert "rotterdam" not in remaining
        assert len(remaining) == 3

        # Second failure
        interceptor.add_constraint("hamburg", "H-903", "US routes blocked")
        remaining = interceptor.get_remaining_options(all_options)
        assert "hamburg" not in remaining
        assert len(remaining) == 2

        # Check injection
        injection = interceptor.get_forbidden_injection()
        assert "rotterdam" in injection.lower()
        assert "hamburg" in injection.lower()

        # Stats should reflect constraints
        stats = interceptor.get_stats()
        assert stats["total_constraints"] == 2

    def test_prevents_dumb_retries(self):
        """Test that interceptor prevents retrying forbidden options."""
        interceptor = create_refine_interceptor()

        # Add constraint
        interceptor.add_constraint("rotterdam", "R-482", "Blocked")

        # Simulate LLM suggesting forbidden option
        suggested = "rotterdam"
        if interceptor.is_forbidden(suggested):
            interceptor.record_prevented_retry()

        stats = interceptor.get_stats()
        assert stats["dumb_retries_prevented"] == 1
