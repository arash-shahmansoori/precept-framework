#!/usr/bin/env python3
"""
Tests for CSP Constraint Manager.

Tests the constraint satisfaction problem handling:
1. Constraint hierarchy (Physics > Policy > Instruction)
2. Dependency graph mapping
3. Causal chain tracking
4. Conflict detection and resolution
5. Constraint discovery and satisfaction
"""

import pytest
import sys
import os
import time
from unittest.mock import MagicMock, patch

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from precept.csp_constraint_manager import (
    Constraint,
    ConstraintTier,
    ConstraintType,
    CausalChain,
    CSPConstraintManager,
    ExecutionFeedback,
    ConflictResolver,
)


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def csp_manager():
    """Create a fresh CSP constraint manager."""
    return CSPConstraintManager()


@pytest.fixture
def sample_physics_constraint():
    """Create a sample Physics constraint."""
    return Constraint(
        id="NETWORK_DOWN",
        name="Network Unavailable",
        tier=ConstraintTier.PHYSICS,
        type=ConstraintType.HARD,
        description="Network connectivity is unavailable",
        solution_patterns=["use offline mode", "use cached data"],
    )


@pytest.fixture
def sample_policy_constraint():
    """Create a sample Policy constraint."""
    return Constraint(
        id="NO_EXTERNAL_API",
        name="No External API Calls",
        tier=ConstraintTier.POLICY,
        type=ConstraintType.HARD,
        description="External API calls are prohibited",
        solution_patterns=["use internal services", "use mock data"],
    )


@pytest.fixture
def sample_instruction_constraint():
    """Create a sample Instruction constraint."""
    return Constraint(
        id="USE_JSON_FORMAT",
        name="Use JSON Format",
        tier=ConstraintTier.INSTRUCTION,
        type=ConstraintType.SOFT,
        description="Output should be in JSON format",
        solution_patterns=["format output as JSON"],
    )


# =============================================================================
# CONSTRAINT TIER TESTS
# =============================================================================

class TestConstraintTier:
    """Tests for constraint tier hierarchy."""
    
    def test_tier_values(self):
        """Test that tiers have correct priority values."""
        assert ConstraintTier.INSTRUCTION.value == 1
        assert ConstraintTier.POLICY.value == 2
        assert ConstraintTier.PHYSICS.value == 3
    
    def test_physics_highest_priority(self):
        """Test that Physics has highest priority."""
        assert ConstraintTier.PHYSICS.value > ConstraintTier.POLICY.value
        assert ConstraintTier.PHYSICS.value > ConstraintTier.INSTRUCTION.value
    
    def test_policy_mid_priority(self):
        """Test that Policy has middle priority."""
        assert ConstraintTier.POLICY.value > ConstraintTier.INSTRUCTION.value
        assert ConstraintTier.POLICY.value < ConstraintTier.PHYSICS.value


# =============================================================================
# CONSTRAINT TYPE TESTS
# =============================================================================

class TestConstraintType:
    """Tests for constraint types."""
    
    def test_all_types_exist(self):
        """Test that all constraint types exist."""
        assert ConstraintType.HARD is not None
        assert ConstraintType.SOFT is not None
        assert ConstraintType.INTERDEPENDENT is not None
        assert ConstraintType.CONTRADICTING is not None


# =============================================================================
# CONSTRAINT CREATION TESTS
# =============================================================================

class TestConstraintCreation:
    """Tests for constraint creation."""
    
    def test_basic_constraint_creation(self):
        """Test creating a basic constraint."""
        constraint = Constraint(
            id="TEST_001",
            name="Test Constraint",
            tier=ConstraintTier.POLICY,
            type=ConstraintType.HARD,
            description="A test constraint",
        )
        
        assert constraint.id == "TEST_001"
        assert constraint.name == "Test Constraint"
        assert constraint.tier == ConstraintTier.POLICY
        assert constraint.type == ConstraintType.HARD
        assert constraint.discovered is False
        assert constraint.satisfied is False
    
    def test_constraint_with_dependencies(self):
        """Test creating a constraint with dependencies."""
        constraint = Constraint(
            id="GIT_CLONE",
            name="Git Clone",
            tier=ConstraintTier.INSTRUCTION,
            type=ConstraintType.INTERDEPENDENT,
            description="Clone a git repository",
            dependencies=["NETWORK_AVAILABLE", "GIT_INSTALLED"],
        )
        
        assert len(constraint.dependencies) == 2
        assert "NETWORK_AVAILABLE" in constraint.dependencies
    
    def test_constraint_with_conflicts(self):
        """Test creating a constraint with conflicts."""
        constraint = Constraint(
            id="ENABLE_DEBUG",
            name="Enable Debug Mode",
            tier=ConstraintTier.INSTRUCTION,
            type=ConstraintType.CONTRADICTING,
            description="Enable debug logging",
            conflicts_with=["PRODUCTION_MODE"],
        )
        
        assert len(constraint.conflicts_with) == 1
        assert "PRODUCTION_MODE" in constraint.conflicts_with


# =============================================================================
# CSP MANAGER BASIC TESTS
# =============================================================================

class TestCSPManagerBasic:
    """Basic tests for CSP Constraint Manager."""
    
    def test_manager_creation(self, csp_manager):
        """Test creating a CSP manager."""
        assert csp_manager is not None
    
    def test_manager_has_constraints_dict(self, csp_manager):
        """Test that manager has constraints storage."""
        assert hasattr(csp_manager, 'constraints')
        assert isinstance(csp_manager.constraints, dict)
    
    def test_manager_has_causal_tracker(self, csp_manager):
        """Test that manager has causal chain tracker."""
        assert hasattr(csp_manager, 'causal_tracker')
    
    def test_manager_has_stats(self, csp_manager):
        """Test that manager tracks statistics."""
        assert hasattr(csp_manager, 'stats')
        assert 'constraints_discovered' in csp_manager.stats


# =============================================================================
# CONSTRAINT HIERARCHY RESOLUTION TESTS
# =============================================================================

class TestConstraintHierarchyResolution:
    """Tests for constraint hierarchy resolution."""
    
    def test_physics_tier_highest_value(self):
        """Test that Physics tier has highest value."""
        assert ConstraintTier.PHYSICS.value == 3
        assert ConstraintTier.POLICY.value == 2
        assert ConstraintTier.INSTRUCTION.value == 1
    
    def test_constraint_tier_ordering(self):
        """Test constraint tier ordering for resolution."""
        tiers = [ConstraintTier.INSTRUCTION, ConstraintTier.POLICY, ConstraintTier.PHYSICS]
        sorted_tiers = sorted(tiers, key=lambda t: t.value, reverse=True)
        
        assert sorted_tiers[0] == ConstraintTier.PHYSICS
        assert sorted_tiers[1] == ConstraintTier.POLICY
        assert sorted_tiers[2] == ConstraintTier.INSTRUCTION


# =============================================================================
# CAUSAL CHAIN TESTS
# =============================================================================

class TestCausalChain:
    """Tests for causal chain tracking."""
    
    def test_causal_chain_creation(self):
        """Test creating a causal chain."""
        chain = CausalChain(
            root_constraint="NETWORK_DOWN",
            triggered_constraints=["PIP_FAILS", "GIT_FAILS", "APT_FAILS"],
        )
        
        assert chain.root_constraint == "NETWORK_DOWN"
        assert len(chain.triggered_constraints) == 3
        assert chain.frequency == 1
    
    def test_causal_chain_to_dict(self):
        """Test serializing causal chain."""
        chain = CausalChain(
            root_constraint="NETWORK_DOWN",
            triggered_constraints=["PIP_FAILS"],
        )
        
        chain_dict = chain.to_dict()
        assert chain_dict["root"] == "NETWORK_DOWN"
        assert "triggers" in chain_dict
    
    def test_causal_chain_from_dict(self):
        """Test deserializing causal chain."""
        data = {
            "root": "NETWORK_DOWN",
            "triggers": ["PIP_FAILS", "GIT_FAILS"],
            "discovered_at": time.time(),
            "frequency": 3,
        }
        
        chain = CausalChain.from_dict(data)
        assert chain.root_constraint == "NETWORK_DOWN"
        assert chain.frequency == 3


class TestCausalChainTracking:
    """Tests for causal chain tracking in CSP Manager."""
    
    def test_manager_has_causal_tracker(self, csp_manager):
        """Test that manager has causal chain tracker."""
        assert csp_manager.causal_tracker is not None
    
    def test_causal_tracker_can_record(self, csp_manager):
        """Test that causal tracker can record chains."""
        tracker = csp_manager.causal_tracker
        # Tracker should have method to get triggered constraints
        assert hasattr(tracker, 'get_triggered_constraints')


# =============================================================================
# CONSTRAINT DISCOVERY TESTS
# =============================================================================

class TestConstraintDiscovery:
    """Tests for constraint discovery."""
    
    def test_constraint_discovered_field(self, sample_physics_constraint):
        """Test constraint discovered field."""
        assert sample_physics_constraint.discovered is False
        sample_physics_constraint.discovered = True
        assert sample_physics_constraint.discovered is True
    
    def test_constraint_satisfied_field(self, sample_physics_constraint):
        """Test constraint satisfied field."""
        assert sample_physics_constraint.satisfied is False
        sample_physics_constraint.satisfied = True
        assert sample_physics_constraint.satisfied is True
    
    def test_manager_tracks_discovered_count(self, csp_manager):
        """Test that manager tracks constraints discovered."""
        assert csp_manager.stats['constraints_discovered'] == 0


# =============================================================================
# CONFLICT DETECTION TESTS
# =============================================================================

class TestConflictDetection:
    """Tests for conflict detection."""
    
    def test_constraint_conflicts_with_field(self):
        """Test constraint conflicts_with field."""
        constraint = Constraint(
            id="DEBUG_MODE",
            name="Debug Mode",
            tier=ConstraintTier.INSTRUCTION,
            type=ConstraintType.CONTRADICTING,
            description="Enable debug mode",
            conflicts_with=["PRODUCTION_MODE"],
        )
        
        assert "PRODUCTION_MODE" in constraint.conflicts_with
    
    def test_conflict_resolver_exists(self):
        """Test that ConflictResolver class exists."""
        from precept.csp_constraint_manager import ConflictResolver
        assert ConflictResolver is not None
    
    def test_manager_has_conflict_check(self, csp_manager):
        """Test that manager can check for conflicts."""
        assert hasattr(csp_manager, 'has_conflicts')
        # Initially no conflicts
        assert csp_manager.has_conflicts() is False


# =============================================================================
# EXECUTION FEEDBACK TESTS
# =============================================================================

class TestExecutionFeedback:
    """Tests for execution feedback processing."""
    
    def test_feedback_class_exists(self):
        """Test that ExecutionFeedback class exists."""
        assert ExecutionFeedback is not None
    
    def test_feedback_dataclass_fields(self):
        """Test ExecutionFeedback has expected fields."""
        # Check the class has expected attributes defined
        import dataclasses
        if dataclasses.is_dataclass(ExecutionFeedback):
            field_names = [f.name for f in dataclasses.fields(ExecutionFeedback)]
            # Should have basic execution feedback fields
            assert len(field_names) > 0
    
    def test_manager_has_intercept_feedback(self, csp_manager):
        """Test that manager can intercept feedback."""
        assert hasattr(csp_manager, 'intercept_feedback')


# =============================================================================
# DEPENDENCY GRAPH TESTS
# =============================================================================

class TestDependencyGraph:
    """Tests for dependency graph functionality."""
    
    def test_constraint_dependencies_field(self):
        """Test constraint dependencies field."""
        constraint = Constraint(
            id="GIT_CLONE",
            name="Git Clone",
            tier=ConstraintTier.INSTRUCTION,
            type=ConstraintType.INTERDEPENDENT,
            description="Clone repository",
            dependencies=["NETWORK_AVAILABLE"],
        )
        
        assert "NETWORK_AVAILABLE" in constraint.dependencies
    
    def test_interdependent_type(self):
        """Test interdependent constraint type."""
        constraint = Constraint(
            id="TEST",
            name="Test",
            tier=ConstraintTier.INSTRUCTION,
            type=ConstraintType.INTERDEPENDENT,
            description="Test",
            dependencies=["DEP_A", "DEP_B"],
        )
        
        assert constraint.type == ConstraintType.INTERDEPENDENT
        assert len(constraint.dependencies) == 2


# =============================================================================
# SOLUTION PATTERN TESTS
# =============================================================================

class TestSolutionPatterns:
    """Tests for solution pattern matching."""
    
    def test_constraint_solution_patterns_field(self, sample_physics_constraint):
        """Test constraint solution patterns field."""
        patterns = sample_physics_constraint.solution_patterns
        
        assert len(patterns) >= 1
        assert "use offline mode" in patterns or "use cached data" in patterns
    
    def test_add_solution_pattern_to_constraint(self):
        """Test adding solution pattern to constraint."""
        constraint = Constraint(
            id="TEST",
            name="Test",
            tier=ConstraintTier.PHYSICS,
            type=ConstraintType.HARD,
            description="Test",
            solution_patterns=["pattern_a"],
        )
        
        constraint.solution_patterns.append("pattern_b")
        
        assert "pattern_b" in constraint.solution_patterns


# =============================================================================
# SERIALIZATION TESTS
# =============================================================================

class TestSerialization:
    """Tests for CSP constraint serialization."""
    
    def test_causal_chain_to_dict(self):
        """Test CausalChain to_dict method."""
        chain = CausalChain(
            root_constraint="NET_DOWN",
            triggered_constraints=["A", "B"],
        )
        
        d = chain.to_dict()
        assert d["root"] == "NET_DOWN"
        assert "triggers" in d
    
    def test_causal_chain_from_dict(self):
        """Test CausalChain from_dict method."""
        data = {
            "root": "NET_DOWN",
            "triggers": ["A", "B"],
            "discovered_at": 12345.0,
            "frequency": 5,
        }
        
        chain = CausalChain.from_dict(data)
        assert chain.root_constraint == "NET_DOWN"
        assert chain.frequency == 5


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
