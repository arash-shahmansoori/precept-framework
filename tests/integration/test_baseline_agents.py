"""
Integration Tests for Baseline Agents.

Tests LLMBaselineAgent, ReflexionBaselineAgent, and FullReflexionBaselineAgent.
"""

import pytest

from precept.config import BaselineConfig, get_default_config


class TestLLMBaselineAgentInitialization:
    """Tests for LLMBaselineAgent initialization."""

    def test_agent_import(self):
        """Test that LLMBaselineAgent can be imported."""
        from precept import LLMBaselineAgent

        assert LLMBaselineAgent is not None

    def test_agent_with_config(self, mock_baseline_strategy):
        """Test agent initialization with config."""
        from precept import LLMBaselineAgent

        config = BaselineConfig(max_attempts=5, verbose=True)

        agent = LLMBaselineAgent(
            baseline_strategy=mock_baseline_strategy,
            config=config,
        )

        assert agent.config.max_attempts == 5
        assert agent.config.verbose is True

    def test_agent_legacy_parameters(self, mock_baseline_strategy):
        """Test agent with legacy parameters."""
        from precept import LLMBaselineAgent

        agent = LLMBaselineAgent(
            baseline_strategy=mock_baseline_strategy,
            model="gpt-4o",
            verbose=True,
            max_internal_workers=5,
        )

        assert agent.model == "gpt-4o"
        assert agent.verbose is True
        assert agent.max_internal_workers == 5


class TestLLMBaselineAgentProperties:
    """Tests for LLMBaselineAgent properties."""

    def test_property_accessors(self, mock_baseline_strategy):
        """Test property accessors."""
        from precept import LLMBaselineAgent

        agent = LLMBaselineAgent(baseline_strategy=mock_baseline_strategy)

        assert isinstance(agent.model, str)
        assert isinstance(agent.verbose, bool)
        assert isinstance(agent.max_internal_workers, int)
        assert isinstance(agent.MAX_ATTEMPTS, int)


class TestLLMBaselineAgentStatistics:
    """Tests for LLMBaselineAgent statistics."""

    def test_initial_stats(self, mock_baseline_strategy):
        """Test initial statistics."""
        from precept import LLMBaselineAgent

        agent = LLMBaselineAgent(baseline_strategy=mock_baseline_strategy)

        assert agent.total_tasks == 0
        assert agent.successful_tasks == 0
        assert agent.llm_calls == 0

    def test_get_success_rate(self, mock_baseline_strategy):
        """Test success rate calculation."""
        from precept import LLMBaselineAgent

        agent = LLMBaselineAgent(baseline_strategy=mock_baseline_strategy)
        assert agent.get_success_rate() == 0.0

    def test_get_stats(self, mock_baseline_strategy):
        """Test get_stats returns expected structure."""
        from precept import LLMBaselineAgent

        agent = LLMBaselineAgent(baseline_strategy=mock_baseline_strategy)
        stats = agent.get_stats()

        assert "total_tasks" in stats
        assert "successful_tasks" in stats
        assert "success_rate" in stats
        assert "avg_steps" in stats
        assert "llm_calls" in stats
        assert "llm_accuracy" in stats
        assert "baseline_type" in stats
        assert stats["baseline_type"] == "adapted_react"


class TestReflexionBaselineAgentInitialization:
    """Tests for ReflexionBaselineAgent initialization."""

    def test_agent_import(self):
        """Test that ReflexionBaselineAgent can be imported."""
        from precept import ReflexionBaselineAgent

        assert ReflexionBaselineAgent is not None

    def test_agent_with_config(self, mock_baseline_strategy):
        """Test agent initialization with config."""
        from precept import ReflexionBaselineAgent

        config = BaselineConfig(max_attempts=5)

        agent = ReflexionBaselineAgent(
            baseline_strategy=mock_baseline_strategy,
            config=config,
        )

        assert agent.MAX_ATTEMPTS == 5


class TestReflexionBaselineAgentStatistics:
    """Tests for ReflexionBaselineAgent statistics."""

    def test_get_stats(self, mock_baseline_strategy):
        """Test get_stats returns reflexion-specific stats."""
        from precept import ReflexionBaselineAgent

        agent = ReflexionBaselineAgent(baseline_strategy=mock_baseline_strategy)
        stats = agent.get_stats()

        assert "baseline_type" in stats
        assert stats["baseline_type"] == "adapted_reflexion"
        assert "reflections_generated" in stats
        assert "reflections_per_task" in stats


class TestFullReflexionBaselineAgentInitialization:
    """Tests for FullReflexionBaselineAgent initialization."""

    def test_agent_import(self):
        """Test that FullReflexionBaselineAgent can be imported."""
        from precept import FullReflexionBaselineAgent

        assert FullReflexionBaselineAgent is not None

    def test_agent_with_config(self, mock_baseline_strategy):
        """Test agent initialization with config."""
        from precept import FullReflexionBaselineAgent

        config = BaselineConfig(max_reflections_per_type=10)

        agent = FullReflexionBaselineAgent(
            baseline_strategy=mock_baseline_strategy,
            config=config,
        )

        assert agent.max_reflections == 10


class TestFullReflexionMemory:
    """Tests for FullReflexionBaselineAgent memory management."""

    def test_class_level_memory(self):
        """Test class-level memory operations."""
        from precept import FullReflexionBaselineAgent

        # Clear any existing memory
        FullReflexionBaselineAgent.clear_memory()

        # Add reflection
        FullReflexionBaselineAgent.add_reflection(
            "test_type",
            {"episode": 1, "task": "Test task"},
        )

        # Get memory
        memory = FullReflexionBaselineAgent.get_reflection_memory("test_type")
        assert len(memory) == 1

        # Get stats
        stats = FullReflexionBaselineAgent.get_memory_stats()
        assert "test_type" in stats
        assert stats["test_type"] == 1

        # Clean up
        FullReflexionBaselineAgent.clear_memory()

    def test_clear_specific_memory(self):
        """Test clearing specific task type memory."""
        from precept import FullReflexionBaselineAgent

        FullReflexionBaselineAgent.clear_memory()

        FullReflexionBaselineAgent.add_reflection("type_a", {"episode": 1})
        FullReflexionBaselineAgent.add_reflection("type_b", {"episode": 1})

        FullReflexionBaselineAgent.clear_memory("type_a")

        assert len(FullReflexionBaselineAgent.get_reflection_memory("type_a")) == 0
        assert len(FullReflexionBaselineAgent.get_reflection_memory("type_b")) == 1

        FullReflexionBaselineAgent.clear_memory()


class TestFullReflexionAgentStatistics:
    """Tests for FullReflexionBaselineAgent statistics."""

    def test_get_stats(self, mock_baseline_strategy):
        """Test get_stats returns full reflexion-specific stats."""
        from precept import FullReflexionBaselineAgent

        agent = FullReflexionBaselineAgent(baseline_strategy=mock_baseline_strategy)
        stats = agent.get_stats()

        assert "baseline_type" in stats
        assert stats["baseline_type"] == "full_reflexion"
        assert "total_episodes" in stats
        assert "reflections_reused" in stats
        assert "memory_stats" in stats


class TestBaselineAgentComparison:
    """Tests comparing baseline agents."""

    def test_all_baselines_have_same_interface(self, mock_baseline_strategy):
        """Test that all baselines have consistent interface."""
        from precept import (
            FullReflexionBaselineAgent,
            LLMBaselineAgent,
            ReflexionBaselineAgent,
        )

        agents = [
            LLMBaselineAgent(baseline_strategy=mock_baseline_strategy),
            ReflexionBaselineAgent(baseline_strategy=mock_baseline_strategy),
            FullReflexionBaselineAgent(baseline_strategy=mock_baseline_strategy),
        ]

        for agent in agents:
            # All should have these methods
            assert hasattr(agent, "run_task")
            assert hasattr(agent, "connect")
            assert hasattr(agent, "disconnect")
            assert hasattr(agent, "get_stats")
            assert hasattr(agent, "get_success_rate")

            # All should have these properties
            assert hasattr(agent, "model")
            assert hasattr(agent, "verbose")
            assert hasattr(agent, "MAX_ATTEMPTS")

    def test_baseline_types_are_different(self, mock_baseline_strategy):
        """Test that baseline types are correctly identified."""
        from precept import (
            FullReflexionBaselineAgent,
            LLMBaselineAgent,
            ReflexionBaselineAgent,
        )

        llm = LLMBaselineAgent(baseline_strategy=mock_baseline_strategy)
        ref = ReflexionBaselineAgent(baseline_strategy=mock_baseline_strategy)
        full = FullReflexionBaselineAgent(baseline_strategy=mock_baseline_strategy)

        assert llm.get_stats()["baseline_type"] == "adapted_react"
        assert ref.get_stats()["baseline_type"] == "adapted_reflexion"
        assert full.get_stats()["baseline_type"] == "full_reflexion"
