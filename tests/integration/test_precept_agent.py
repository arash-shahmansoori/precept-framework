"""
Integration Tests for PRECEPTAgent.

Tests the agent's integration with MCP client, domain strategies,
and learning components.
"""

from precept.config import get_default_config


class TestPRECEPTAgentInitialization:
    """Tests for PRECEPTAgent initialization."""

    def test_agent_import(self):
        """Test that PRECEPTAgent can be imported."""
        from precept import PRECEPTAgent

        assert PRECEPTAgent is not None

    def test_agent_with_config(self, mock_domain_strategy):
        """Test agent initialization with config."""
        from precept import PRECEPTAgent

        config = get_default_config()
        config.agent.max_retries = 5  # AgentConfig uses max_retries

        agent = PRECEPTAgent(
            domain_strategy=mock_domain_strategy,
            config=config,
        )

        assert agent.config.agent.max_retries == 5
        assert agent.strategy == mock_domain_strategy

    def test_agent_legacy_parameters(self, mock_domain_strategy):
        """Test agent with legacy parameters."""
        from precept import PRECEPTAgent

        agent = PRECEPTAgent(
            domain_strategy=mock_domain_strategy,
            model="gpt-4o",
            enable_llm_reasoning=False,
            verbose_llm=True,
            max_internal_workers=5,
        )

        assert agent.model == "gpt-4o"
        assert agent.enable_llm_reasoning is False
        assert agent.verbose_llm is True
        assert agent.max_internal_workers == 5


class TestPRECEPTAgentProperties:
    """Tests for PRECEPTAgent property accessors."""

    def test_property_accessors(self, mock_domain_strategy):
        """Test property accessors work correctly."""
        from precept import PRECEPTAgent

        agent = PRECEPTAgent(domain_strategy=mock_domain_strategy)

        # Test properties - use lowercase property names
        assert isinstance(agent.model, str)
        assert isinstance(agent.enable_llm_reasoning, bool)
        assert isinstance(agent.force_llm_reasoning, bool)
        assert isinstance(agent.max_internal_workers, int)
        assert isinstance(agent.consolidation_interval, int)
        assert isinstance(agent.compass_evolution_interval, int)
        assert isinstance(agent.failure_threshold, int)


class TestPRECEPTAgentStatistics:
    """Tests for PRECEPTAgent statistics methods."""

    def test_initial_stats(self, mock_domain_strategy):
        """Test initial statistics are zero."""
        from precept import PRECEPTAgent

        agent = PRECEPTAgent(domain_strategy=mock_domain_strategy)

        assert agent.total_tasks == 0
        assert agent.successful_tasks == 0
        assert agent.get_success_rate() == 0.0

    def test_get_stats(self, mock_domain_strategy):
        """Test get_stats returns expected structure."""
        from precept import PRECEPTAgent

        agent = PRECEPTAgent(domain_strategy=mock_domain_strategy)
        stats = agent.get_stats()

        assert "total_tasks" in stats
        assert "successful_tasks" in stats
        assert "success_rate" in stats
        assert "avg_steps" in stats
        assert "domain" in stats
        assert "learning_events" in stats

    def test_get_llm_reasoning_stats(self, mock_domain_strategy):
        """Test LLM reasoning statistics."""
        from precept import PRECEPTAgent

        agent = PRECEPTAgent(domain_strategy=mock_domain_strategy)
        stats = agent.get_llm_reasoning_stats()

        assert "total_calls" in stats
        assert "successes" in stats
        assert "failures" in stats
        assert "success_rate" in stats
        assert "enabled" in stats

    def test_get_pruning_stats(self, mock_domain_strategy):
        """Test pruning statistics."""
        from precept import PRECEPTAgent

        agent = PRECEPTAgent(domain_strategy=mock_domain_strategy)
        stats = agent.get_pruning_stats()

        assert "total_constraints" in stats
        assert "hard_constraints" in stats
        assert "soft_constraints" in stats
        assert "dumb_retries_prevented" in stats
        assert "pruning_efficiency" in stats

    def test_get_prompt_stats(self, mock_domain_strategy):
        """Test prompt evolution statistics."""
        from precept import PRECEPTAgent

        agent = PRECEPTAgent(domain_strategy=mock_domain_strategy)
        stats = agent.get_prompt_stats()

        assert "prompt_generation" in stats
        assert "has_evolved" in stats


class TestPRECEPTAgentSession:
    """Tests for PRECEPTAgent session management."""

    def test_start_session(self, mock_domain_strategy):
        """Test starting a session."""
        from precept import PRECEPTAgent

        agent = PRECEPTAgent(domain_strategy=mock_domain_strategy)
        session_id = agent.start_session()

        assert session_id is not None
        assert len(session_id) > 0

    def test_start_session_with_id(self, mock_domain_strategy):
        """Test starting session with custom ID."""
        from precept import PRECEPTAgent

        agent = PRECEPTAgent(domain_strategy=mock_domain_strategy)
        session_id = agent.start_session("test-session")

        assert session_id == "test-session"

    def test_end_session(self, mock_domain_strategy):
        """Test ending a session."""
        from precept import PRECEPTAgent

        agent = PRECEPTAgent(domain_strategy=mock_domain_strategy)
        agent.start_session("test")
        stats = agent.end_session()

        assert "session_id" in stats
        assert "duration" in stats
        assert "tasks" in stats

    def test_end_session_no_active(self, mock_domain_strategy):
        """Test ending session when none active."""
        from precept import PRECEPTAgent

        agent = PRECEPTAgent(domain_strategy=mock_domain_strategy)
        result = agent.end_session()

        assert "error" in result

    def test_reset_conversation(self, mock_domain_strategy):
        """Test resetting conversation."""
        from precept import PRECEPTAgent

        agent = PRECEPTAgent(domain_strategy=mock_domain_strategy)
        agent.start_session()
        agent.reset_conversation()

        history = agent.get_conversation_history()
        assert len(history) == 0


class TestPRECEPTAgentPromptEvolution:
    """Tests for PRECEPT prompt evolution."""

    def test_get_current_prompt(self, mock_domain_strategy):
        """Test getting current prompt."""
        from precept import PRECEPTAgent

        agent = PRECEPTAgent(domain_strategy=mock_domain_strategy)

        # Initially empty until connected
        prompt = agent.get_current_prompt()
        assert prompt == ""
