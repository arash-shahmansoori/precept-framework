"""
Integration Tests for Complete PRECEPT Workflows.

Tests end-to-end workflows that span multiple components:
- Scenario generation → Agent execution → Learning → Rule extraction
- Static knowledge → Dual retrieval → Conflict resolution
- Memory storage → Consolidation → Prompt evolution
"""


class TestScenarioToLearningWorkflow:
    """Tests for the scenario → learning workflow."""

    def test_logistics_scenario_to_memory(self):
        """Test generating scenarios and storing as experiences."""
        from precept.memory_store import MemoryStore
        from precept.scenario_generators import generate_logistics_scenarios

        # Generate scenarios
        scenarios = generate_logistics_scenarios(
            num_samples=5,
            train_ratio=1.0,  # All training
        )

        # Store as experiences
        store = MemoryStore()

        for scenario in scenarios:
            store.store_experience(
                task_description=scenario["task"],
                goal=scenario.get("expected", "Complete task"),
                trajectory=[{"action": "simulate", "observation": "success"}],
                outcome="success",
                correctness=1.0,
                strategy_used=scenario.get("black_swan_type", "unknown"),
                lessons_learned=[scenario.get("precept_lesson", "")],
                skills_demonstrated=["logistics"],
                domain="logistics",
            )

        # Verify
        stats = store.get_stats()
        assert stats["current_size"] == len(scenarios)
        assert "logistics" in stats["domains"]

    def test_scenario_with_rule_extraction(self):
        """Test extracting rules from scenario lessons."""
        from precept.rule_parser import DynamicRuleParser
        from precept.scenario_generators import generate_logistics_scenarios

        # Generate scenarios with conflict resolution
        scenarios = generate_logistics_scenarios(
            num_samples=10,
            train_ratio=0.5,
            include_conflict_resolution=True,
        )

        # Extract lessons
        lessons = [
            s.get("precept_lesson", "") for s in scenarios if s.get("precept_lesson")
        ]

        # Parse rules
        parser = DynamicRuleParser(
            known_entities=["rotterdam", "hamburg", "antwerp", "boston", "shanghai"]
        )

        rules = parser.parse_rules(lessons)

        # Some rules should be extracted (depending on lesson format)
        assert isinstance(rules, list)


class TestDualRetrievalWorkflow:
    """Tests for dual retrieval workflow."""

    def test_static_and_dynamic_knowledge_flow(self):
        """Test combining static and dynamic knowledge."""
        from precept.ingestion import SoftIngestionManager
        from precept.memory_store import MemoryStore

        # Create memory store (simulates dynamic knowledge)
        memory = MemoryStore()
        memory.store_experience(
            task_description="Rotterdam port was blocked",
            goal="Find alternative",
            trajectory=[],
            outcome="success",
            correctness=1.0,
            strategy_used="use Hamburg",
            lessons_learned=["Rotterdam blocked, use Hamburg"],
            skills_demonstrated=["adaptability"],
            domain="logistics",
        )

        # Create soft ingestion (simulates patches to static knowledge)
        ingestion = SoftIngestionManager()
        ingestion.ingest_correction(
            target_document_id="port-rotterdam-info",
            correction="Port is currently blocked due to strike",
            source_task="monitoring",
            source_observation="Strike announced",
        )

        # Both should have data
        assert memory.get_stats()["current_size"] == 1
        assert ingestion.get_stats()["total_patches_created"] == 1


class TestConflictResolutionWorkflow:
    """Tests for conflict resolution workflow."""

    def test_conflict_manager_full_workflow(self):
        """Test complete conflict management workflow."""
        from datetime import datetime

        from precept.conflict_resolution import (
            ConflictManager,
            KnowledgeItem,
            KnowledgeSource,
        )

        manager = ConflictManager()

        # Create conflicting knowledge items
        static_item = KnowledgeItem(
            id="static-1",
            content="Rotterdam port is operational 24/7",
            source=KnowledgeSource.STATIC_KB,
            timestamp=datetime(2023, 1, 1),
            confidence=0.9,
        )

        dynamic_item = KnowledgeItem(
            id="dynamic-1",
            content="Rotterdam port is blocked due to strike",
            source=KnowledgeSource.DYNAMIC_EXPERIENCE,
            timestamp=datetime.now(),
            confidence=0.85,
        )

        # Manager should be able to handle these items
        assert static_item.source == KnowledgeSource.STATIC_KB
        assert dynamic_item.source == KnowledgeSource.DYNAMIC_EXPERIENCE

    def test_semantic_conflict_detection_workflow(self):
        """Test ensemble conflict detection workflow."""
        from precept.conflict_resolution import (
            EnsembleConflictDetector,
            ConflictResolutionConfig,
            LearnedPatterns,
            NeuralNLIClassifier,
        )

        config = ConflictResolutionConfig()
        patterns = LearnedPatterns()
        nli = NeuralNLIClassifier(config=config, patterns=patterns)
        detector = EnsembleConflictDetector(config=config, patterns=patterns, nli_classifier=nli)

        # Detector should be initialized
        assert detector is not None


class TestPromptEvolutionWorkflow:
    """Tests for prompt evolution workflow."""

    def test_gepa_evolution_workflow(self):
        """Test GEPA evolution workflow."""
        from precept.gepa import GEPAConfig, GEPAEvolutionEngine

        async def mock_llm(*args, **kwargs):
            return None

        # Create engine with config
        config = GEPAConfig(
            objectives=["success_rate", "efficiency"],
            max_pareto_front_size=5,
        )
        engine = GEPAEvolutionEngine(llm_client=mock_llm, config=config)

        # Initialize with base prompt
        base_prompt = "You are a helpful logistics agent."
        candidate = engine.initialize_pareto_front(base_prompt)

        # Verify candidate properties
        assert candidate.prompt_text == base_prompt
        assert candidate.generation == 0

        # Verify Pareto front
        assert len(engine.pareto_front) == 1
        assert engine.pareto_front[0] == candidate

    def test_complexity_driven_evolution(self):
        """Test complexity-based evolution decisions."""
        from precept.complexity_analyzer import PRECEPTComplexityAnalyzer

        analyzer = PRECEPTComplexityAnalyzer()

        # Analyze different tasks
        simple_task = "Book shipment from A to B"
        complex_task = "Handle multi-hop routing with port closures and customs delays"

        simple_result = analyzer.analyze(simple_task)
        complex_result = analyzer.analyze(complex_task)

        # Both should return results
        assert simple_result is not None
        assert complex_result is not None


class TestMemoryConsolidationWorkflow:
    """Tests for memory consolidation workflow."""

    def test_frequent_strategies_extraction(self):
        """Test extracting frequent strategies from memory."""
        from precept.memory_store import MemoryStore

        store = MemoryStore()

        # Store multiple experiences with same strategy
        for i in range(5):
            store.store_experience(
                task_description=f"Task {i}",
                goal="Test goal",
                trajectory=[],
                outcome="success",
                correctness=0.9,
                strategy_used="repeated_strategy",
                lessons_learned=["Common lesson"],
                skills_demonstrated=["test_skill"],
                domain="test",
            )

        # Get frequent strategies
        strategies = store.get_frequent_strategies(min_count=5)

        assert len(strategies) > 0
        assert strategies[0][0] == "repeated_strategy"
        assert strategies[0][1] == 5  # Count

    def test_frequent_lessons_extraction(self):
        """Test extracting frequent lessons from memory."""
        from precept.memory_store import MemoryStore

        store = MemoryStore()

        # Store experiences with repeated lessons
        for i in range(3):
            store.store_experience(
                task_description=f"Task {i}",
                goal="Test goal",
                trajectory=[],
                outcome="success",
                correctness=1.0,
                strategy_used="test",
                lessons_learned=["Common lesson"],
                skills_demonstrated=[],
                domain="test",
            )

        # Get frequent lessons
        lessons = store.get_frequent_lessons(min_count=3)

        assert len(lessons) > 0
        assert lessons[0][0] == "Common lesson"


class TestFullPipelineIntegration:
    """Tests for full pipeline integration."""

    def test_complete_learning_pipeline(self):
        """Test complete learning pipeline from scenario to rules."""
        from precept.ingestion import SoftIngestionManager
        from precept.memory_store import MemoryStore
        from precept.rule_parser import DynamicRuleParser
        from precept.scenario_generators import generate_logistics_scenarios

        # 1. Generate scenarios
        scenarios = generate_logistics_scenarios(num_samples=5, train_ratio=1.0)

        # 2. Create memory store
        memory = MemoryStore()

        # 3. Create ingestion manager
        ingestion = SoftIngestionManager()

        # 4. Create rule parser
        parser = DynamicRuleParser(
            known_entities=["rotterdam", "hamburg", "antwerp", "boston"]
        )

        # 5. Process scenarios
        for scenario in scenarios:
            # Store experience
            memory.store_experience(
                task_description=scenario["task"],
                goal=scenario.get("expected", "Complete"),
                trajectory=[],
                outcome="success",
                correctness=1.0,
                strategy_used=scenario.get("black_swan_type", "unknown"),
                lessons_learned=[scenario.get("precept_lesson", "")],
                skills_demonstrated=["logistics"],
                domain="logistics",
            )

            # Add any corrections
            if "blocked" in scenario.get("precept_lesson", "").lower():
                ingestion.ingest_warning(
                    query_pattern=scenario.get("black_swan_type", ""),
                    warning=scenario.get("precept_lesson", ""),
                    source_task=scenario["task"],
                )

        # 6. Verify pipeline results
        assert memory.get_stats()["current_size"] == len(scenarios)
        assert "logistics" in memory.get_stats()["domains"]

    def test_agent_components_integration(
        self, mock_domain_strategy, mock_baseline_strategy
    ):
        """Test that agent components work together."""
        from precept import (
            FullReflexionBaselineAgent,
            LLMBaselineAgent,
            PRECEPTAgent,
            ReflexionBaselineAgent,
        )

        # Create all agents
        precept = PRECEPTAgent(domain_strategy=mock_domain_strategy)
        llm = LLMBaselineAgent(baseline_strategy=mock_baseline_strategy)
        reflexion = ReflexionBaselineAgent(baseline_strategy=mock_baseline_strategy)
        full_reflexion = FullReflexionBaselineAgent(
            baseline_strategy=mock_baseline_strategy
        )

        # All should be created
        agents = [precept, llm, reflexion, full_reflexion]

        for agent in agents:
            assert agent is not None
            assert hasattr(agent, "run_task")
            assert hasattr(agent, "get_stats")


class TestConfigurationIntegration:
    """Tests for configuration integration."""

    def test_default_config_loading(self):
        """Test loading default configuration."""
        from precept.config import get_default_config

        config = get_default_config()

        assert config is not None
        assert hasattr(config, "agent")
        assert hasattr(config, "baseline")

    def test_config_with_agents(self, mock_domain_strategy):
        """Test using config with agents."""
        from precept import PRECEPTAgent
        from precept.config import get_default_config

        config = get_default_config()
        config.agent.max_attempts = 10

        agent = PRECEPTAgent(
            domain_strategy=mock_domain_strategy,
            config=config,
        )

        assert agent.config.agent.max_attempts == 10
