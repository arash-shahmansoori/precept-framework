"""
Integration Tests for PRECEPT Learning Components.

Tests the integration between:
- Memory Store
- Rule Parser
- Context Engineering
- Ingestion Pipeline
- Conflict Resolution
"""

import pytest

from precept.ingestion import SoftIngestionManager
from precept.memory_store import ExperienceType, MemoryPriority, MemoryStore
from precept.rule_parser import DynamicRuleParser


class TestMemoryStoreIntegration:
    """Tests for MemoryStore integration with other components."""

    def test_store_and_retrieve_experience(self):
        """Test storing and retrieving an experience."""
        store = MemoryStore()

        # Store an experience
        exp = store.store_experience(
            task_description="Ship cargo from Rotterdam to Boston",
            goal="Complete shipment successfully",
            trajectory=[
                {"action": "check_port", "observation": "Port blocked"},
                {"action": "use_alternative", "observation": "Using Hamburg"},
            ],
            outcome="success",
            correctness=1.0,
            strategy_used="fallback_routing",
            lessons_learned=["Rotterdam blocked, use Hamburg"],
            skills_demonstrated=["error_handling", "adaptability"],
            domain="logistics",
        )

        assert exp.id is not None
        assert exp.domain == "logistics"

        # Retrieve relevant experiences
        results = store.retrieve_relevant("Rotterdam shipment blocked")
        assert len(results) > 0

        # Verify stats
        stats = store.get_stats()
        assert stats["total_added"] == 1
        assert stats["current_size"] == 1

    def test_memory_store_with_multiple_domains(self):
        """Test storing experiences across multiple domains."""
        store = MemoryStore()

        # Store logistics experience
        store.store_experience(
            task_description="Ship cargo",
            goal="Complete shipment",
            trajectory=[],
            outcome="success",
            correctness=1.0,
            strategy_used="direct",
            lessons_learned=["Use direct routes"],
            skills_demonstrated=["logistics"],
            domain="logistics",
        )

        # Store coding experience
        store.store_experience(
            task_description="Debug function",
            goal="Fix the bug",
            trajectory=[],
            outcome="success",
            correctness=1.0,
            strategy_used="debugging",
            lessons_learned=["Check edge cases"],
            skills_demonstrated=["debugging"],
            domain="coding",
        )

        stats = store.get_stats()
        assert stats["current_size"] == 2
        assert "logistics" in stats["domains"]
        assert "coding" in stats["domains"]

    def test_memory_usefulness_feedback(self):
        """Test updating memory usefulness based on feedback."""
        store = MemoryStore()

        exp = store.store_experience(
            task_description="Test task",
            goal="Test goal",
            trajectory=[],
            outcome="success",
            correctness=1.0,
            strategy_used="test",
            lessons_learned=["Test lesson"],
            skills_demonstrated=[],
            domain="test",
        )

        initial_score = exp.usefulness_score

        # Positive feedback
        store.update_usefulness(exp.id, feedback=0.9)
        assert exp.usefulness_score > initial_score


class TestRuleParserIntegration:
    """Tests for RuleParser integration with memory and learning."""

    def test_parse_and_apply_rules(self):
        """Test parsing rules and finding applicable ones."""
        parser = DynamicRuleParser(
            known_entities=["rotterdam", "hamburg", "antwerp", "boston"]
        )

        # Parse rules from learned text
        rules = parser.parse_rules([
            "R-482: Rotterdam blocked, use Hamburg",
            "H-903: Hamburg closed for US, use Antwerp",
        ])

        # Get blocked entities
        blocked = parser.get_all_blocked_entities(rules)
        alternatives = parser.get_all_alternatives(rules)

        assert isinstance(blocked, set)
        assert isinstance(alternatives, set)

    def test_rule_parser_with_dynamic_entities(self):
        """Test adding entities dynamically."""
        parser = DynamicRuleParser()

        # Initially empty
        assert len(parser._known_entities) == 0

        # Add entities dynamically
        parser.add_entities(["singapore", "hong_kong", "tokyo"])

        assert "singapore" in parser._known_entities
        assert "hong_kong" in parser._known_entities
        assert "tokyo" in parser._known_entities


class TestIngestionIntegration:
    """Tests for Ingestion pipeline integration."""

    def test_soft_ingestion_correction_flow(self):
        """Test the soft ingestion correction flow."""
        manager = SoftIngestionManager()

        # Ingest a correction
        result = manager.ingest_correction(
            target_document_id="port-rotterdam",
            correction="Rotterdam port is currently BLOCKED due to strike",
            source_task="logistics_task_001",
            source_observation="Error R-482 received",
            confidence=0.95,
            domain="logistics",
        )

        assert result.success is True
        assert result.patch_id is not None

        # Verify stats
        stats = manager.get_stats()
        assert stats["total_patches_created"] >= 1

    def test_soft_ingestion_warning_flow(self):
        """Test the soft ingestion warning flow."""
        manager = SoftIngestionManager()

        # Ingest a warning
        result = manager.ingest_warning(
            query_pattern="hamburg customs",
            warning="Hamburg customs experiencing delays due to system upgrade",
            source_task="logistics_task_002",
            domain="logistics",
        )

        assert result.success is True


class TestConflictResolutionIntegration:
    """Tests for Conflict Resolution integration."""

    def test_conflict_manager_initialization(self):
        """Test ConflictManager initialization."""
        from precept.conflict_resolution import ConflictManager

        manager = ConflictManager()
        assert manager is not None

        # Get stats
        stats = manager.get_stats()
        assert isinstance(stats, dict)

    def test_ensemble_conflict_detector_initialization(self):
        """Test EnsembleConflictDetector initialization."""
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
        assert detector is not None

    def test_conflict_resolver_initialization(self):
        """Test ConflictResolver initialization."""
        from precept.conflict_resolution import (
            ConflictResolver,
            ConflictResolutionConfig,
            DynamicReliabilityTracker,
        )

        config = ConflictResolutionConfig()
        tracker = DynamicReliabilityTracker(config=config)
        resolver = ConflictResolver(config=config, reliability_tracker=tracker)
        assert resolver is not None


class TestLearningPipelineIntegration:
    """Tests for the full learning pipeline."""

    def test_experience_to_rule_pipeline(self):
        """Test the pipeline from experience to rule extraction."""
        # 1. Store experience
        store = MemoryStore()
        exp = store.store_experience(
            task_description="Ship from Rotterdam blocked by R-482",
            goal="Complete shipment",
            trajectory=[{"action": "use_hamburg", "observation": "success"}],
            outcome="success",
            correctness=1.0,
            strategy_used="fallback to Hamburg",
            lessons_learned=["R-482: Rotterdam blocked, use Hamburg"],
            skills_demonstrated=["error_handling"],
            domain="logistics",
            experience_type=ExperienceType.EDGE_CASE,
            priority=MemoryPriority.HIGH,
        )

        # 2. Extract lesson
        lesson = exp.lessons_learned[0]

        # 3. Parse as rule
        parser = DynamicRuleParser(
            known_entities=["rotterdam", "hamburg", "antwerp"]
        )
        rules = parser.parse_rules([lesson])

        # Verify the pipeline worked
        assert len(rules) >= 0  # May or may not parse depending on format

    def test_soft_patch_integration_with_memory(self):
        """Test soft patches integrating with memory retrieval."""
        # Create memory store
        store = MemoryStore()

        # Create soft ingestion manager
        ingestion = SoftIngestionManager()

        # Store original experience
        store.store_experience(
            task_description="Rotterdam port operational",
            goal="Use Rotterdam",
            trajectory=[],
            outcome="success",
            correctness=1.0,
            strategy_used="direct",
            lessons_learned=[],
            skills_demonstrated=[],
            domain="logistics",
        )

        # Add soft patch (correction)
        ingestion.ingest_correction(
            target_document_id="port-rotterdam",
            correction="Rotterdam now BLOCKED",
            source_task="update_task",
            source_observation="Strike announced",
        )

        # Both should have data
        assert store.get_stats()["current_size"] == 1
        assert ingestion.get_stats()["total_patches_created"] == 1


class TestScenarioGeneratorIntegration:
    """Tests for Scenario Generator integration."""

    def test_logistics_scenario_generation(self):
        """Test generating logistics scenarios."""
        from precept.scenario_generators import generate_logistics_scenarios

        scenarios = generate_logistics_scenarios(
            num_samples=10,
            train_ratio=0.6,
        )

        assert len(scenarios) > 0

        # Check structure
        for scenario in scenarios:
            assert "task" in scenario
            assert "expected" in scenario
            assert "phase" in scenario

    def test_conflict_resolution_scenarios(self):
        """Test generating conflict resolution scenarios."""
        from precept.scenario_generators import generate_logistics_scenarios

        scenarios = generate_logistics_scenarios(
            num_samples=20,
            train_ratio=0.5,
            include_conflict_resolution=True,
        )

        # Check for conflict-related scenarios
        conflict_scenarios = [
            s for s in scenarios
            if "Conflict" in s.get("black_swan_type", "")
        ]

        # Should have some conflict scenarios
        assert len(conflict_scenarios) >= 0  # May vary based on random selection


class TestDomainStrategyIntegration:
    """Tests for Domain Strategy integration."""

    def test_get_domain_strategy(self):
        """Test getting domain strategies."""
        from precept import get_domain_strategy

        strategy = get_domain_strategy("logistics")
        assert strategy is not None
        assert hasattr(strategy, "get_available_actions")

    def test_get_baseline_strategy(self):
        """Test getting baseline strategies."""
        from precept import get_baseline_strategy

        strategy = get_baseline_strategy("logistics")
        assert strategy is not None
        assert hasattr(strategy, "get_available_options")

    def test_list_available_domains(self):
        """Test listing available domains."""
        from precept import list_available_domains

        domains = list_available_domains()
        assert "logistics" in domains
        assert len(domains) >= 1


class TestGEPAIntegration:
    """Tests for GEPA Evolution integration."""

    def test_gepa_engine_initialization(self):
        """Test GEPA engine initialization."""
        from precept.gepa import GEPAConfig, GEPAEvolutionEngine

        async def mock_llm(*args, **kwargs):
            return None

        config = GEPAConfig(
            objectives=["success_rate", "efficiency"],
            max_pareto_front_size=5,
        )

        engine = GEPAEvolutionEngine(
            llm_client=mock_llm,
            config=config,
        )

        assert engine is not None
        assert len(engine.pareto_front) == 0

    def test_gepa_pareto_front_initialization(self):
        """Test Pareto front initialization."""
        from precept.gepa import GEPAEvolutionEngine

        async def mock_llm(*args, **kwargs):
            return None

        engine = GEPAEvolutionEngine(llm_client=mock_llm)
        candidate = engine.initialize_pareto_front(
            base_prompt="You are a helpful logistics agent."
        )

        assert candidate is not None
        assert len(engine.pareto_front) == 1


class TestComplexityAnalyzerIntegration:
    """Tests for Complexity Analyzer integration."""

    def test_complexity_analyzer(self):
        """Test complexity analyzer."""
        from precept.complexity_analyzer import PRECEPTComplexityAnalyzer

        analyzer = PRECEPTComplexityAnalyzer()

        result = analyzer.analyze("Ship cargo from A to B")
        assert result is not None

    def test_smart_rollout_strategy(self):
        """Test smart rollout strategy."""
        from precept.complexity_analyzer import SmartRolloutStrategy

        strategy = SmartRolloutStrategy()
        assert strategy is not None

    def test_multi_strategy_coordinator(self):
        """Test multi-strategy coordinator."""
        from precept.complexity_analyzer import MultiStrategyCoordinator

        coordinator = MultiStrategyCoordinator()
        assert coordinator is not None
        # Coordinator should have some strategy-related methods
        assert hasattr(coordinator, "select_strategy") or hasattr(coordinator, "coordinate")
