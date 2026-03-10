"""
Comprehensive Unit Tests for precept.gepa module.

Tests GEPAEvolutionEngine, GEPAReflection, GEPAMutation, GEPAParetoCandidate,
and GEPAConfig with detailed functionality coverage.
"""

import pytest

from precept.gepa import (
    GEPAConfig,
    GEPAEvolutionEngine,
    GEPAMutation,
    GEPAParetoCandidate,
    GEPAReflection,
)


# =============================================================================
# TEST GEPA PYDANTIC MODELS
# =============================================================================


class TestGEPAReflection:
    """Tests for GEPAReflection model."""

    def test_reflection_creation(self):
        """Test creating a GEPA reflection."""
        reflection = GEPAReflection(
            diagnosis="Agent failed to handle port closure",
            root_cause="Lack of fallback strategy",
            suggested_fix="Add fallback port logic",
            confidence=0.85,
            affected_objectives=["success_rate", "adaptation_speed"],
        )
        assert reflection.diagnosis == "Agent failed to handle port closure"
        assert reflection.root_cause == "Lack of fallback strategy"
        assert reflection.confidence == 0.85
        assert len(reflection.affected_objectives) == 2

    def test_reflection_default_objectives(self):
        """Test reflection with default objectives."""
        reflection = GEPAReflection(
            diagnosis="Test",
            root_cause="Test",
            suggested_fix="Test",
            confidence=0.5,
        )
        assert reflection.affected_objectives == []


class TestGEPAMutation:
    """Tests for GEPAMutation model."""

    def test_mutation_creation(self):
        """Test creating a GEPA mutation."""
        mutation = GEPAMutation(
            mutated_prompt="You are a logistics agent. Handle port closures gracefully.",
            mutation_type="addition",
            lessons_incorporated=["Check port status", "Use fallback ports"],
            expected_improvement="Better handling of blocked ports",
            parent_prompt_id="parent-001",
        )
        assert "logistics agent" in mutation.mutated_prompt
        assert mutation.mutation_type == "addition"
        assert len(mutation.lessons_incorporated) == 2

    def test_mutation_types(self):
        """Test different mutation types."""
        mutation_types = ["addition", "removal", "rewrite", "merge"]
        for mt in mutation_types:
            mutation = GEPAMutation(
                mutated_prompt="Test prompt",
                mutation_type=mt,
                lessons_incorporated=[],
                expected_improvement="Test",
            )
            assert mutation.mutation_type == mt


class TestGEPAParetoCandidate:
    """Tests for GEPAParetoCandidate model."""

    def test_candidate_creation(self):
        """Test creating a Pareto candidate."""
        candidate = GEPAParetoCandidate(
            prompt_id="candidate-001",
            prompt_text="You are an expert logistics agent.",
            scores={"success_rate": 0.8, "step_efficiency": 0.7},
            generation=1,
            parent_id="parent-001",
        )
        assert candidate.prompt_id == "candidate-001"
        assert candidate.generation == 1
        assert candidate.scores["success_rate"] == 0.8

    def test_candidate_dominates_true(self):
        """Test Pareto dominance when A dominates B."""
        candidate_a = GEPAParetoCandidate(
            prompt_id="a",
            prompt_text="A",
            scores={"obj1": 0.9, "obj2": 0.8},
            generation=1,
        )
        candidate_b = GEPAParetoCandidate(
            prompt_id="b",
            prompt_text="B",
            scores={"obj1": 0.7, "obj2": 0.6},
            generation=1,
        )

        assert candidate_a.dominates(candidate_b)
        assert not candidate_b.dominates(candidate_a)

    def test_candidate_dominates_false_equal(self):
        """Test Pareto dominance when candidates are equal."""
        candidate_a = GEPAParetoCandidate(
            prompt_id="a",
            prompt_text="A",
            scores={"obj1": 0.8, "obj2": 0.8},
            generation=1,
        )
        candidate_b = GEPAParetoCandidate(
            prompt_id="b",
            prompt_text="B",
            scores={"obj1": 0.8, "obj2": 0.8},
            generation=1,
        )

        assert not candidate_a.dominates(candidate_b)
        assert not candidate_b.dominates(candidate_a)

    def test_candidate_dominates_false_tradeoff(self):
        """Test Pareto dominance with trade-off (no dominance)."""
        candidate_a = GEPAParetoCandidate(
            prompt_id="a",
            prompt_text="A",
            scores={"obj1": 0.9, "obj2": 0.6},
            generation=1,
        )
        candidate_b = GEPAParetoCandidate(
            prompt_id="b",
            prompt_text="B",
            scores={"obj1": 0.6, "obj2": 0.9},
            generation=1,
        )

        # Neither dominates the other (trade-off)
        assert not candidate_a.dominates(candidate_b)
        assert not candidate_b.dominates(candidate_a)

    def test_candidate_get_average_score(self):
        """Test calculating average score."""
        candidate = GEPAParetoCandidate(
            prompt_id="a",
            prompt_text="A",
            scores={"obj1": 0.8, "obj2": 0.6, "obj3": 0.7},
            generation=1,
        )

        avg = candidate.get_average_score()
        assert abs(avg - 0.7) < 0.01

    def test_candidate_average_score_empty(self):
        """Test average score with empty scores."""
        candidate = GEPAParetoCandidate(
            prompt_id="a",
            prompt_text="A",
            scores={},
            generation=1,
        )

        assert candidate.get_average_score() == 0.0


class TestGEPAConfig:
    """Tests for GEPAConfig model."""

    def test_config_default_values(self):
        """Test default configuration values."""
        config = GEPAConfig()
        assert "success_rate" in config.objectives
        assert "step_efficiency" in config.objectives
        assert config.max_pareto_front_size == 10
        assert config.selection_noise == 0.2
        assert config.min_reflection_confidence == 0.5

    def test_config_custom_values(self):
        """Test custom configuration values."""
        config = GEPAConfig(
            objectives=["accuracy", "speed"],
            max_pareto_front_size=20,
            selection_noise=0.1,
        )
        assert config.objectives == ["accuracy", "speed"]
        assert config.max_pareto_front_size == 20
        assert config.selection_noise == 0.1


# =============================================================================
# TEST GEPA EVOLUTION ENGINE
# =============================================================================


class TestGEPAEvolutionEngineInitialization:
    """Tests for GEPAEvolutionEngine initialization."""

    @pytest.fixture
    def mock_llm_client(self):
        """Create a mock LLM client."""
        async def mock_client(system_prompt, user_prompt, response_model):
            if response_model == GEPAReflection:
                return GEPAReflection(
                    diagnosis="Test diagnosis",
                    root_cause="Test cause",
                    suggested_fix="Test fix",
                    confidence=0.8,
                    affected_objectives=["success_rate"],
                )
            elif response_model == GEPAMutation:
                return GEPAMutation(
                    mutated_prompt="Mutated prompt",
                    mutation_type="addition",
                    lessons_incorporated=["lesson1"],
                    expected_improvement="improvement",
                )
            return None
        return mock_client

    def test_engine_creation(self, mock_llm_client):
        """Test creating a GEPA evolution engine."""
        engine = GEPAEvolutionEngine(llm_client=mock_llm_client)
        assert engine.pareto_front == []
        assert engine.generation == 0
        assert engine.stats["total_mutations"] == 0

    def test_engine_with_config(self, mock_llm_client):
        """Test creating engine with custom config."""
        config = GEPAConfig(max_pareto_front_size=5)
        engine = GEPAEvolutionEngine(llm_client=mock_llm_client, config=config)
        assert engine.config.max_pareto_front_size == 5

    def test_engine_with_rules_getter(self, mock_llm_client):
        """Test creating engine with learned rules getter."""
        rules_getter = lambda: ["rule1", "rule2"]
        engine = GEPAEvolutionEngine(
            llm_client=mock_llm_client,
            learned_rules_getter=rules_getter,
        )
        assert engine.learned_rules_getter() == ["rule1", "rule2"]


class TestGEPAEvolutionEngineParetoFront:
    """Tests for GEPAEvolutionEngine Pareto front management."""

    @pytest.fixture
    def engine(self):
        """Create a GEPA engine for testing."""
        async def mock_client(*args, **kwargs):
            return None
        return GEPAEvolutionEngine(llm_client=mock_client)

    def test_initialize_pareto_front(self, engine):
        """Test initializing Pareto front with base prompt."""
        base_prompt = "You are a helpful logistics agent."
        candidate = engine.initialize_pareto_front(base_prompt)

        assert len(engine.pareto_front) == 1
        assert candidate.prompt_text == base_prompt
        assert candidate.generation == 0

    def test_initialize_pareto_front_with_scores(self, engine):
        """Test initializing Pareto front with initial scores."""
        base_prompt = "Test prompt"
        initial_scores = {"success_rate": 0.7, "efficiency": 0.6}
        candidate = engine.initialize_pareto_front(base_prompt, initial_scores)

        assert candidate.scores["success_rate"] == 0.7
        assert candidate.scores["efficiency"] == 0.6

    def test_pareto_front_prompt_id_generation(self, engine):
        """Test that prompt IDs are generated correctly."""
        candidate = engine.initialize_pareto_front("Test prompt")
        assert candidate.prompt_id is not None
        assert len(candidate.prompt_id) > 0


class TestGEPAEvolutionEngineEvaluation:
    """Tests for GEPAEvolutionEngine candidate evaluation."""

    @pytest.fixture
    def engine(self):
        """Create a GEPA engine for testing."""
        async def mock_client(*args, **kwargs):
            return None
        engine = GEPAEvolutionEngine(llm_client=mock_client)
        engine.initialize_pareto_front("Base prompt")
        return engine

    def test_create_candidate(self, engine):
        """Test creating a candidate from mutation."""
        mutation = GEPAMutation(
            mutated_prompt="New mutated prompt",
            mutation_type="addition",
            lessons_incorporated=["lesson1"],
            expected_improvement="Better performance",
        )
        scores = {"success_rate": 0.9, "step_efficiency": 0.8}

        candidate = engine.create_candidate(mutation, scores, parent_id="parent-001")

        assert candidate.prompt_text == mutation.mutated_prompt
        assert candidate.scores == scores
        assert candidate.parent_id == "parent-001"
        assert candidate.generation == 1

    def test_evaluate_candidate_scores(self, engine):
        """Test evaluating candidate scores from results."""
        task_results = [
            {"success": True, "steps": 3},
            {"success": True, "steps": 5},
            {"success": False, "steps": 7},
        ]

        scores = engine.evaluate_candidate(task_results)

        assert "success_rate" in scores
        assert scores["success_rate"] == pytest.approx(2/3, rel=0.01)


class TestGEPAEvolutionEngineStats:
    """Tests for GEPAEvolutionEngine statistics."""

    def test_stats_initialization(self):
        """Test initial statistics."""
        async def mock_client(*args, **kwargs):
            return None
        engine = GEPAEvolutionEngine(llm_client=mock_client)

        assert engine.stats["total_mutations"] == 0
        assert engine.stats["successful_mutations"] == 0
        assert engine.stats["pareto_updates"] == 0
        assert engine.stats["reflections_performed"] == 0
        assert engine.stats["candidates_dominated"] == 0


class TestGEPAEvolutionEngineAsync:
    """Async tests for GEPAEvolutionEngine."""

    @pytest.fixture
    def mock_llm_client(self):
        """Create a mock LLM client for async tests."""
        async def mock_client(system_prompt, user_prompt, response_model):
            if response_model == GEPAReflection:
                return GEPAReflection(
                    diagnosis="Diagnosis from mock",
                    root_cause="Root cause from mock",
                    suggested_fix="Fix from mock",
                    confidence=0.9,
                    affected_objectives=["success_rate"],
                )
            elif response_model == GEPAMutation:
                return GEPAMutation(
                    mutated_prompt="Mutated prompt from mock",
                    mutation_type="rewrite",
                    lessons_incorporated=["mock lesson"],
                    expected_improvement="Better results",
                )
            return None
        return mock_client

    @pytest.mark.asyncio
    async def test_reflect_on_trajectory(self, mock_llm_client):
        """Test reflecting on a trajectory."""
        engine = GEPAEvolutionEngine(llm_client=mock_llm_client)

        trajectory = [
            {"thought": "Planning shipment", "action": "check_port", "observation": "Port blocked"},
            {"thought": "Need alternative", "action": "find_alternative", "observation": "Found Hamburg"},
        ]

        reflection = await engine.reflect_on_trajectory(
            trajectory=trajectory,
            task="Ship cargo from Rotterdam",
            success=False,
            prompt_used="You are a logistics agent.",
        )

        assert reflection.diagnosis == "Diagnosis from mock"
        assert reflection.confidence == 0.9
        assert engine.stats["reflections_performed"] == 1

    @pytest.mark.asyncio
    async def test_mutate_prompt(self, mock_llm_client):
        """Test mutating a prompt."""
        engine = GEPAEvolutionEngine(llm_client=mock_llm_client)
        engine.initialize_pareto_front("Base prompt")

        reflections = [
            GEPAReflection(
                diagnosis="Test diagnosis",
                root_cause="Test cause",
                suggested_fix="Test fix",
                confidence=0.8,
                affected_objectives=["success_rate"],
            )
        ]

        mutation = await engine.mutate_prompt(
            parent_prompt="You are a logistics agent.",
            parent_id="parent-001",
            reflections=reflections,
            learned_rules=["R-482: Rotterdam blocked, use Hamburg"],
        )

        assert mutation.mutated_prompt == "Mutated prompt from mock"
        assert mutation.mutation_type == "rewrite"
        assert engine.stats["total_mutations"] == 1
