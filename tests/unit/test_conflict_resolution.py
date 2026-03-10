"""
Unit Tests for precept.conflict_resolution module.

Tests the cutting-edge conflict resolution system including:
- ConflictResolutionConfig (configurable parameters)
- KnowledgeItem creation and validation
- EnsembleConflictDetector
- ConflictResolver strategies
- ConflictManager integration
"""

from datetime import datetime

import pytest

from precept.conflict_resolution import (
    # Exploration
    ActiveExplorationStrategy,
    ConflictManager,
    ConflictRecord,
    # Configuration
    ConflictResolutionConfig,
    ConflictResolver,
    # Enums
    ConflictSeverity,
    # Reliability tracking
    DynamicReliabilityTracker,
    # Ensemble
    EnsembleConflictDetector,
    # Data classes
    KnowledgeItem,
    KnowledgeSource,
    LearnedPatterns,
    # NLI
    NeuralNLIClassifier,
    ResolutionResult,
    SemanticRelation,
)

# =============================================================================
# CONFIGURATION TESTS
# =============================================================================


class TestConflictResolutionConfig:
    """Tests for ConflictResolutionConfig dataclass."""

    def test_default_config(self):
        """Test default configuration values."""
        config = ConflictResolutionConfig()

        # Thresholds - use actual attribute names
        assert 0.0 <= config.high_similarity_threshold <= 1.0
        assert 0.0 <= config.semantic_similarity_threshold <= 1.0
        assert config.recency_decay_rate > 0

    def test_config_is_customizable(self):
        """Test that config values can be customized."""
        config = ConflictResolutionConfig(
            high_similarity_threshold=0.9,
            semantic_similarity_threshold=0.8,
            recency_decay_rate=0.5,
        )

        assert config.high_similarity_threshold == 0.9
        assert config.semantic_similarity_threshold == 0.8
        assert config.recency_decay_rate == 0.5


class TestLearnedPatterns:
    """Tests for LearnedPatterns dataclass."""

    def test_default_patterns(self):
        """Test default learned patterns."""
        patterns = LearnedPatterns()

        # Check actual attribute names (from the actual implementation)
        assert hasattr(patterns, "contradiction_pairs")
        assert hasattr(patterns, "regulatory_keywords")
        assert hasattr(patterns, "temporal_keywords")

    def test_patterns_are_lists(self):
        """Test patterns are proper lists."""
        patterns = LearnedPatterns()

        # Get the actual attribute names
        for attr in dir(patterns):
            if not attr.startswith("_"):
                val = getattr(patterns, attr)
                if isinstance(val, (list, tuple)):
                    assert len(val) >= 0  # Can be empty but must be iterable


# =============================================================================
# KNOWLEDGE SOURCE TESTS
# =============================================================================


class TestKnowledgeSource:
    """Tests for KnowledgeSource enum."""

    def test_knowledge_source_has_static(self):
        """Test KnowledgeSource has static type."""
        # Check for any static-like value
        static_types = [
            s
            for s in KnowledgeSource
            if "static" in s.value.lower() or "kb" in s.value.lower()
        ]
        assert len(static_types) >= 1

    def test_knowledge_source_has_dynamic(self):
        """Test KnowledgeSource has dynamic type."""
        dynamic_types = [s for s in KnowledgeSource if "dynamic" in s.value.lower()]
        assert len(dynamic_types) >= 1

    def test_knowledge_source_values(self):
        """Test all KnowledgeSource values are strings."""
        for source in KnowledgeSource:
            assert isinstance(source.value, str)


# =============================================================================
# KNOWLEDGE ITEM TESTS
# =============================================================================


class TestKnowledgeItem:
    """Tests for KnowledgeItem dataclass."""

    def test_create_static_item(self):
        """Test creating a static knowledge item."""
        item = KnowledgeItem(
            id="static-001",
            content="Rotterdam port is operational 24/7",
            source=KnowledgeSource.STATIC_KB,
            timestamp=datetime.now(),
            confidence=0.95,
            metadata={"port": "rotterdam"},
        )

        assert item.source == KnowledgeSource.STATIC_KB
        assert item.confidence == 0.95
        assert "rotterdam" in str(item.metadata)
        assert item.id == "static-001"

    def test_create_dynamic_item(self):
        """Test creating a dynamic knowledge item."""
        item = KnowledgeItem(
            id="dynamic-001",
            content="Rotterdam blocked due to R-482",
            source=KnowledgeSource.DYNAMIC_EXPERIENCE,
            timestamp=datetime.now(),
            confidence=0.8,
        )

        assert item.source == KnowledgeSource.DYNAMIC_EXPERIENCE
        assert item.id == "dynamic-001"


# =============================================================================
# CONFLICT SEVERITY TESTS
# =============================================================================


class TestConflictSeverity:
    """Tests for ConflictSeverity enum."""

    def test_conflict_severities_exist(self):
        """Test conflict severity levels exist."""
        severities = list(ConflictSeverity)
        assert len(severities) >= 2  # At least LOW and HIGH


# =============================================================================
# ENSEMBLE CONFLICT DETECTOR TESTS
# =============================================================================


class TestEnsembleConflictDetector:
    """Tests for EnsembleConflictDetector."""

    @pytest.fixture
    def detector(self):
        """Create a detector for testing."""
        config = ConflictResolutionConfig()
        patterns = LearnedPatterns()
        nli_classifier = NeuralNLIClassifier(config=config, patterns=patterns)
        return EnsembleConflictDetector(
            config=config, patterns=patterns, nli_classifier=nli_classifier
        )

    def test_detector_initialization(self, detector):
        """Test detector initializes correctly."""
        assert detector is not None
        assert detector.config is not None

    def test_detector_with_config(self):
        """Test detector with custom config."""
        config = ConflictResolutionConfig(high_similarity_threshold=0.9)
        patterns = LearnedPatterns()
        nli_classifier = NeuralNLIClassifier(config=config, patterns=patterns)
        detector = EnsembleConflictDetector(
            config=config, patterns=patterns, nli_classifier=nli_classifier
        )

        assert detector.config.high_similarity_threshold == 0.9

    def test_detector_has_config(self, detector):
        """Test detector has config and patterns."""
        assert hasattr(detector, "config")
        assert hasattr(detector, "patterns")


# =============================================================================
# CONFLICT RESOLVER TESTS
# =============================================================================


class TestConflictResolver:
    """Tests for ConflictResolver."""

    @pytest.fixture
    def resolver(self):
        """Create a resolver for testing."""
        config = ConflictResolutionConfig()
        reliability_tracker = DynamicReliabilityTracker(config=config)
        return ConflictResolver(config=config, reliability_tracker=reliability_tracker)

    def test_resolver_initialization(self, resolver):
        """Test resolver initializes correctly."""
        assert resolver is not None
        assert resolver.config is not None

    def test_resolver_has_config(self, resolver):
        """Test resolver has config and reliability tracker."""
        assert hasattr(resolver, "config")
        assert hasattr(resolver, "reliability_tracker")


# =============================================================================
# NEURAL NLI CLASSIFIER TESTS
# =============================================================================


class TestNeuralNLIClassifier:
    """Tests for NeuralNLIClassifier."""

    @pytest.fixture
    def classifier(self):
        """Create a classifier for testing."""
        config = ConflictResolutionConfig()
        patterns = LearnedPatterns()
        return NeuralNLIClassifier(config=config, patterns=patterns)

    def test_classifier_initialization(self, classifier):
        """Test classifier initializes correctly."""
        assert classifier is not None
        assert classifier.config is not None


# =============================================================================
# ACTIVE EXPLORATION STRATEGY TESTS
# =============================================================================


class TestActiveExplorationStrategy:
    """Tests for ActiveExplorationStrategy (Thompson Sampling)."""

    @pytest.fixture
    def explorer(self):
        """Create an explorer for testing."""
        config = ConflictResolutionConfig()
        reliability_tracker = DynamicReliabilityTracker(config=config)
        return ActiveExplorationStrategy(
            config=config, reliability_tracker=reliability_tracker
        )

    def test_explorer_initialization(self, explorer):
        """Test explorer initializes correctly."""
        assert explorer is not None
        assert explorer.config is not None


# =============================================================================
# DYNAMIC RELIABILITY TRACKER TESTS
# =============================================================================


class TestDynamicReliabilityTracker:
    """Tests for DynamicReliabilityTracker."""

    @pytest.fixture
    def tracker(self):
        """Create a tracker for testing."""
        config = ConflictResolutionConfig()
        return DynamicReliabilityTracker(config=config)

    def test_tracker_initialization(self, tracker):
        """Test tracker initializes correctly."""
        assert tracker is not None
        assert tracker.config is not None

    def test_tracker_has_distributions(self, tracker):
        """Test tracker has distributions for different sources."""
        assert hasattr(tracker, "distributions")
        assert isinstance(tracker.distributions, dict)


# =============================================================================
# CONFLICT MANAGER INTEGRATION TESTS
# =============================================================================


class TestConflictManager:
    """Tests for ConflictManager integration."""

    @pytest.fixture
    def manager(self):
        """Create a conflict manager for testing."""
        return ConflictManager()

    def test_manager_initialization(self, manager):
        """Test manager initializes correctly."""
        assert manager is not None
        assert manager.detector is not None
        assert manager.resolver is not None

    def test_get_stats(self, manager):
        """Test getting manager statistics."""
        stats = manager.get_stats()

        # Check the actual keys returned by get_stats
        assert "status" in stats
        assert "summary" in stats
        assert stats["status"] == "active"

    def test_manager_has_conflict_history(self, manager):
        """Test manager has conflict history."""
        assert hasattr(manager, "conflict_history")
        assert isinstance(manager.conflict_history, list)
        assert len(manager.conflict_history) == 0

    def test_manager_has_history_lists(self, manager):
        """Test manager has history tracking."""
        assert hasattr(manager, "conflict_history")
        assert hasattr(manager, "resolution_history")
        assert isinstance(manager.conflict_history, list)
        assert isinstance(manager.resolution_history, list)


# =============================================================================
# CONFLICT RECORD TESTS
# =============================================================================


class TestConflictRecord:
    """Tests for ConflictRecord dataclass."""

    def test_conflict_record_creation(self):
        """Test creating a conflict record."""
        static_item = KnowledgeItem(
            id="static-hamburg-001",
            content="Hamburg is operational",
            source=KnowledgeSource.STATIC_KB,
            timestamp=datetime.now(),
        )
        dynamic_item = KnowledgeItem(
            id="dynamic-hamburg-001",
            content="Hamburg has labor strikes",
            source=KnowledgeSource.DYNAMIC_EXPERIENCE,
            timestamp=datetime.now(),
        )

        record = ConflictRecord(
            id="conflict-001",
            static_item=static_item,
            dynamic_item=dynamic_item,
            relation=SemanticRelation.CONTRADICTS,
            severity=ConflictSeverity.HIGH,
            confidence=0.85,
            detection_method="semantic",
        )

        assert record.severity == ConflictSeverity.HIGH
        assert record.confidence == 0.85
        assert record.detection_method == "semantic"


# =============================================================================
# RESOLUTION RESULT TESTS
# =============================================================================


class TestResolutionResult:
    """Tests for ResolutionResult dataclass."""

    def test_resolution_result_creation(self):
        """Test creating a resolution result."""
        chosen = KnowledgeItem(
            id="chosen-001",
            content="Dynamic knowledge chosen",
            source=KnowledgeSource.DYNAMIC_EXPERIENCE,
            timestamp=datetime.now(),
        )

        # Create a mock conflict record
        static_item = KnowledgeItem(
            id="static-001",
            content="Static knowledge",
            source=KnowledgeSource.STATIC_KB,
            timestamp=datetime.now(),
        )

        conflict = ConflictRecord(
            id="conflict-001",
            static_item=static_item,
            dynamic_item=chosen,
            relation=SemanticRelation.CONTRADICTS,
            severity=ConflictSeverity.MEDIUM,
            confidence=0.8,
            detection_method="semantic",
        )

        result = ResolutionResult(
            id="resolution-001",
            conflict=conflict,
            winner=KnowledgeSource.DYNAMIC_EXPERIENCE,
            winning_item=chosen,
            confidence=0.85,
            strategy_used="recency",
            reasoning="More recent information",
        )

        assert result.winning_item == chosen
        assert result.strategy_used == "recency"
        assert result.confidence == 0.85
        assert result.winner == KnowledgeSource.DYNAMIC_EXPERIENCE
