"""
Comprehensive Unit Tests for precept.ingestion module.

Tests DocumentChunk, SoftIngestionManager, FeedbackIngestion,
and the three-stream ingestion architecture.
"""

import time

import pytest

from precept.ingestion import (
    DocumentChunk,
    IngestionPriority,
    IngestionType,
    SoftIngestionManager,
    SoftPatch,
)


# =============================================================================
# TEST INGESTION ENUMS
# =============================================================================


class TestIngestionType:
    """Tests for IngestionType enum."""

    def test_ingestion_type_values(self):
        """Test all ingestion type values exist."""
        assert IngestionType.HARD.value == "hard"
        assert IngestionType.SOFT.value == "soft"
        assert IngestionType.FEEDBACK.value == "feedback"


class TestIngestionPriority:
    """Tests for IngestionPriority enum."""

    def test_ingestion_priority_values(self):
        """Test all ingestion priority values exist."""
        assert IngestionPriority.CRITICAL.value == "critical"
        assert IngestionPriority.HIGH.value == "high"
        assert IngestionPriority.NORMAL.value == "normal"
        assert IngestionPriority.LOW.value == "low"


# =============================================================================
# TEST DOCUMENT CHUNK
# =============================================================================


class TestDocumentChunk:
    """Tests for DocumentChunk dataclass."""

    def test_document_chunk_creation(self):
        """Test creating a document chunk."""
        chunk = DocumentChunk(
            id="chunk-001",
            content="Rotterdam port handles container ships",
            source="logistics_kb.pdf",
            metadata={"page": 1, "section": "ports"},
        )

        assert chunk.id == "chunk-001"
        assert chunk.source == "logistics_kb.pdf"
        assert "Rotterdam" in chunk.content

    def test_document_chunk_to_dict(self):
        """Test converting chunk to dictionary."""
        chunk = DocumentChunk(
            id="chunk-001",
            content="Test content",
            source="test.pdf",
        )

        data = chunk.to_dict()
        assert data["id"] == "chunk-001"
        assert data["content"] == "Test content"
        assert data["source"] == "test.pdf"
        assert "created_at" in data

    def test_document_chunk_default_metadata(self):
        """Test default metadata is empty dict."""
        chunk = DocumentChunk(
            id="chunk-001",
            content="Test",
            source="test.pdf",
        )

        assert chunk.metadata == {}

    def test_document_chunk_with_embedding(self):
        """Test chunk with embedding."""
        chunk = DocumentChunk(
            id="chunk-001",
            content="Test",
            source="test.pdf",
            embedding=[0.1, 0.2, 0.3],
        )

        assert chunk.embedding == [0.1, 0.2, 0.3]


# =============================================================================
# TEST SOFT PATCH
# =============================================================================


class TestSoftPatch:
    """Tests for SoftPatch dataclass."""

    def test_soft_patch_creation(self):
        """Test creating a soft patch."""
        patch = SoftPatch(
            id="patch-001",
            target_document_id="doc-001",
            patch_content="Rotterdam port is currently BLOCKED",
            source_task="test_task",
            source_observation="Port closed due to strike",
            priority=IngestionPriority.HIGH,
        )

        assert patch.id == "patch-001"
        assert patch.target_document_id == "doc-001"
        assert patch.priority == IngestionPriority.HIGH

    def test_soft_patch_properties(self):
        """Test SoftPatch properties."""
        patch = SoftPatch(
            id="patch-001",
            patch_content="Test correction",
            source_task="test_task",
            source_observation="Test reason",
        )

        # Test usefulness score
        assert patch.usefulness_score >= 0

        # Test is_expired
        assert not patch.is_expired()

        # Test to_context_string
        context = patch.to_context_string()
        assert isinstance(context, str)


# =============================================================================
# TEST SOFT INGESTION MANAGER
# =============================================================================


class TestSoftIngestionManager:
    """Tests for SoftIngestionManager class."""

    @pytest.fixture
    def manager(self):
        """Create a soft ingestion manager."""
        return SoftIngestionManager()

    def test_manager_creation(self, manager):
        """Test creating a soft ingestion manager."""
        assert manager is not None
        assert manager.patches == {}

    def test_ingest_correction(self, manager):
        """Test ingesting a correction."""
        result = manager.ingest_correction(
            target_document_id="doc-001",
            correction="Updated information",
            source_task="test_task",
            source_observation="New data available",
        )

        assert result.success is True
        assert result.patch_id is not None

    def test_ingest_correction_with_confidence(self, manager):
        """Test ingesting a correction with confidence."""
        result = manager.ingest_correction(
            target_document_id="doc-001",
            correction="Critical update",
            source_task="test_task",
            source_observation="Emergency change",
            confidence=0.95,
        )

        assert result.success is True
        patch = manager.patches.get(result.patch_id)
        if patch:
            assert patch.confidence == 0.95

    def test_ingest_warning(self, manager):
        """Test ingesting a warning."""
        result = manager.ingest_warning(
            query_pattern="rotterdam",
            warning="Rotterdam port is blocked due to strike",
            source_task="logistics_task",
            priority=IngestionPriority.HIGH,
        )

        assert result.success is True

    def test_get_stats(self, manager):
        """Test getting ingestion statistics."""
        manager.ingest_correction(
            target_document_id="doc-001",
            correction="Test",
            source_task="test",
            source_observation="test",
        )

        stats = manager.get_stats()
        assert "total_patches_created" in stats
        assert stats["total_patches_created"] >= 1

    def test_cleanup_expired(self, manager):
        """Test cleaning up expired patches."""
        # This should not raise any errors
        expired_count = manager.cleanup_expired()
        assert expired_count >= 0


# =============================================================================
# TEST FEEDBACK INGESTION (via FeedbackIngestionManager)
# =============================================================================


class TestFeedbackIngestion:
    """Tests for feedback ingestion functionality."""

    def test_feedback_ingestion_manager_import(self):
        """Test that FeedbackIngestionManager can be imported."""
        from precept.ingestion import FeedbackIngestionManager

        assert FeedbackIngestionManager is not None

    def test_feedback_ingestion_manager_creation(self):
        """Test creating a feedback ingestion manager."""
        from precept.ingestion import FeedbackIngestionManager

        manager = FeedbackIngestionManager()
        assert manager is not None


# =============================================================================
# INTEGRATION TESTS
# =============================================================================


class TestIngestionIntegration:
    """Integration tests for ingestion components."""

    def test_patch_lifecycle(self):
        """Test complete patch lifecycle."""
        manager = SoftIngestionManager()

        # Add initial patch
        result1 = manager.ingest_correction(
            target_document_id="logistics-kb",
            correction="Rotterdam port status: BLOCKED",
            source_task="test_task",
            source_observation="Strike announced",
        )

        # Add follow-up patch
        result2 = manager.ingest_correction(
            target_document_id="logistics-kb",
            correction="Rotterdam port status: OPEN (strike resolved)",
            source_task="test_task",
            source_observation="Strike ended",
        )

        assert result1.success is True
        assert result2.success is True

        # Check stats
        stats = manager.get_stats()
        assert stats["total_patches_created"] == 2

    def test_document_chunk_with_patch(self):
        """Test document chunk patching workflow."""
        # Create original chunk
        chunk = DocumentChunk(
            id="port-rotterdam",
            content="Rotterdam port is operational 24/7",
            source="logistics_kb.json",
            metadata={"port": "rotterdam", "type": "port_info"},
        )

        # Create patch manager and add correction
        manager = SoftIngestionManager()
        result = manager.ingest_correction(
            target_document_id=chunk.id,
            correction="WARNING: Rotterdam currently blocked",
            source_task="test_task",
            source_observation="Emergency closure",
        )

        assert result.success is True
        assert result.patch_id is not None

    def test_multi_domain_patches(self):
        """Test patches across multiple domains."""
        manager = SoftIngestionManager()

        # Add patches for different domains
        manager.ingest_correction(
            target_document_id="doc-1",
            correction="Logistics update",
            source_task="task1",
            source_observation="test",
            domain="logistics",
        )
        manager.ingest_correction(
            target_document_id="doc-2",
            correction="Finance update",
            source_task="task2",
            source_observation="test",
            domain="finance",
        )

        # Check stats
        stats = manager.get_stats()
        assert stats["total_patches_created"] == 2
