"""
PRECEPT Ingestion Module: Three-Stream Data Flow Architecture

PRECEPT = Planning Resilience via Experience, Context Engineering & Probing Trajectories

In PRECEPT, "Ingestion" is NOT a single event. It splits into three distinct streams:

1. HARD INGESTION (Knowledge Stream)
   - When: Pre-Deployment (Asynchronous/External Pipeline)
   - What: Raw documents, PDFs, APIs → Vector DB
   - Why: Keep heavy processing out of inference loop
   - Who: External ETL pipeline (not the agent)

2. SOFT INGESTION (Wisdom Stream)
   - When: Evo-Memory Phase (Runtime/Real-Time)
   - What: Meta-data, corrections, experience patches
   - Why: Instantly "patch" flaws without re-indexing
   - Who: Agent (during ReMem Refine step)

3. FEEDBACK INGESTION (Training Stream)
   - When: COMPASS Phase (Optimization/Batch)
   - What: Execution traces, success/failure logs
   - Why: Teach the system HOW to search better
   - Who: COMPASS optimizer

This module provides interfaces for all three ingestion streams.
"""

import hashlib
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set

from pydantic import BaseModel, Field


class IngestionType(Enum):
    """Types of ingestion in PRECEPT."""

    HARD = "hard"  # Document ingestion (external, pre-deployment)
    SOFT = "soft"  # Experience ingestion (runtime, real-time)
    FEEDBACK = "feedback"  # Trace ingestion (COMPASS, batch)


class IngestionPriority(Enum):
    """Priority levels for ingested items."""

    CRITICAL = "critical"  # Must be processed immediately
    HIGH = "high"  # Process soon
    NORMAL = "normal"  # Standard processing
    LOW = "low"  # Can be deferred


# =============================================================================
# HARD INGESTION (Document/Knowledge Stream)
# =============================================================================


@dataclass
class DocumentChunk:
    """
    A chunk of a document for hard ingestion.

    This is what gets embedded and stored in the Vector DB.
    """

    id: str
    content: str
    source: str  # Original document path/URL
    metadata: Dict[str, Any] = field(default_factory=dict)

    # Embedding (populated by embedding pipeline)
    embedding: Optional[List[float]] = None

    # Timestamps
    created_at: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "content": self.content,
            "source": self.source,
            "metadata": self.metadata,
            "created_at": self.created_at,
        }


class HardIngestionPipeline(ABC):
    """
    Abstract interface for Hard Ingestion (Document Stream).

    This runs OUTSIDE the agent's reasoning loop, typically as a
    pre-deployment ETL pipeline.

    The agent is the READER, not the LIBRARIAN.
    """

    @abstractmethod
    async def ingest_document(
        self,
        source_path: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> List[DocumentChunk]:
        """
        Ingest a document into the Vector DB.

        This is an asynchronous, external process.

        Args:
            source_path: Path to document (PDF, URL, etc.)
            metadata: Optional metadata to attach

        Returns:
            List of created document chunks
        """
        pass

    @abstractmethod
    async def ingest_batch(
        self,
        sources: List[str],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, List[DocumentChunk]]:
        """
        Ingest multiple documents in batch.

        Args:
            sources: List of source paths
            metadata: Shared metadata

        Returns:
            Dict mapping source -> chunks
        """
        pass

    @abstractmethod
    def get_ingestion_stats(self) -> Dict[str, Any]:
        """Get statistics about ingested documents."""
        pass


class DefaultHardIngestionPipeline(HardIngestionPipeline):
    """
    Default implementation of Hard Ingestion.

    ═══════════════════════════════════════════════════════════════════════════════
    USES LANGCHAIN (Battle-tested ecosystem with 80+ document loaders)
    ═══════════════════════════════════════════════════════════════════════════════

    This pipeline leverages LangChain's document loaders and text splitters:
    - PyPDFLoader, PDFPlumberLoader for PDFs
    - WebBaseLoader for web pages
    - TextLoader for plain text
    - UnstructuredMarkdownLoader for Markdown
    - And 75+ more loaders available!

    Falls back to custom processors if LangChain is unavailable.

    SUPPORTED FORMATS (via LangChain):
    - PDF (.pdf) - via LangChain's PyPDFLoader or PDFPlumberLoader
    - Text (.txt, .log, .csv) - via LangChain's TextLoader
    - Markdown (.md) - via LangChain's UnstructuredMarkdownLoader
    - Web pages (http://, https://) - via LangChain's WebBaseLoader
    - Plus: DOCX, JSON, CSV, Notion, Confluence, GitHub, and more!

    ADDING CUSTOM PROCESSORS (for unsupported formats):
    ```python
    from precept.document_processors import ProcessorRegistry, DocumentProcessor

    @ProcessorRegistry.register("custom", extensions=[".xyz"])
    class CustomProcessor(DocumentProcessor):
        async def process(self, source, **kwargs):
            # Your implementation
            ...
    ```
    """

    def __init__(
        self,
        vector_store: Any = None,  # Vector store instance
        embedding_fn: Optional[Callable[[str], List[float]]] = None,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        prefer_langchain: bool = True,  # Use LangChain when available
    ):
        self.vector_store = vector_store
        self.embedding_fn = embedding_fn
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.prefer_langchain = prefer_langchain

        # LangChain adapter (lazy init)
        self._langchain_adapter = None

        # Statistics
        self.stats = {
            "documents_ingested": 0,
            "chunks_created": 0,
            "total_tokens": 0,
            "last_ingestion": None,
            "loaders_used": {},
            "errors": [],
        }

    @property
    def langchain_adapter(self):
        """Get or create LangChain adapter."""
        if self._langchain_adapter is None:
            from .document_processors import LangChainAdapter

            self._langchain_adapter = LangChainAdapter(
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
            )
        return self._langchain_adapter

    async def ingest_document(
        self,
        source_path: str,
        metadata: Optional[Dict[str, Any]] = None,
        use_langchain: Optional[bool] = None,
        **kwargs: Any,
    ) -> List[DocumentChunk]:
        """
        Ingest a single document using LangChain (recommended) or custom processors.

        This is the main entry point for document ingestion. It:
        1. Uses LangChain loaders if available (80+ document types!)
        2. Falls back to custom processors if LangChain unavailable
        3. Stores in vector store if configured

        Args:
            source_path: Path to document, URL, or raw content
            metadata: Additional metadata to attach to chunks
            use_langchain: Force LangChain (True) or custom (False), default: auto
            **kwargs: Passed to the loader/processor

        Returns:
            List of DocumentChunk objects

        Example:
            # Auto-detect (uses LangChain if available)
            chunks = await pipeline.ingest_document("report.pdf")

            # Web scraping
            chunks = await pipeline.ingest_document("https://example.com/article")

            # Force custom processor
            chunks = await pipeline.ingest_document("data.xyz", use_langchain=False)
        """
        # Determine whether to use LangChain
        should_use_langchain = (
            use_langchain if use_langchain is not None else self.prefer_langchain
        )

        try:
            # ═══════════════════════════════════════════════════════════════════
            # OPTION 1: LangChain (RECOMMENDED - battle-tested ecosystem)
            # ═══════════════════════════════════════════════════════════════════
            if should_use_langchain and self.langchain_adapter.is_available():
                raw_chunks = await self.langchain_adapter.load_and_split(
                    source_path,
                    metadata=metadata,
                    **kwargs,
                )

                # Convert to DocumentChunk format
                chunks = []
                for raw in raw_chunks:
                    chunk = DocumentChunk(
                        id=raw["id"],
                        content=raw["content"],
                        source=source_path,
                        metadata=raw.get("metadata", {}),
                    )

                    # Generate embedding if configured
                    if self.embedding_fn:
                        chunk.embedding = self.embedding_fn(raw["content"])

                    chunks.append(chunk)

                # Track loader usage
                loader_name = (
                    raw_chunks[0]["metadata"].get("loader", "unknown")
                    if raw_chunks
                    else "unknown"
                )
                self.stats["loaders_used"][loader_name] = (
                    self.stats["loaders_used"].get(loader_name, 0) + 1
                )

            # ═══════════════════════════════════════════════════════════════════
            # OPTION 2: Custom Processors (fallback)
            # ═══════════════════════════════════════════════════════════════════
            else:
                from .document_processors import ProcessingConfig, ProcessorFactory

                config = ProcessingConfig(
                    chunk_size=self.chunk_size,
                    chunk_overlap=self.chunk_overlap,
                    generate_embeddings=self.embedding_fn is not None,
                    embedding_fn=self.embedding_fn,
                )

                processor = ProcessorFactory.create(source=source_path, config=config)
                result = await processor.process(source_path, **kwargs)

                if not result.success:
                    self.stats["errors"].append(
                        {
                            "source": source_path,
                            "error": result.error,
                            "time": time.time(),
                        }
                    )
                    return []

                # Convert to DocumentChunk format
                chunks = []
                for proc_chunk in result.chunks:
                    chunk_metadata = {**(metadata or {}), **proc_chunk.metadata}
                    chunk = DocumentChunk(
                        id=proc_chunk.id,
                        content=proc_chunk.content,
                        source=proc_chunk.source,
                        metadata=chunk_metadata,
                        embedding=proc_chunk.embedding,
                    )
                    chunks.append(chunk)

                # Track processor usage
                self.stats["loaders_used"][processor.processor_name] = (
                    self.stats["loaders_used"].get(processor.processor_name, 0) + 1
                )

            # Store in vector store if available
            if self.vector_store and chunks:
                await self._store_chunks(chunks)

            # Update stats
            self.stats["documents_ingested"] += 1
            self.stats["chunks_created"] += len(chunks)
            self.stats["last_ingestion"] = time.time()

            return chunks

        except Exception as e:
            self.stats["errors"].append(
                {
                    "source": source_path,
                    "error": str(e),
                    "time": time.time(),
                }
            )
            raise

    async def ingest_batch(
        self,
        sources: List[str],
        metadata: Optional[Dict[str, Any]] = None,
        parallel: bool = True,
        max_concurrent: int = 5,
    ) -> Dict[str, List[DocumentChunk]]:
        """
        Ingest multiple documents in batch.

        Args:
            sources: List of source paths/URLs
            metadata: Shared metadata for all documents
            parallel: Process documents in parallel (faster)
            max_concurrent: Maximum concurrent processing tasks

        Returns:
            Dict mapping source -> list of chunks
        """
        import asyncio

        results = {}

        if parallel and len(sources) > 1:
            # Process in parallel with semaphore for rate limiting
            semaphore = asyncio.Semaphore(max_concurrent)

            async def process_with_limit(source: str):
                async with semaphore:
                    return source, await self.ingest_document(source, metadata)

            tasks = [process_with_limit(source) for source in sources]
            completed = await asyncio.gather(*tasks, return_exceptions=True)

            for item in completed:
                if isinstance(item, Exception):
                    continue
                source, chunks = item
                results[source] = chunks
        else:
            # Process sequentially
            for source in sources:
                try:
                    results[source] = await self.ingest_document(source, metadata)
                except Exception:
                    results[source] = []

        return results

    def get_ingestion_stats(self) -> Dict[str, Any]:
        """Get comprehensive ingestion statistics."""
        return {
            **self.stats,
            "error_count": len(self.stats.get("errors", [])),
            "recent_errors": self.stats.get("errors", [])[-5:],
        }

    def get_available_processors(self) -> Dict[str, Any]:
        """Get information about available document processors."""
        from .document_processors import ProcessorFactory

        return ProcessorFactory.get_available_processors()

    def _generate_chunk_id(self, source: str, index: int) -> str:
        """Generate unique ID for a chunk."""
        content = f"{source}:{index}:{time.time()}"
        return hashlib.md5(content.encode()).hexdigest()[:12]

    async def _store_chunks(self, chunks: List[DocumentChunk]) -> None:
        """
        Store chunks in vector store.

        Integrates with PRECEPT's vector store (ChromaDB).
        """
        if not self.vector_store:
            return

        try:
            # Prepare texts and metadata for vector store
            texts = [c.content for c in chunks]
            metadatas = [c.metadata for c in chunks]
            ids = [c.id for c in chunks]

            # Add to vector store (LangChain Chroma interface)
            if hasattr(self.vector_store, "add_texts"):
                self.vector_store.add_texts(
                    texts=texts,
                    metadatas=metadatas,
                    ids=ids,
                )
        except Exception as e:
            self.stats["errors"].append(
                {
                    "source": "vector_store",
                    "error": f"Failed to store chunks: {e}",
                    "time": time.time(),
                }
            )


# =============================================================================
# SOFT INGESTION (Experience/Wisdom Stream)
# =============================================================================


@dataclass
class SoftPatch:
    """
    A "soft patch" to the knowledge base.

    This is the key innovation of PRECEPT - allowing the agent to instantly
    correct/augment the static knowledge without re-indexing.

    Example:
    - Vector DB says: "Hamburg port is operational"
    - Agent discovers: "Hamburg has hidden strike delay"
    - Soft Patch: "Warning: Hamburg shows 'operational' but has strike delays"
    """

    id: str

    # What this patch corrects/augments
    target_document_id: Optional[str] = None  # ID of doc being patched (if any)
    target_query_pattern: Optional[str] = None  # Query patterns this applies to

    # The correction/augmentation
    patch_type: str = "correction"  # correction, augmentation, warning, override
    patch_content: str = ""

    # Context about when/why this patch was created
    source_task: str = ""
    source_observation: str = ""
    confidence: float = 0.8

    # Scope and validity
    valid_until: Optional[float] = None  # Expiration time (for temporary patches)
    domain: str = "general"
    priority: IngestionPriority = IngestionPriority.NORMAL

    # Usage tracking
    times_applied: int = 0
    times_helpful: int = 0
    times_ignored: int = 0

    # Timestamps
    created_at: float = field(default_factory=time.time)
    last_used: Optional[float] = None

    @property
    def usefulness_score(self) -> float:
        """Calculate how useful this patch has been."""
        total = self.times_helpful + self.times_ignored
        if total == 0:
            return 0.5  # Neutral
        return self.times_helpful / total

    def is_expired(self) -> bool:
        """Check if this patch has expired."""
        if self.valid_until is None:
            return False
        return time.time() > self.valid_until

    def to_context_string(self) -> str:
        """Format patch as context string for retrieval augmentation."""
        prefix_map = {
            "correction": "⚠️ CORRECTION",
            "augmentation": "📝 NOTE",
            "warning": "⚡ WARNING",
            "override": "🔴 OVERRIDE",
        }
        prefix = prefix_map.get(self.patch_type, "📌 PATCH")
        return f"[{prefix}]: {self.patch_content}"


class SoftIngestionResult(BaseModel):
    """Result of a soft ingestion operation."""

    patch_id: str = Field(description="ID of created patch")
    patch_type: str = Field(description="Type of patch created")
    success: bool = Field(description="Whether ingestion succeeded")
    reason: str = Field(description="Reason for success/failure")


class SoftIngestionManager:
    """
    Manager for Soft Ingestion (Experience/Wisdom Stream).

    This runs DURING the agent's reasoning loop (ReMem Refine step).
    It allows the agent to instantly "patch" the knowledge base without
    modifying the actual Vector DB.

    Key capability: One-shot learning / Instant adaptation
    """

    def __init__(
        self,
        max_patches: int = 500,
        auto_expire_days: float = 30.0,
    ):
        self.patches: Dict[str, SoftPatch] = {}
        self.max_patches = max_patches
        self.auto_expire_days = auto_expire_days

        # Indices for efficient retrieval
        self.domain_index: Dict[str, Set[str]] = {}  # domain -> patch_ids
        self.document_index: Dict[str, Set[str]] = {}  # doc_id -> patch_ids

        # Statistics
        self.stats = {
            "total_patches_created": 0,
            "total_patches_applied": 0,
            "patches_promoted_to_rules": 0,
            "patches_expired": 0,
        }

    def ingest_correction(
        self,
        target_document_id: str,
        correction: str,
        source_task: str,
        source_observation: str,
        confidence: float = 0.8,
        domain: str = "general",
    ) -> SoftIngestionResult:
        """
        Ingest a correction to an existing document.

        Example: "Document says Hamburg is open, but there's a strike."
        """
        patch = SoftPatch(
            id=self._generate_patch_id(),
            target_document_id=target_document_id,
            patch_type="correction",
            patch_content=correction,
            source_task=source_task,
            source_observation=source_observation,
            confidence=confidence,
            domain=domain,
            valid_until=time.time() + (self.auto_expire_days * 86400),
        )

        return self._store_patch(patch)

    def ingest_warning(
        self,
        query_pattern: str,
        warning: str,
        source_task: str,
        priority: IngestionPriority = IngestionPriority.HIGH,
        domain: str = "general",
    ) -> SoftIngestionResult:
        """
        Ingest a warning for certain query patterns.

        Example: "For all 'Speed' shipments, avoid Hamburg."
        """
        patch = SoftPatch(
            id=self._generate_patch_id(),
            target_query_pattern=query_pattern,
            patch_type="warning",
            patch_content=warning,
            source_task=source_task,
            source_observation="",
            priority=priority,
            domain=domain,
        )

        return self._store_patch(patch)

    def ingest_augmentation(
        self,
        context: str,
        augmentation: str,
        source_task: str,
        domain: str = "general",
    ) -> SoftIngestionResult:
        """
        Ingest additional context/knowledge.

        Example: "Route via Rotterdam is 2h faster during peak season."
        """
        patch = SoftPatch(
            id=self._generate_patch_id(),
            patch_type="augmentation",
            patch_content=augmentation,
            source_task=source_task,
            source_observation=context,
            domain=domain,
        )

        return self._store_patch(patch)

    def ingest_override(
        self,
        target_document_id: str,
        override: str,
        source_task: str,
        valid_until: Optional[float] = None,
        domain: str = "general",
    ) -> SoftIngestionResult:
        """
        Ingest a hard override for a document.

        Example: "IGNORE Hamburg entirely until strike ends."
        """
        patch = SoftPatch(
            id=self._generate_patch_id(),
            target_document_id=target_document_id,
            patch_type="override",
            patch_content=override,
            source_task=source_task,
            source_observation="",
            priority=IngestionPriority.CRITICAL,
            domain=domain,
            valid_until=valid_until,
        )

        return self._store_patch(patch)

    def get_patches_for_retrieval(
        self,
        query: str,
        document_ids: Optional[List[str]] = None,
        domain: Optional[str] = None,
    ) -> List[SoftPatch]:
        """
        Get relevant patches to augment retrieval results.

        This is called during the retrieval phase to "patch" the
        raw Vector DB results.
        """
        relevant_patches = []

        for patch in self.patches.values():
            # Skip expired patches
            if patch.is_expired():
                continue

            # Check domain match
            if domain and patch.domain != domain and patch.domain != "general":
                continue

            # Check if patch applies to any of the retrieved documents
            if document_ids and patch.target_document_id:
                if patch.target_document_id in document_ids:
                    relevant_patches.append(patch)
                    continue

            # Check query pattern match
            if patch.target_query_pattern:
                if patch.target_query_pattern.lower() in query.lower():
                    relevant_patches.append(patch)
                    continue

            # Include high-priority patches for the domain
            if patch.priority in [IngestionPriority.CRITICAL, IngestionPriority.HIGH]:
                if patch.domain == domain or patch.domain == "general":
                    relevant_patches.append(patch)

        # Sort by priority and recency
        relevant_patches.sort(
            key=lambda p: (p.priority.value, -p.created_at),
            reverse=True,
        )

        # Update usage stats
        for patch in relevant_patches:
            patch.times_applied += 1
            patch.last_used = time.time()

        self.stats["total_patches_applied"] += len(relevant_patches)

        return relevant_patches

    def record_patch_feedback(self, patch_id: str, was_helpful: bool) -> None:
        """Record whether a patch was helpful."""
        if patch_id in self.patches:
            patch = self.patches[patch_id]
            if was_helpful:
                patch.times_helpful += 1
            else:
                patch.times_ignored += 1

    def get_consolidation_candidates(
        self, min_uses: int = 5, min_usefulness: float = 0.7
    ) -> List[SoftPatch]:
        """
        Get patches that should be considered for consolidation into prompts.

        These are patches that have been used frequently and proven useful.
        """
        candidates = []

        for patch in self.patches.values():
            if patch.times_applied >= min_uses:
                if patch.usefulness_score >= min_usefulness:
                    candidates.append(patch)

        # Sort by usefulness
        candidates.sort(key=lambda p: p.usefulness_score, reverse=True)

        return candidates

    def remove_patch(self, patch_id: str) -> bool:
        """Remove a patch (e.g., after consolidation)."""
        if patch_id in self.patches:
            patch = self.patches[patch_id]

            # Remove from indices
            if patch.domain in self.domain_index:
                self.domain_index[patch.domain].discard(patch_id)
            if (
                patch.target_document_id
                and patch.target_document_id in self.document_index
            ):
                self.document_index[patch.target_document_id].discard(patch_id)

            del self.patches[patch_id]
            return True
        return False

    def cleanup_expired(self) -> int:
        """Remove expired patches."""
        expired_ids = [pid for pid, patch in self.patches.items() if patch.is_expired()]

        for pid in expired_ids:
            self.remove_patch(pid)

        self.stats["patches_expired"] += len(expired_ids)
        return len(expired_ids)

    def _store_patch(self, patch: SoftPatch) -> SoftIngestionResult:
        """Store a patch and update indices."""
        # Check capacity
        if len(self.patches) >= self.max_patches:
            self._prune_least_useful()

        # Store patch
        self.patches[patch.id] = patch

        # Update domain index
        if patch.domain not in self.domain_index:
            self.domain_index[patch.domain] = set()
        self.domain_index[patch.domain].add(patch.id)

        # Update document index
        if patch.target_document_id:
            if patch.target_document_id not in self.document_index:
                self.document_index[patch.target_document_id] = set()
            self.document_index[patch.target_document_id].add(patch.id)

        self.stats["total_patches_created"] += 1

        return SoftIngestionResult(
            patch_id=patch.id,
            patch_type=patch.patch_type,
            success=True,
            reason="Patch created successfully",
        )

    def _prune_least_useful(self) -> None:
        """Prune least useful patches when at capacity."""
        # Sort by usefulness (lowest first)
        sorted_patches = sorted(
            self.patches.values(),
            key=lambda p: (p.usefulness_score, p.times_applied),
        )

        # Remove bottom 10%
        num_to_remove = max(1, len(sorted_patches) // 10)
        for patch in sorted_patches[:num_to_remove]:
            self.remove_patch(patch.id)

    def _generate_patch_id(self) -> str:
        """Generate unique patch ID."""
        content = f"patch:{time.time()}:{len(self.patches)}"
        return hashlib.md5(content.encode()).hexdigest()[:12]

    def get_stats(self) -> Dict[str, Any]:
        """Get ingestion statistics."""
        return {
            **self.stats,
            "active_patches": len(self.patches),
            "patches_by_type": {
                ptype: sum(1 for p in self.patches.values() if p.patch_type == ptype)
                for ptype in ["correction", "warning", "augmentation", "override"]
            },
            "domains_covered": list(self.domain_index.keys()),
        }


# =============================================================================
# FEEDBACK INGESTION (Training Stream)
# =============================================================================


@dataclass
class ExecutionTrace:
    """
    An execution trace for feedback ingestion.

    This captures what happened during a task execution,
    used by COMPASS to learn and evolve prompts.
    """

    id: str

    # Task info
    task: str
    goal: str
    domain: str

    # Execution details
    steps: List[Dict[str, Any]]  # Think-Act-Observe sequence
    total_steps: int

    # Outcome
    success: bool
    final_answer: str
    confidence: float

    # Retrieval details
    documents_retrieved: List[str]
    patches_applied: List[str]

    # Performance metrics
    execution_time_ms: float
    llm_calls: int
    tokens_used: int

    # What went well / wrong
    success_factors: List[str] = field(default_factory=list)
    failure_factors: List[str] = field(default_factory=list)

    # Timestamps
    started_at: float = field(default_factory=time.time)
    completed_at: Optional[float] = None


class FeedbackIngestionManager:
    """
    Manager for Feedback Ingestion (Training Stream).

    This runs during the COMPASS Phase (batch optimization).
    It ingests execution traces to teach the system how to search better.
    """

    def __init__(
        self,
        max_traces: int = 1000,
        retention_days: float = 30.0,
    ):
        self.traces: List[ExecutionTrace] = []
        self.max_traces = max_traces
        self.retention_days = retention_days

        # Aggregated insights
        self.insights = {
            "common_failures": {},
            "successful_patterns": {},
            "retrieval_effectiveness": {},
        }

        # Statistics
        self.stats = {
            "total_traces_ingested": 0,
            "total_traces_analyzed": 0,
            "insights_generated": 0,
        }

    def ingest_trace(self, trace: ExecutionTrace) -> None:
        """Ingest a new execution trace."""
        self.traces.append(trace)
        self.stats["total_traces_ingested"] += 1

        # Maintain capacity
        if len(self.traces) > self.max_traces:
            self._prune_old_traces()

    def get_traces_for_analysis(
        self,
        domain: Optional[str] = None,
        success_only: bool = False,
        failure_only: bool = False,
        min_confidence: float = 0.0,
        since: Optional[float] = None,
    ) -> List[ExecutionTrace]:
        """
        Get traces for COMPASS analysis.

        Called during the optimization phase.
        """
        filtered = self.traces

        if domain:
            filtered = [t for t in filtered if t.domain == domain]

        if success_only:
            filtered = [t for t in filtered if t.success]
        elif failure_only:
            filtered = [t for t in filtered if not t.success]

        if min_confidence > 0:
            filtered = [t for t in filtered if t.confidence >= min_confidence]

        if since:
            filtered = [t for t in filtered if t.started_at >= since]

        return filtered

    def analyze_patterns(self) -> Dict[str, Any]:
        """
        Analyze traces to find patterns for COMPASS.

        Returns insights about:
        - Common failure patterns
        - Successful strategies
        - Retrieval effectiveness
        """
        analysis = {
            "total_traces": len(self.traces),
            "success_rate": 0.0,
            "avg_steps": 0.0,
            "common_failures": [],
            "successful_patterns": [],
            "patch_effectiveness": {},
        }

        if not self.traces:
            return analysis

        # Basic stats
        successes = [t for t in self.traces if t.success]
        analysis["success_rate"] = len(successes) / len(self.traces)
        analysis["avg_steps"] = sum(t.total_steps for t in self.traces) / len(
            self.traces
        )

        # Failure analysis
        failure_factors = {}
        for trace in self.traces:
            if not trace.success:
                for factor in trace.failure_factors:
                    failure_factors[factor] = failure_factors.get(factor, 0) + 1

        analysis["common_failures"] = sorted(
            failure_factors.items(),
            key=lambda x: x[1],
            reverse=True,
        )[:10]

        # Success pattern analysis
        success_factors = {}
        for trace in successes:
            for factor in trace.success_factors:
                success_factors[factor] = success_factors.get(factor, 0) + 1

        analysis["successful_patterns"] = sorted(
            success_factors.items(),
            key=lambda x: x[1],
            reverse=True,
        )[:10]

        # Patch effectiveness
        patch_usage = {}
        for trace in self.traces:
            for patch_id in trace.patches_applied:
                if patch_id not in patch_usage:
                    patch_usage[patch_id] = {"applied": 0, "successful": 0}
                patch_usage[patch_id]["applied"] += 1
                if trace.success:
                    patch_usage[patch_id]["successful"] += 1

        analysis["patch_effectiveness"] = {
            pid: stats["successful"] / stats["applied"]
            for pid, stats in patch_usage.items()
            if stats["applied"] >= 3
        }

        self.stats["total_traces_analyzed"] += len(self.traces)

        return analysis

    def get_consolidation_recommendations(
        self, min_occurrences: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Get recommendations for what should be consolidated into prompts.

        Called by COMPASS during the optimization phase.
        """
        recommendations = []

        # Analyze failure patterns
        failure_counts = {}
        for trace in self.traces:
            if not trace.success:
                for factor in trace.failure_factors:
                    failure_counts[factor] = failure_counts.get(factor, 0) + 1

        for factor, count in failure_counts.items():
            if count >= min_occurrences:
                recommendations.append(
                    {
                        "type": "add_warning",
                        "content": f"Avoid: {factor}",
                        "occurrences": count,
                        "source": "failure_analysis",
                    }
                )

        # Analyze success patterns
        success_counts = {}
        for trace in self.traces:
            if trace.success:
                for factor in trace.success_factors:
                    success_counts[factor] = success_counts.get(factor, 0) + 1

        for factor, count in success_counts.items():
            if count >= min_occurrences:
                recommendations.append(
                    {
                        "type": "add_instruction",
                        "content": f"Strategy: {factor}",
                        "occurrences": count,
                        "source": "success_analysis",
                    }
                )

        self.stats["insights_generated"] += len(recommendations)

        return recommendations

    def clear_analyzed_traces(self, trace_ids: Optional[List[str]] = None) -> int:
        """Clear traces after they've been analyzed."""
        if trace_ids:
            original_count = len(self.traces)
            self.traces = [t for t in self.traces if t.id not in trace_ids]
            return original_count - len(self.traces)
        else:
            count = len(self.traces)
            self.traces = []
            return count

    def _prune_old_traces(self) -> None:
        """Remove old traces."""
        cutoff = time.time() - (self.retention_days * 86400)
        self.traces = [t for t in self.traces if t.started_at >= cutoff]

        # If still over capacity, remove oldest
        if len(self.traces) > self.max_traces:
            self.traces = sorted(self.traces, key=lambda t: t.started_at, reverse=True)[
                : self.max_traces
            ]

    def get_stats(self) -> Dict[str, Any]:
        """Get ingestion statistics."""
        return {
            **self.stats,
            "active_traces": len(self.traces),
            "oldest_trace": min((t.started_at for t in self.traces), default=None),
            "newest_trace": max((t.started_at for t in self.traces), default=None),
        }


# =============================================================================
# UNIFIED INGESTION COORDINATOR
# =============================================================================


class PRECEPTIngestionCoordinator:
    """
    Coordinates all three ingestion streams in PRECEPT.

    This is the main interface for the ingestion architecture.
    """

    def __init__(
        self,
        hard_pipeline: Optional[HardIngestionPipeline] = None,
        soft_manager: Optional[SoftIngestionManager] = None,
        feedback_manager: Optional[FeedbackIngestionManager] = None,
    ):
        self.hard = hard_pipeline or DefaultHardIngestionPipeline()
        self.soft = soft_manager or SoftIngestionManager()
        self.feedback = feedback_manager or FeedbackIngestionManager()

    # Hard Ingestion (Document Stream)
    async def ingest_document(
        self, source: str, metadata: Optional[Dict] = None
    ) -> List[DocumentChunk]:
        """Ingest a document (Hard Ingestion)."""
        return await self.hard.ingest_document(source, metadata)

    # Soft Ingestion (Experience Stream)
    def create_correction(
        self,
        document_id: str,
        correction: str,
        task: str,
        observation: str,
        **kwargs,
    ) -> SoftIngestionResult:
        """Create a correction patch (Soft Ingestion)."""
        return self.soft.ingest_correction(
            target_document_id=document_id,
            correction=correction,
            source_task=task,
            source_observation=observation,
            **kwargs,
        )

    def create_warning(
        self, query_pattern: str, warning: str, task: str, **kwargs
    ) -> SoftIngestionResult:
        """Create a warning patch (Soft Ingestion)."""
        return self.soft.ingest_warning(
            query_pattern=query_pattern,
            warning=warning,
            source_task=task,
            **kwargs,
        )

    def get_retrieval_patches(
        self, query: str, document_ids: List[str], domain: str
    ) -> List[SoftPatch]:
        """Get patches to augment retrieval (Soft Ingestion)."""
        return self.soft.get_patches_for_retrieval(query, document_ids, domain)

    # Feedback Ingestion (Training Stream)
    def record_execution(self, trace: ExecutionTrace) -> None:
        """Record an execution trace (Feedback Ingestion)."""
        self.feedback.ingest_trace(trace)

    def analyze_for_optimization(self) -> Dict[str, Any]:
        """Analyze traces for COMPASS optimization."""
        return self.feedback.analyze_patterns()

    def get_consolidation_recommendations(self) -> List[Dict[str, Any]]:
        """Get recommendations for prompt consolidation."""
        return self.feedback.get_consolidation_recommendations()

    # Unified stats
    def get_all_stats(self) -> Dict[str, Any]:
        """Get statistics from all ingestion streams."""
        return {
            "hard_ingestion": self.hard.get_ingestion_stats(),
            "soft_ingestion": self.soft.get_stats(),
            "feedback_ingestion": self.feedback.get_stats(),
        }
