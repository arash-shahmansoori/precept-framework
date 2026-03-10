"""
PRECEPT Document Processors: Extensible Hard Ingestion Architecture

This module provides document processing for PRECEPT's Hard Ingestion stream.

═══════════════════════════════════════════════════════════════════════════════════
RECOMMENDED: Use LangChain Adapter (leverages battle-tested ecosystem)
═══════════════════════════════════════════════════════════════════════════════════

LangChain provides 80+ document loaders and is production-ready:

    from precept.document_processors import load_document, LangChainAdapter

    # Simple usage (auto-detects type, uses LangChain under the hood)
    chunks = await load_document("report.pdf")
    chunks = await load_document("https://example.com/article")

    # With configuration
    adapter = LangChainAdapter(chunk_size=500, chunk_overlap=50)
    chunks = await adapter.load_and_split("data.txt")

═══════════════════════════════════════════════════════════════════════════════════
ALTERNATIVE: Custom Processors (for unsupported formats or special needs)
═══════════════════════════════════════════════════════════════════════════════════

Use custom processors when LangChain doesn't have what you need:

    from precept.document_processors import ProcessorFactory, ProcessorRegistry

    # Register a custom processor
    @ProcessorRegistry.register("custom", extensions=[".xyz"])
    class MyProcessor(DocumentProcessor):
        async def process(self, source, **kwargs):
            ...

    # Use it
    processor = ProcessorFactory.create("data.xyz")
    chunks = await processor.process("data.xyz")

ARCHITECTURE:
═══════════════════════════════════════════════════════════════════════════════════

    ┌─────────────────────────────────────────────────────────────────────────────┐
    │                         HARD INGESTION PIPELINE                             │
    ├─────────────────────────────────────────────────────────────────────────────┤
    │                                                                             │
    │   OPTION 1: LangChain Adapter (RECOMMENDED)                                 │
    │   ──────────────────────────────────────────                                │
    │   Source ──► LangChainAdapter ──► LangChain Loaders ──► Chunks              │
    │              (80+ formats)        (PyPDF, Web, etc.)                        │
    │                                                                             │
    │   OPTION 2: Custom Processors (for special cases)                           │
    │   ─────────────────────────────────────────────────                         │
    │   Source ──► ProcessorFactory ──► DocumentProcessor ──► Chunks              │
    │              (plugin registry)    (your custom code)                        │
    │                                                                             │
    └─────────────────────────────────────────────────────────────────────────────┘
"""

# LangChain Adapter (RECOMMENDED - uses battle-tested ecosystem)
# Base classes (for custom processors)
from .base import (
    ChunkingStrategy,
    DocumentChunk,
    DocumentMetadata,
    DocumentProcessor,
    ProcessingConfig,
    ProcessingResult,
)
from .chunkers import (
    FixedSizeChunker,
    RecursiveChunker,
    SemanticChunker,
    SentenceChunker,
)
from .factory import ProcessorFactory
from .langchain_adapter import (
    LangChainAdapter,
    load_document,
    load_document_sync,
)
from .processors import (
    MarkdownProcessor,
    PDFProcessor,
    TextProcessor,
    WebScrapingProcessor,
)
from .registry import ProcessorRegistry

__all__ = [
    # LangChain Adapter (RECOMMENDED)
    "LangChainAdapter",
    "load_document",
    "load_document_sync",
    # Base classes
    "DocumentProcessor",
    "DocumentChunk",
    "DocumentMetadata",
    "ProcessingConfig",
    "ProcessingResult",
    "ChunkingStrategy",
    # Chunking strategies
    "FixedSizeChunker",
    "SentenceChunker",
    "RecursiveChunker",
    "SemanticChunker",
    # Concrete processors (fallback when LangChain unavailable)
    "PDFProcessor",
    "TextProcessor",
    "WebScrapingProcessor",
    "MarkdownProcessor",
    # Factory and registry (for custom processors)
    "ProcessorFactory",
    "ProcessorRegistry",
]
