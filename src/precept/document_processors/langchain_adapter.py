"""
LangChain Adapter for PRECEPT Document Processors.

This module wraps LangChain's excellent document loaders and text splitters,
providing a unified interface for PRECEPT while leveraging the battle-tested
LangChain ecosystem.

WHY LANGCHAIN?
═══════════════════════════════════════════════════════════════════════════════════
LangChain provides:
- 80+ document loaders (PDF, Web, DOCX, CSV, JSON, Notion, Confluence, etc.)
- Battle-tested text splitters (recursive, semantic, token-based)
- Direct integration with vector stores (Chroma, Pinecone, Weaviate)
- Active community and maintenance

This adapter:
- Wraps LangChain loaders in PRECEPT's interface
- Provides fallback to custom implementation if LangChain unavailable
- Adds PRECEPT-specific metadata and processing

USAGE:
═══════════════════════════════════════════════════════════════════════════════════

    from precept.document_processors import LangChainAdapter

    # Use LangChain's loaders through PRECEPT's interface
    adapter = LangChainAdapter()

    # Process PDF
    chunks = await adapter.load_and_split("report.pdf")

    # Process web page
    chunks = await adapter.load_and_split("https://example.com/article")

    # Process with custom splitter config
    chunks = await adapter.load_and_split(
        "doc.txt",
        chunk_size=500,
        chunk_overlap=50,
    )
"""

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Type, Union

# Check LangChain availability
try:
    from langchain_core.documents import Document
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    LANGCHAIN_CORE_AVAILABLE = True
except ImportError:
    LANGCHAIN_CORE_AVAILABLE = False
    Document = None
    RecursiveCharacterTextSplitter = None

# LangChain Document Loaders
try:
    from langchain_community.document_loaders import (
        PyPDFLoader,
        TextLoader,
        UnstructuredMarkdownLoader,
        WebBaseLoader,
    )

    LANGCHAIN_LOADERS_AVAILABLE = True
except ImportError:
    LANGCHAIN_LOADERS_AVAILABLE = False
    PyPDFLoader = None
    TextLoader = None
    UnstructuredMarkdownLoader = None
    WebBaseLoader = None

# Optional: Better PDF handling
try:
    from langchain_community.document_loaders import PDFPlumberLoader

    PDFPLUMBER_LOADER_AVAILABLE = True
except ImportError:
    PDFPLUMBER_LOADER_AVAILABLE = False
    PDFPlumberLoader = None

# Optional: Async web loading
try:
    from langchain_community.document_loaders import AsyncHtmlLoader

    ASYNC_HTML_AVAILABLE = True
except ImportError:
    ASYNC_HTML_AVAILABLE = False
    AsyncHtmlLoader = None


@dataclass
class LoaderConfig:
    """Configuration for a LangChain loader."""

    loader_class: Type
    extensions: List[str]
    kwargs: Dict[str, Any] = None

    def __post_init__(self):
        if self.kwargs is None:
            self.kwargs = {}


class LangChainAdapter:
    """
    Adapter that wraps LangChain document loaders for PRECEPT.

    Provides:
    - Auto-detection of document type
    - Unified interface for all loaders
    - Fallback to custom processors if LangChain unavailable
    - PRECEPT-specific metadata enrichment
    """

    # Default loader mappings (extension -> loader config)
    DEFAULT_LOADERS: Dict[str, LoaderConfig] = {}

    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        use_pdfplumber: bool = True,
    ):
        """
        Initialize the LangChain adapter.

        Args:
            chunk_size: Default chunk size for text splitting
            chunk_overlap: Default overlap between chunks
            use_pdfplumber: Use PDFPlumber for better PDF handling
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.use_pdfplumber = use_pdfplumber

        # Statistics
        self.stats = {
            "documents_loaded": 0,
            "chunks_created": 0,
            "loaders_used": {},
        }

        # Initialize loader mappings
        self._init_loaders()

    def _init_loaders(self) -> None:
        """Initialize available LangChain loaders."""
        if not LANGCHAIN_LOADERS_AVAILABLE:
            return

        # PDF Loader
        if self.use_pdfplumber and PDFPLUMBER_LOADER_AVAILABLE:
            self.DEFAULT_LOADERS[".pdf"] = LoaderConfig(
                loader_class=PDFPlumberLoader,
                extensions=[".pdf"],
            )
        elif PyPDFLoader:
            self.DEFAULT_LOADERS[".pdf"] = LoaderConfig(
                loader_class=PyPDFLoader,
                extensions=[".pdf"],
            )

        # Text Loader
        if TextLoader:
            for ext in [".txt", ".text", ".log", ".csv", ".tsv"]:
                self.DEFAULT_LOADERS[ext] = LoaderConfig(
                    loader_class=TextLoader,
                    extensions=[ext],
                    kwargs={"encoding": "utf-8"},
                )

        # Markdown Loader
        if UnstructuredMarkdownLoader:
            for ext in [".md", ".markdown"]:
                self.DEFAULT_LOADERS[ext] = LoaderConfig(
                    loader_class=UnstructuredMarkdownLoader,
                    extensions=[ext],
                )

    def get_loader(self, source: str) -> Optional[Any]:
        """
        Get the appropriate LangChain loader for a source.

        Args:
            source: File path or URL

        Returns:
            Configured loader instance or None
        """
        source_lower = source.lower()

        # Check if it's a URL
        if source_lower.startswith(("http://", "https://")):
            if WebBaseLoader:
                return WebBaseLoader([source])
            return None

        # Check file extension
        for ext, config in self.DEFAULT_LOADERS.items():
            if source_lower.endswith(ext):
                return config.loader_class(source, **config.kwargs)

        return None

    def get_text_splitter(
        self,
        chunk_size: Optional[int] = None,
        chunk_overlap: Optional[int] = None,
    ) -> Any:
        """
        Get a configured text splitter.

        Args:
            chunk_size: Override default chunk size
            chunk_overlap: Override default chunk overlap

        Returns:
            Configured text splitter
        """
        if not LANGCHAIN_CORE_AVAILABLE:
            raise ImportError("LangChain not available. pip install langchain")

        return RecursiveCharacterTextSplitter(
            chunk_size=chunk_size or self.chunk_size,
            chunk_overlap=chunk_overlap or self.chunk_overlap,
            length_function=len,
            is_separator_regex=False,
        )

    async def load_and_split(
        self,
        source: Union[str, Path],
        chunk_size: Optional[int] = None,
        chunk_overlap: Optional[int] = None,
        metadata: Optional[Dict[str, Any]] = None,
        **loader_kwargs: Any,
    ) -> List[Dict[str, Any]]:
        """
        Load a document and split into chunks using LangChain.

        This is the main entry point for document processing.

        Args:
            source: File path or URL
            chunk_size: Override default chunk size
            chunk_overlap: Override default chunk overlap
            metadata: Additional metadata to attach
            **loader_kwargs: Passed to the loader

        Returns:
            List of chunk dictionaries with content and metadata

        Example:
            chunks = await adapter.load_and_split("report.pdf")
            for chunk in chunks:
                print(chunk["content"][:100])
                print(chunk["metadata"])
        """
        source_str = str(source)
        start_time = time.time()

        # Get appropriate loader
        loader = self.get_loader(source_str)

        if loader is None:
            # Fallback: try custom processors
            return await self._fallback_load(
                source_str, chunk_size, chunk_overlap, metadata
            )

        # Load documents
        if hasattr(loader, "aload"):
            # Async loading
            documents = await loader.aload()
        else:
            # Sync loading
            documents = loader.load()

        # Split documents
        splitter = self.get_text_splitter(chunk_size, chunk_overlap)
        split_docs = splitter.split_documents(documents)

        # Convert to PRECEPT chunk format
        chunks = []
        for i, doc in enumerate(split_docs):
            chunk_metadata = {
                "source": source_str,
                "chunk_index": i,
                "total_chunks": len(split_docs),
                "loader": loader.__class__.__name__,
                "processing_time_ms": (time.time() - start_time) * 1000,
                **(doc.metadata or {}),
                **(metadata or {}),
            }

            chunks.append(
                {
                    "id": f"{source_str}:{i}",
                    "content": doc.page_content,
                    "metadata": chunk_metadata,
                }
            )

        # Update stats
        loader_name = loader.__class__.__name__
        self.stats["documents_loaded"] += 1
        self.stats["chunks_created"] += len(chunks)
        self.stats["loaders_used"][loader_name] = (
            self.stats["loaders_used"].get(loader_name, 0) + 1
        )

        return chunks

    async def _fallback_load(
        self,
        source: str,
        chunk_size: Optional[int],
        chunk_overlap: Optional[int],
        metadata: Optional[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """
        Fallback to custom processors when LangChain loader unavailable.

        Args:
            source: File path
            chunk_size: Chunk size
            chunk_overlap: Chunk overlap
            metadata: Additional metadata

        Returns:
            List of chunks
        """
        # Try custom processors
        from .base import ProcessingConfig
        from .factory import ProcessorFactory

        config = ProcessingConfig(
            chunk_size=chunk_size or self.chunk_size,
            chunk_overlap=chunk_overlap or self.chunk_overlap,
        )

        processor = ProcessorFactory.create(source, config=config)
        result = await processor.process(source)

        if not result.success:
            raise ValueError(f"Failed to process {source}: {result.error}")

        # Convert to chunk format
        chunks = []
        for chunk in result.chunks:
            chunks.append(
                {
                    "id": chunk.id,
                    "content": chunk.content,
                    "metadata": {**chunk.metadata, **(metadata or {})},
                }
            )

        return chunks

    def load_and_split_sync(
        self,
        source: Union[str, Path],
        chunk_size: Optional[int] = None,
        chunk_overlap: Optional[int] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Synchronous version of load_and_split.

        Useful for non-async contexts.
        """
        import asyncio

        return asyncio.run(
            self.load_and_split(source, chunk_size, chunk_overlap, metadata)
        )

    @staticmethod
    def is_available() -> bool:
        """Check if LangChain is available."""
        return LANGCHAIN_CORE_AVAILABLE and LANGCHAIN_LOADERS_AVAILABLE

    @staticmethod
    def get_available_loaders() -> Dict[str, bool]:
        """Get availability status of each loader type."""
        return {
            "langchain_core": LANGCHAIN_CORE_AVAILABLE,
            "langchain_loaders": LANGCHAIN_LOADERS_AVAILABLE,
            "pdf_pypdf": LANGCHAIN_LOADERS_AVAILABLE and PyPDFLoader is not None,
            "pdf_pdfplumber": PDFPLUMBER_LOADER_AVAILABLE,
            "text": LANGCHAIN_LOADERS_AVAILABLE and TextLoader is not None,
            "markdown": LANGCHAIN_LOADERS_AVAILABLE
            and UnstructuredMarkdownLoader is not None,
            "web": LANGCHAIN_LOADERS_AVAILABLE and WebBaseLoader is not None,
            "web_async": ASYNC_HTML_AVAILABLE,
        }

    def get_stats(self) -> Dict[str, Any]:
        """Get processing statistics."""
        return {
            **self.stats,
            "langchain_available": self.is_available(),
            "available_loaders": self.get_available_loaders(),
        }


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================


async def load_document(
    source: Union[str, Path],
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
    **kwargs: Any,
) -> List[Dict[str, Any]]:
    """
    Convenience function to load and chunk a document.

    Uses LangChain if available, falls back to custom processors.

    Args:
        source: File path or URL
        chunk_size: Chunk size
        chunk_overlap: Chunk overlap
        **kwargs: Additional loader options

    Returns:
        List of chunks

    Example:
        chunks = await load_document("report.pdf")
        chunks = await load_document("https://example.com/article")
    """
    adapter = LangChainAdapter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return await adapter.load_and_split(source, **kwargs)


def load_document_sync(
    source: Union[str, Path],
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
    **kwargs: Any,
) -> List[Dict[str, Any]]:
    """
    Synchronous version of load_document.

    Example:
        chunks = load_document_sync("report.pdf")
    """
    adapter = LangChainAdapter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return adapter.load_and_split_sync(source, **kwargs)
