"""
Base classes for PRECEPT Document Processors.

This module defines the core abstractions for the document processing system:
- DocumentProcessor: Strategy interface for all processors
- ChunkingStrategy: Interface for text chunking algorithms
- DocumentChunk: Standard output format
- ProcessingConfig: Configuration for processing

SOLID PRINCIPLES:
═══════════════════════════════════════════════════════════════════════════════════
- Single Responsibility: Each class has one purpose
- Open/Closed: Open for extension, closed for modification
- Liskov Substitution: All processors are interchangeable
- Interface Segregation: Minimal, focused interfaces
- Dependency Inversion: Depend on abstractions, not concretions
"""

import hashlib
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Protocol,
    Union,
)

# =============================================================================
# ENUMS
# =============================================================================


class SourceType(Enum):
    """Types of document sources."""

    FILE = "file"  # Local file (PDF, TXT, MD, etc.)
    URL = "url"  # Web URL for scraping
    API = "api"  # API endpoint
    RAW = "raw"  # Raw text content


class ChunkingMethod(Enum):
    """Available chunking methods."""

    FIXED_SIZE = "fixed_size"  # Fixed character/token count
    SENTENCE = "sentence"  # Sentence boundaries
    PARAGRAPH = "paragraph"  # Paragraph boundaries
    RECURSIVE = "recursive"  # Recursive splitting (LangChain style)
    SEMANTIC = "semantic"  # Semantic similarity based


# =============================================================================
# CONFIGURATION
# =============================================================================


@dataclass
class ProcessingConfig:
    """
    Configuration for document processing.

    This allows customization of how documents are processed without
    modifying processor code.
    """

    # Chunking configuration
    chunk_size: int = 1000  # Target chunk size in characters
    chunk_overlap: int = 200  # Overlap between chunks
    chunking_method: ChunkingMethod = ChunkingMethod.RECURSIVE

    # Content cleaning
    remove_extra_whitespace: bool = True
    remove_headers_footers: bool = True  # For PDFs
    preserve_tables: bool = True
    preserve_code_blocks: bool = True

    # Metadata extraction
    extract_title: bool = True
    extract_author: bool = True
    extract_dates: bool = True
    extract_headings: bool = True

    # Processing limits
    max_pages: Optional[int] = None  # Limit pages for PDFs
    max_content_length: Optional[int] = None  # Max characters to process

    # Embedding (optional, can be done separately)
    generate_embeddings: bool = False
    embedding_fn: Optional[Callable[[str], List[float]]] = None


# =============================================================================
# DATA CLASSES
# =============================================================================


@dataclass
class DocumentMetadata:
    """
    Metadata extracted from a document.

    This is separated from content to allow flexible metadata handling.
    """

    # Source information
    source: str  # Original path/URL
    source_type: SourceType
    file_type: Optional[str] = None  # e.g., "pdf", "html", "txt"

    # Document metadata
    title: Optional[str] = None
    author: Optional[str] = None
    created_date: Optional[str] = None
    modified_date: Optional[str] = None

    # Content structure
    page_count: Optional[int] = None
    word_count: Optional[int] = None
    section_count: Optional[int] = None
    headings: List[str] = field(default_factory=list)

    # Processing info
    processed_at: float = field(default_factory=time.time)
    processor_name: str = ""
    processing_time_ms: float = 0.0

    # Custom metadata (extensible)
    custom: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage."""
        return {
            "source": self.source,
            "source_type": self.source_type.value,
            "file_type": self.file_type,
            "title": self.title,
            "author": self.author,
            "created_date": self.created_date,
            "modified_date": self.modified_date,
            "page_count": self.page_count,
            "word_count": self.word_count,
            "section_count": self.section_count,
            "headings": self.headings,
            "processed_at": self.processed_at,
            "processor_name": self.processor_name,
            "processing_time_ms": self.processing_time_ms,
            **self.custom,
        }


@dataclass
class DocumentChunk:
    """
    A chunk of processed document content.

    This is the standard output format for all processors.
    Designed for direct storage in Vector DB.
    """

    id: str  # Unique identifier
    content: str  # The actual text content
    source: str  # Original document source

    # Position within document
    chunk_index: int = 0  # Position in sequence
    total_chunks: int = 1  # Total chunks from this document
    page_number: Optional[int] = None  # For paginated documents
    section: Optional[str] = None  # Section/heading this belongs to

    # Metadata (shared reference to document metadata)
    metadata: Dict[str, Any] = field(default_factory=dict)

    # Embedding (populated if requested)
    embedding: Optional[List[float]] = None

    # Timestamps
    created_at: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage."""
        return {
            "id": self.id,
            "content": self.content,
            "source": self.source,
            "chunk_index": self.chunk_index,
            "total_chunks": self.total_chunks,
            "page_number": self.page_number,
            "section": self.section,
            "metadata": self.metadata,
            "created_at": self.created_at,
        }

    @staticmethod
    def generate_id(source: str, index: int, content: str = "") -> str:
        """Generate a unique, deterministic ID for a chunk."""
        hash_input = f"{source}:{index}:{content[:100]}"
        return hashlib.md5(hash_input.encode()).hexdigest()[:16]


@dataclass
class ProcessingResult:
    """
    Result of processing a document.

    Contains both the chunks and metadata about the processing.
    """

    success: bool
    chunks: List[DocumentChunk]
    metadata: DocumentMetadata

    # Error information (if failed)
    error: Optional[str] = None
    error_details: Optional[str] = None

    # Statistics
    processing_time_ms: float = 0.0
    total_characters: int = 0
    total_chunks: int = 0

    def __post_init__(self):
        """Calculate statistics after initialization."""
        self.total_chunks = len(self.chunks)
        self.total_characters = sum(len(c.content) for c in self.chunks)


# =============================================================================
# ABSTRACT INTERFACES
# =============================================================================


class ChunkingStrategy(Protocol):
    """
    Protocol for chunking strategies.

    Using Protocol for structural subtyping - any class with matching
    methods can be used as a ChunkingStrategy.
    """

    def chunk(self, text: str, config: ProcessingConfig) -> List[str]:
        """
        Split text into chunks.

        Args:
            text: The text to chunk
            config: Processing configuration

        Returns:
            List of text chunks
        """
        ...


class DocumentProcessor(ABC):
    """
    Abstract base class for document processors.

    STRATEGY PATTERN: Each concrete processor is a strategy for handling
    a specific document type.

    TEMPLATE METHOD PATTERN: The `process` method defines the algorithm
    structure, with subclasses implementing specific steps.
    """

    # Class-level configuration
    processor_name: str = "base"
    supported_extensions: List[str] = []
    supported_source_types: List[SourceType] = [SourceType.FILE]

    def __init__(
        self,
        config: Optional[ProcessingConfig] = None,
        chunker: Optional[ChunkingStrategy] = None,
    ):
        """
        Initialize the processor.

        Args:
            config: Processing configuration (uses defaults if not provided)
            chunker: Chunking strategy (uses default based on config if not provided)
        """
        self.config = config or ProcessingConfig()
        self._chunker = chunker

    @property
    def chunker(self) -> ChunkingStrategy:
        """Get the chunking strategy (lazy initialization)."""
        if self._chunker is None:
            # Import here to avoid circular imports
            from .chunkers import get_default_chunker

            self._chunker = get_default_chunker(self.config.chunking_method)
        return self._chunker

    @abstractmethod
    async def process(
        self,
        source: Union[str, Path],
        **kwargs: Any,
    ) -> ProcessingResult:
        """
        Process a document and return chunks.

        This is the main entry point for document processing.

        Args:
            source: Path to file, URL, or raw content
            **kwargs: Additional processor-specific options

        Returns:
            ProcessingResult containing chunks and metadata
        """
        pass

    @abstractmethod
    async def extract_text(self, source: Union[str, Path]) -> str:
        """
        Extract raw text from the source.

        This is the core extraction logic specific to each document type.

        Args:
            source: The document source

        Returns:
            Extracted text content
        """
        pass

    @abstractmethod
    async def extract_metadata(self, source: Union[str, Path]) -> DocumentMetadata:
        """
        Extract metadata from the source.

        Args:
            source: The document source

        Returns:
            Extracted metadata
        """
        pass

    def clean_text(self, text: str) -> str:
        """
        Clean and normalize extracted text.

        Override in subclasses for document-specific cleaning.

        Args:
            text: Raw extracted text

        Returns:
            Cleaned text
        """
        if self.config.remove_extra_whitespace:
            # Normalize whitespace while preserving paragraph breaks
            import re

            text = re.sub(r"[ \t]+", " ", text)
            text = re.sub(r"\n{3,}", "\n\n", text)
            text = text.strip()

        return text

    def create_chunks(
        self,
        text: str,
        source: str,
        metadata: DocumentMetadata,
    ) -> List[DocumentChunk]:
        """
        Create chunks from text using the configured strategy.

        Args:
            text: The text to chunk
            source: Original source for ID generation
            metadata: Document metadata to attach

        Returns:
            List of DocumentChunk objects
        """
        # Use chunker to split text
        raw_chunks = self.chunker.chunk(text, self.config)

        # Create DocumentChunk objects
        chunks = []
        total = len(raw_chunks)

        for i, content in enumerate(raw_chunks):
            chunk = DocumentChunk(
                id=DocumentChunk.generate_id(source, i, content),
                content=content,
                source=source,
                chunk_index=i,
                total_chunks=total,
                metadata=metadata.to_dict(),
            )

            # Generate embedding if configured
            if self.config.generate_embeddings and self.config.embedding_fn:
                chunk.embedding = self.config.embedding_fn(content)

            chunks.append(chunk)

        return chunks

    def can_process(self, source: Union[str, Path]) -> bool:
        """
        Check if this processor can handle the given source.

        Args:
            source: The source to check

        Returns:
            True if this processor can handle the source
        """
        source_str = str(source).lower()

        # Check file extension
        for ext in self.supported_extensions:
            if source_str.endswith(ext.lower()):
                return True

        return False

    @classmethod
    def get_processor_info(cls) -> Dict[str, Any]:
        """Get information about this processor."""
        return {
            "name": cls.processor_name,
            "supported_extensions": cls.supported_extensions,
            "supported_source_types": [st.value for st in cls.supported_source_types],
        }
