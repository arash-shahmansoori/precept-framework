"""
Chunking Strategies for PRECEPT Document Processors.

This module provides various text chunking strategies following the
Strategy Pattern. Each chunker can be used interchangeably.

AVAILABLE STRATEGIES:
═══════════════════════════════════════════════════════════════════════════════════
1. FixedSizeChunker - Split by character/token count
2. SentenceChunker - Split at sentence boundaries
3. RecursiveChunker - Hierarchical splitting (paragraph → sentence → word)
4. SemanticChunker - Split based on semantic similarity (requires embeddings)
"""

import re
from abc import ABC, abstractmethod
from typing import List, Optional

from .base import ChunkingMethod, ProcessingConfig


class BaseChunker(ABC):
    """Base class for all chunking strategies."""

    @abstractmethod
    def chunk(self, text: str, config: ProcessingConfig) -> List[str]:
        """
        Split text into chunks.

        Args:
            text: The text to chunk
            config: Processing configuration

        Returns:
            List of text chunks
        """
        pass

    def _add_overlap(
        self,
        chunks: List[str],
        overlap: int,
    ) -> List[str]:
        """
        Add overlap between chunks by prepending previous chunk's ending.

        Args:
            chunks: List of chunks without overlap
            overlap: Number of characters to overlap

        Returns:
            List of chunks with overlap
        """
        if overlap <= 0 or len(chunks) <= 1:
            return chunks

        overlapped = [chunks[0]]

        for i in range(1, len(chunks)):
            prev_chunk = chunks[i - 1]
            current_chunk = chunks[i]

            # Get the end of the previous chunk for overlap
            overlap_text = prev_chunk[-overlap:] if len(prev_chunk) >= overlap else prev_chunk

            # Add overlap to current chunk
            overlapped.append(overlap_text + current_chunk)

        return overlapped


class FixedSizeChunker(BaseChunker):
    """
    Split text into fixed-size chunks.

    Simple but effective for uniform processing.
    Good for: Large documents, consistent vector sizes.
    """

    def chunk(self, text: str, config: ProcessingConfig) -> List[str]:
        """Split text into fixed-size chunks with overlap."""
        chunk_size = config.chunk_size
        overlap = config.chunk_overlap

        if not text:
            return []

        # Simple character-based splitting
        chunks = []
        start = 0

        while start < len(text):
            end = start + chunk_size
            chunk = text[start:end]

            # Try to break at word boundary if not at the end
            if end < len(text):
                # Find last space within chunk
                last_space = chunk.rfind(" ")
                if last_space > chunk_size * 0.5:  # At least half the chunk
                    chunk = chunk[:last_space]
                    end = start + last_space

            chunks.append(chunk.strip())

            # Move start, accounting for overlap
            start = end - overlap if overlap > 0 else end

        return [c for c in chunks if c]  # Remove empty chunks


class SentenceChunker(BaseChunker):
    """
    Split text at sentence boundaries.

    Preserves semantic meaning by keeping sentences intact.
    Good for: QA systems, semantic search.
    """

    # Sentence ending patterns
    SENTENCE_ENDINGS = re.compile(r"(?<=[.!?])\s+(?=[A-Z])")

    # Abbreviations to avoid splitting on
    ABBREVIATIONS = {
        "mr.",
        "mrs.",
        "ms.",
        "dr.",
        "prof.",
        "sr.",
        "jr.",
        "vs.",
        "etc.",
        "i.e.",
        "e.g.",
        "inc.",
        "ltd.",
        "co.",
        "corp.",
        "st.",
        "ave.",
        "blvd.",
    }

    def chunk(self, text: str, config: ProcessingConfig) -> List[str]:
        """Split text into sentence-based chunks."""
        chunk_size = config.chunk_size

        if not text:
            return []

        # Split into sentences
        sentences = self._split_sentences(text)

        # Group sentences into chunks
        chunks = []
        current_chunk = []
        current_length = 0

        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue

            sentence_length = len(sentence)

            # If single sentence exceeds chunk size, add it as its own chunk
            if sentence_length > chunk_size:
                if current_chunk:
                    chunks.append(" ".join(current_chunk))
                    current_chunk = []
                    current_length = 0
                chunks.append(sentence)
                continue

            # If adding this sentence would exceed chunk size, start new chunk
            if current_length + sentence_length + 1 > chunk_size:
                if current_chunk:
                    chunks.append(" ".join(current_chunk))
                current_chunk = [sentence]
                current_length = sentence_length
            else:
                current_chunk.append(sentence)
                current_length += sentence_length + 1  # +1 for space

        # Add remaining sentences
        if current_chunk:
            chunks.append(" ".join(current_chunk))

        return chunks

    def _split_sentences(self, text: str) -> List[str]:
        """Split text into sentences, handling abbreviations."""
        # Simple approach: split on sentence-ending punctuation followed by space and capital
        sentences = self.SENTENCE_ENDINGS.split(text)

        # Refine to handle edge cases
        refined = []
        for sentence in sentences:
            # Check if it's just an abbreviation
            if sentence.lower().strip() in self.ABBREVIATIONS:
                if refined:
                    refined[-1] += " " + sentence
                else:
                    refined.append(sentence)
            else:
                refined.append(sentence)

        return refined


class RecursiveChunker(BaseChunker):
    """
    Recursively split text using a hierarchy of separators.

    Tries to preserve structure by splitting on larger boundaries first,
    then falling back to smaller ones.

    Good for: Structured documents, code, mixed content.
    """

    # Default separator hierarchy (from coarsest to finest)
    DEFAULT_SEPARATORS = [
        "\n\n\n",  # Multiple blank lines (sections)
        "\n\n",  # Paragraph breaks
        "\n",  # Line breaks
        ". ",  # Sentences
        ", ",  # Clauses
        " ",  # Words
        "",  # Characters (last resort)
    ]

    def __init__(self, separators: Optional[List[str]] = None):
        """
        Initialize with optional custom separators.

        Args:
            separators: List of separators from coarsest to finest
        """
        self.separators = separators or self.DEFAULT_SEPARATORS

    def chunk(self, text: str, config: ProcessingConfig) -> List[str]:
        """Recursively split text into chunks."""
        chunk_size = config.chunk_size
        overlap = config.chunk_overlap

        if not text:
            return []

        return self._recursive_split(text, chunk_size, overlap, 0)

    def _recursive_split(
        self,
        text: str,
        chunk_size: int,
        overlap: int,
        separator_index: int,
    ) -> List[str]:
        """
        Recursively split text using progressively finer separators.

        Args:
            text: Text to split
            chunk_size: Maximum chunk size
            overlap: Overlap between chunks
            separator_index: Current separator to try

        Returns:
            List of chunks
        """
        # Base case: text is small enough
        if len(text) <= chunk_size:
            return [text.strip()] if text.strip() else []

        # Try current separator
        if separator_index >= len(self.separators):
            # Last resort: force split at chunk_size
            return self._force_split(text, chunk_size, overlap)

        separator = self.separators[separator_index]

        if separator:
            parts = text.split(separator)
        else:
            # Empty separator means split by character
            parts = list(text)

        # Check if splitting helped
        if len(parts) == 1:
            # Separator not found, try next one
            return self._recursive_split(
                text, chunk_size, overlap, separator_index + 1
            )

        # Merge small parts and recursively split large ones
        chunks = []
        current_chunk = []
        current_length = 0

        for part in parts:
            part = part.strip()
            if not part:
                continue

            part_length = len(part)

            # If part itself is too large, recursively split it
            if part_length > chunk_size:
                # First, add current accumulated chunk
                if current_chunk:
                    merged = separator.join(current_chunk)
                    chunks.append(merged)
                    current_chunk = []
                    current_length = 0

                # Recursively split the large part
                sub_chunks = self._recursive_split(
                    part, chunk_size, overlap, separator_index + 1
                )
                chunks.extend(sub_chunks)
                continue

            # Check if adding this part would exceed chunk size
            potential_length = current_length + part_length
            if current_chunk:
                potential_length += len(separator)

            if potential_length > chunk_size:
                # Save current chunk and start new one
                if current_chunk:
                    merged = separator.join(current_chunk)
                    chunks.append(merged)
                current_chunk = [part]
                current_length = part_length
            else:
                current_chunk.append(part)
                current_length = potential_length

        # Add final chunk
        if current_chunk:
            merged = separator.join(current_chunk)
            chunks.append(merged)

        return [c for c in chunks if c.strip()]

    def _force_split(
        self,
        text: str,
        chunk_size: int,
        overlap: int,
    ) -> List[str]:
        """Force split text at exact chunk_size boundaries."""
        chunks = []
        start = 0

        while start < len(text):
            end = min(start + chunk_size, len(text))
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
            start = end - overlap if overlap > 0 else end

        return chunks


class SemanticChunker(BaseChunker):
    """
    Split text based on semantic similarity.

    Uses embeddings to identify natural break points where topic changes.
    Requires an embedding function to be provided in config.

    Good for: Topic-based retrieval, coherent chunks.
    """

    def __init__(
        self,
        similarity_threshold: float = 0.5,
        min_chunk_size: int = 100,
    ):
        """
        Initialize semantic chunker.

        Args:
            similarity_threshold: Threshold below which to split
            min_chunk_size: Minimum chunk size before considering split
        """
        self.similarity_threshold = similarity_threshold
        self.min_chunk_size = min_chunk_size

    def chunk(self, text: str, config: ProcessingConfig) -> List[str]:
        """
        Split text based on semantic similarity.

        Falls back to sentence chunking if no embedding function is provided.
        """
        if not config.embedding_fn:
            # Fallback to sentence chunking
            fallback = SentenceChunker()
            return fallback.chunk(text, config)

        if not text:
            return []

        # Split into sentences first
        sentence_chunker = SentenceChunker()
        sentences = sentence_chunker._split_sentences(text)
        sentences = [s.strip() for s in sentences if s.strip()]

        if len(sentences) <= 1:
            return sentences

        # Get embeddings for each sentence
        embeddings = [config.embedding_fn(s) for s in sentences]

        # Find break points based on similarity
        chunks = []
        current_chunk = [sentences[0]]

        for i in range(1, len(sentences)):
            # Calculate similarity with previous sentence
            similarity = self._cosine_similarity(embeddings[i - 1], embeddings[i])

            # Check if we should split here
            current_length = sum(len(s) for s in current_chunk)

            if similarity < self.similarity_threshold and current_length >= self.min_chunk_size:
                # Topic change detected, start new chunk
                chunks.append(" ".join(current_chunk))
                current_chunk = [sentences[i]]
            else:
                current_chunk.append(sentences[i])

            # Also check max chunk size
            if current_length >= config.chunk_size:
                chunks.append(" ".join(current_chunk))
                current_chunk = []

        # Add final chunk
        if current_chunk:
            chunks.append(" ".join(current_chunk))

        return chunks

    def _cosine_similarity(
        self,
        vec1: List[float],
        vec2: List[float],
    ) -> float:
        """Calculate cosine similarity between two vectors."""
        if not vec1 or not vec2 or len(vec1) != len(vec2):
            return 0.0

        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        magnitude1 = sum(a * a for a in vec1) ** 0.5
        magnitude2 = sum(b * b for b in vec2) ** 0.5

        if magnitude1 == 0 or magnitude2 == 0:
            return 0.0

        return dot_product / (magnitude1 * magnitude2)


# =============================================================================
# FACTORY FUNCTION
# =============================================================================


def get_default_chunker(method: ChunkingMethod) -> BaseChunker:
    """
    Get the default chunker for a given method.

    Args:
        method: The chunking method to use

    Returns:
        A chunker instance
    """
    chunker_map = {
        ChunkingMethod.FIXED_SIZE: FixedSizeChunker,
        ChunkingMethod.SENTENCE: SentenceChunker,
        ChunkingMethod.PARAGRAPH: RecursiveChunker,
        ChunkingMethod.RECURSIVE: RecursiveChunker,
        ChunkingMethod.SEMANTIC: SemanticChunker,
    }

    chunker_class = chunker_map.get(method, RecursiveChunker)
    return chunker_class()
