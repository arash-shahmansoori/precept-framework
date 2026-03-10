"""
Text Document Processor for PRECEPT.

Handles plain text files (.txt, .text, .log, etc.)
"""

import time
from pathlib import Path
from typing import Any, Optional, Union

from ..base import (
    DocumentChunk,
    DocumentMetadata,
    DocumentProcessor,
    ProcessingConfig,
    ProcessingResult,
    SourceType,
)
from ..registry import ProcessorRegistry


@ProcessorRegistry.register(
    "text",
    extensions=[".txt", ".text", ".log", ".csv", ".tsv"],
    source_types=[SourceType.FILE, SourceType.RAW],
)
class TextProcessor(DocumentProcessor):
    """
    Processor for plain text files.

    This is the simplest processor and serves as a reference implementation.
    """

    processor_name = "text"
    supported_extensions = [".txt", ".text", ".log", ".csv", ".tsv"]
    supported_source_types = [SourceType.FILE, SourceType.RAW]

    async def process(
        self,
        source: Union[str, Path],
        **kwargs: Any,
    ) -> ProcessingResult:
        """
        Process a text file.

        Args:
            source: Path to text file or raw text content
            **kwargs: Additional options
                - encoding: File encoding (default: utf-8)
                - is_raw: Treat source as raw content, not a file path

        Returns:
            ProcessingResult with chunks and metadata
        """
        start_time = time.time()

        try:
            # Determine if source is raw text or file path
            is_raw = kwargs.get("is_raw", False)

            # Extract text
            if is_raw:
                text = str(source)
                source_str = "raw_text"
            else:
                text = await self.extract_text(source)
                source_str = str(source)

            # Extract metadata
            metadata = await self.extract_metadata(source)
            metadata.processor_name = self.processor_name

            # Clean text
            text = self.clean_text(text)

            # Check max content length
            if self.config.max_content_length and len(text) > self.config.max_content_length:
                text = text[: self.config.max_content_length]

            # Create chunks
            chunks = self.create_chunks(text, source_str, metadata)

            # Calculate processing time
            processing_time = (time.time() - start_time) * 1000
            metadata.processing_time_ms = processing_time

            return ProcessingResult(
                success=True,
                chunks=chunks,
                metadata=metadata,
                processing_time_ms=processing_time,
            )

        except Exception as e:
            return ProcessingResult(
                success=False,
                chunks=[],
                metadata=DocumentMetadata(
                    source=str(source),
                    source_type=SourceType.FILE,
                    processor_name=self.processor_name,
                ),
                error=str(e),
                error_details=f"Failed to process text file: {e}",
                processing_time_ms=(time.time() - start_time) * 1000,
            )

    async def extract_text(self, source: Union[str, Path]) -> str:
        """
        Extract text from a file.

        Args:
            source: Path to the text file

        Returns:
            File contents as string
        """
        path = Path(source)

        if not path.exists():
            raise FileNotFoundError(f"File not found: {source}")

        # Try common encodings
        encodings = ["utf-8", "utf-8-sig", "latin-1", "cp1252"]

        for encoding in encodings:
            try:
                with open(path, "r", encoding=encoding) as f:
                    return f.read()
            except UnicodeDecodeError:
                continue

        # Last resort: read as bytes and decode with errors='replace'
        with open(path, "rb") as f:
            content = f.read()
            return content.decode("utf-8", errors="replace")

    async def extract_metadata(
        self, source: Union[str, Path]
    ) -> DocumentMetadata:
        """
        Extract metadata from a text file.

        Args:
            source: Path to the text file

        Returns:
            Extracted metadata
        """
        path = Path(source) if not isinstance(source, Path) else source

        metadata = DocumentMetadata(
            source=str(source),
            source_type=SourceType.FILE,
            file_type="txt",
        )

        if path.exists():
            stat = path.stat()
            metadata.modified_date = time.ctime(stat.st_mtime)

            # Get word count (estimate)
            try:
                content = await self.extract_text(source)
                metadata.word_count = len(content.split())
            except Exception:
                pass

            # Extract title from filename
            if self.config.extract_title:
                metadata.title = path.stem.replace("_", " ").replace("-", " ").title()

        return metadata
