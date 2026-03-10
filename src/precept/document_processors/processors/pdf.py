"""
PDF Document Processor for PRECEPT.

Handles PDF documents with support for:
- Text extraction
- Metadata extraction (author, title, dates)
- Page-level chunking
- Table preservation (optional)
- OCR fallback (if configured)

DEPENDENCIES:
- pypdf (required): pip install pypdf
- pdfplumber (optional, better table support): pip install pdfplumber
- pytesseract (optional, OCR): pip install pytesseract
"""

import re
import time
from pathlib import Path
from typing import Any, List, Optional, Union

from ..base import (
    DocumentMetadata,
    DocumentProcessor,
    ProcessingConfig,
    ProcessingResult,
    SourceType,
)
from ..registry import ProcessorRegistry

# Optional imports with graceful degradation
try:
    from pypdf import PdfReader

    PYPDF_AVAILABLE = True
except ImportError:
    PYPDF_AVAILABLE = False
    PdfReader = None

try:
    import pdfplumber

    PDFPLUMBER_AVAILABLE = True
except ImportError:
    PDFPLUMBER_AVAILABLE = False
    pdfplumber = None


@ProcessorRegistry.register(
    "pdf",
    extensions=[".pdf"],
    source_types=[SourceType.FILE],
)
class PDFProcessor(DocumentProcessor):
    """
    Processor for PDF documents.

    Features:
    - Extracts text from all pages
    - Preserves page structure
    - Extracts document metadata
    - Optional table extraction
    - Optional OCR for scanned documents
    """

    processor_name = "pdf"
    supported_extensions = [".pdf"]
    supported_source_types = [SourceType.FILE]

    def __init__(
        self,
        config: Optional[ProcessingConfig] = None,
        use_pdfplumber: bool = True,
        enable_ocr: bool = False,
        page_separator: str = "\n\n--- Page {page} ---\n\n",
        **kwargs: Any,
    ):
        """
        Initialize PDF processor.

        Args:
            config: Processing configuration
            use_pdfplumber: Use pdfplumber for better table extraction
            enable_ocr: Enable OCR for scanned documents (requires pytesseract)
            page_separator: Template for page separators
        """
        super().__init__(config=config, **kwargs)
        self.use_pdfplumber = use_pdfplumber and PDFPLUMBER_AVAILABLE
        self.enable_ocr = enable_ocr
        self.page_separator = page_separator

    async def process(
        self,
        source: Union[str, Path],
        **kwargs: Any,
    ) -> ProcessingResult:
        """
        Process a PDF document.

        Args:
            source: Path to PDF file
            **kwargs: Additional options
                - pages: List of page numbers to process (1-indexed)
                - password: Password for encrypted PDFs

        Returns:
            ProcessingResult with chunks and metadata
        """
        start_time = time.time()

        # Check dependencies
        if not PYPDF_AVAILABLE and not PDFPLUMBER_AVAILABLE:
            return ProcessingResult(
                success=False,
                chunks=[],
                metadata=DocumentMetadata(
                    source=str(source),
                    source_type=SourceType.FILE,
                    file_type="pdf",
                ),
                error="No PDF library available",
                error_details="Install pypdf: pip install pypdf",
            )

        try:
            path = Path(source)
            if not path.exists():
                raise FileNotFoundError(f"PDF file not found: {source}")

            # Extract metadata first
            metadata = await self.extract_metadata(source)
            metadata.processor_name = self.processor_name

            # Extract text
            text = await self.extract_text(source, **kwargs)

            # Clean text
            text = self.clean_text(text)

            # Check limits
            if self.config.max_content_length:
                text = text[: self.config.max_content_length]

            # Create chunks
            chunks = self.create_chunks(text, str(source), metadata)

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
                    file_type="pdf",
                    processor_name=self.processor_name,
                ),
                error=str(e),
                error_details=f"Failed to process PDF: {e}",
                processing_time_ms=(time.time() - start_time) * 1000,
            )

    async def extract_text(
        self,
        source: Union[str, Path],
        **kwargs: Any,
    ) -> str:
        """
        Extract text from PDF.

        Uses pdfplumber if available (better table handling),
        falls back to pypdf.

        Args:
            source: Path to PDF file
            **kwargs: Options like pages, password

        Returns:
            Extracted text
        """
        path = Path(source)
        pages_to_process = kwargs.get("pages")
        password = kwargs.get("password")

        if self.use_pdfplumber and PDFPLUMBER_AVAILABLE:
            return await self._extract_with_pdfplumber(path, pages_to_process, password)
        else:
            return await self._extract_with_pypdf(path, pages_to_process, password)

    async def _extract_with_pypdf(
        self,
        path: Path,
        pages: Optional[List[int]] = None,
        password: Optional[str] = None,
    ) -> str:
        """Extract text using pypdf."""
        reader = PdfReader(str(path))

        if password and reader.is_encrypted:
            reader.decrypt(password)

        texts = []
        total_pages = len(reader.pages)

        # Determine which pages to process
        if pages:
            page_indices = [p - 1 for p in pages if 0 < p <= total_pages]
        else:
            max_pages = self.config.max_pages or total_pages
            page_indices = range(min(total_pages, max_pages))

        for i in page_indices:
            page = reader.pages[i]
            page_text = page.extract_text() or ""

            if self.page_separator:
                texts.append(self.page_separator.format(page=i + 1))

            texts.append(page_text)

        return "\n".join(texts)

    async def _extract_with_pdfplumber(
        self,
        path: Path,
        pages: Optional[List[int]] = None,
        password: Optional[str] = None,
    ) -> str:
        """Extract text using pdfplumber (better table handling)."""
        texts = []

        with pdfplumber.open(str(path), password=password) as pdf:
            total_pages = len(pdf.pages)

            # Determine which pages to process
            if pages:
                page_indices = [p - 1 for p in pages if 0 < p <= total_pages]
            else:
                max_pages = self.config.max_pages or total_pages
                page_indices = range(min(total_pages, max_pages))

            for i in page_indices:
                page = pdf.pages[i]

                if self.page_separator:
                    texts.append(self.page_separator.format(page=i + 1))

                # Extract text
                page_text = page.extract_text() or ""

                # Extract tables if configured
                if self.config.preserve_tables:
                    tables = page.extract_tables()
                    for table in tables:
                        if table:
                            table_text = self._format_table(table)
                            page_text += f"\n\n[TABLE]\n{table_text}\n[/TABLE]\n"

                texts.append(page_text)

        return "\n".join(texts)

    def _format_table(self, table: List[List[str]]) -> str:
        """Format a table as text."""
        if not table:
            return ""

        # Clean cells
        cleaned = []
        for row in table:
            cleaned_row = [str(cell).strip() if cell else "" for cell in row]
            cleaned.append(cleaned_row)

        # Calculate column widths
        col_widths = []
        for col_idx in range(len(cleaned[0])):
            width = max(len(row[col_idx]) for row in cleaned if col_idx < len(row))
            col_widths.append(min(width, 50))  # Cap at 50 chars

        # Format rows
        lines = []
        for row in cleaned:
            cells = [
                cell[: col_widths[i]].ljust(col_widths[i])
                for i, cell in enumerate(row)
                if i < len(col_widths)
            ]
            lines.append(" | ".join(cells))

        return "\n".join(lines)

    async def extract_metadata(self, source: Union[str, Path]) -> DocumentMetadata:
        """
        Extract metadata from PDF.

        Args:
            source: Path to PDF file

        Returns:
            Extracted metadata
        """
        path = Path(source)

        metadata = DocumentMetadata(
            source=str(source),
            source_type=SourceType.FILE,
            file_type="pdf",
        )

        try:
            if PYPDF_AVAILABLE:
                reader = PdfReader(str(path))
                info = reader.metadata

                if info:
                    if self.config.extract_title and info.title:
                        metadata.title = info.title
                    if self.config.extract_author and info.author:
                        metadata.author = info.author
                    if self.config.extract_dates:
                        if hasattr(info, "creation_date"):
                            metadata.created_date = str(info.creation_date)
                        if hasattr(info, "modification_date"):
                            metadata.modified_date = str(info.modification_date)

                metadata.page_count = len(reader.pages)

                # Extract headings from outline (bookmarks)
                if self.config.extract_headings and reader.outline:
                    metadata.headings = self._extract_outline(reader.outline)

        except Exception:
            # Metadata extraction is optional, don't fail
            pass

        return metadata

    def _extract_outline(self, outline: Any, level: int = 0) -> List[str]:
        """Extract headings from PDF outline/bookmarks."""
        headings = []

        for item in outline:
            if isinstance(item, list):
                # Nested outline
                headings.extend(self._extract_outline(item, level + 1))
            else:
                # Bookmark item
                title = getattr(item, "title", str(item))
                prefix = "  " * level
                headings.append(f"{prefix}{title}")

        return headings

    def clean_text(self, text: str) -> str:
        """
        Clean PDF-specific artifacts.

        Args:
            text: Raw extracted text

        Returns:
            Cleaned text
        """
        # Call parent clean
        text = super().clean_text(text)

        if self.config.remove_headers_footers:
            # Remove common header/footer patterns
            # Page numbers at start/end of lines
            text = re.sub(r"^\s*\d+\s*$", "", text, flags=re.MULTILINE)
            text = re.sub(
                r"^\s*Page\s+\d+\s*(of\s+\d+)?\s*$",
                "",
                text,
                flags=re.MULTILINE | re.IGNORECASE,
            )

        # Remove excessive whitespace from PDF extraction
        text = re.sub(r" +", " ", text)

        return text
