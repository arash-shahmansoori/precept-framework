"""
Markdown Document Processor for PRECEPT.

Handles Markdown files with support for:
- Heading extraction
- Code block preservation
- Link/image extraction
- Front matter parsing (YAML)
"""

import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from ..base import (
    DocumentMetadata,
    DocumentProcessor,
    ProcessingConfig,
    ProcessingResult,
    SourceType,
)
from ..registry import ProcessorRegistry

# Optional import for front matter
try:
    import yaml

    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False


@ProcessorRegistry.register(
    "markdown",
    extensions=[".md", ".markdown", ".mdown", ".mkd"],
    source_types=[SourceType.FILE, SourceType.RAW],
)
class MarkdownProcessor(DocumentProcessor):
    """
    Processor for Markdown documents.

    Features:
    - Extracts headings for structure
    - Preserves code blocks
    - Parses YAML front matter
    - Handles links and images
    """

    processor_name = "markdown"
    supported_extensions = [".md", ".markdown", ".mdown", ".mkd"]
    supported_source_types = [SourceType.FILE, SourceType.RAW]

    # Regex patterns for Markdown elements
    FRONT_MATTER_PATTERN = re.compile(r"^---\s*\n(.*?)\n---\s*\n", re.DOTALL)
    HEADING_PATTERN = re.compile(r"^(#{1,6})\s+(.+)$", re.MULTILINE)
    CODE_BLOCK_PATTERN = re.compile(r"```(\w+)?\n(.*?)```", re.DOTALL)
    LINK_PATTERN = re.compile(r"\[([^\]]+)\]\(([^)]+)\)")
    IMAGE_PATTERN = re.compile(r"!\[([^\]]*)\]\(([^)]+)\)")

    def __init__(
        self,
        config: Optional[ProcessingConfig] = None,
        preserve_formatting: bool = True,
        extract_links: bool = True,
        **kwargs: Any,
    ):
        """
        Initialize Markdown processor.

        Args:
            config: Processing configuration
            preserve_formatting: Keep Markdown formatting in output
            extract_links: Extract and list links at the end
        """
        super().__init__(config=config, **kwargs)
        self.preserve_formatting = preserve_formatting
        self.extract_links = extract_links

    async def process(
        self,
        source: Union[str, Path],
        **kwargs: Any,
    ) -> ProcessingResult:
        """
        Process a Markdown document.

        Args:
            source: Path to Markdown file or raw content
            **kwargs: Additional options
                - is_raw: Treat source as raw content

        Returns:
            ProcessingResult with chunks and metadata
        """
        start_time = time.time()

        try:
            is_raw = kwargs.get("is_raw", False)

            if is_raw:
                text = str(source)
                source_str = "raw_markdown"
            else:
                text = await self.extract_text(source)
                source_str = str(source)

            # Extract metadata (including front matter)
            metadata = await self.extract_metadata(source)
            metadata.processor_name = self.processor_name

            # Parse front matter if present
            front_matter, text = self._parse_front_matter(text)
            if front_matter:
                metadata.custom["front_matter"] = front_matter
                if "title" in front_matter:
                    metadata.title = front_matter["title"]
                if "author" in front_matter:
                    metadata.author = front_matter["author"]
                if "date" in front_matter:
                    metadata.created_date = str(front_matter["date"])

            # Clean text
            text = self.clean_text(text)

            # Optionally append links
            if self.extract_links:
                links = self._extract_links(text)
                if links:
                    text += "\n\n## References\n"
                    for name, url in links:
                        text += f"- [{name}]({url})\n"

            # Create chunks
            chunks = self.create_chunks(text, source_str, metadata)

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
                    file_type="markdown",
                    processor_name=self.processor_name,
                ),
                error=str(e),
                error_details=f"Failed to process Markdown: {e}",
                processing_time_ms=(time.time() - start_time) * 1000,
            )

    async def extract_text(self, source: Union[str, Path]) -> str:
        """
        Extract text from Markdown file.

        Args:
            source: Path to Markdown file

        Returns:
            File contents
        """
        path = Path(source)

        if not path.exists():
            raise FileNotFoundError(f"Markdown file not found: {source}")

        with open(path, "r", encoding="utf-8") as f:
            return f.read()

    async def extract_metadata(
        self, source: Union[str, Path]
    ) -> DocumentMetadata:
        """
        Extract metadata from Markdown document.

        Args:
            source: Path to Markdown file

        Returns:
            Extracted metadata
        """
        metadata = DocumentMetadata(
            source=str(source),
            source_type=SourceType.FILE,
            file_type="markdown",
        )

        try:
            if isinstance(source, (str, Path)):
                path = Path(source)
                if path.exists():
                    # Title from filename
                    if self.config.extract_title:
                        metadata.title = path.stem.replace("-", " ").replace("_", " ").title()

                    # Read content for heading extraction
                    content = await self.extract_text(source)

                    # Extract headings
                    if self.config.extract_headings:
                        metadata.headings = self._extract_headings(content)

                        # Use first H1 as title if available
                        for heading in metadata.headings:
                            if heading.startswith("# "):
                                metadata.title = heading[2:].strip()
                                break

                    # Word count
                    metadata.word_count = len(content.split())

                    # File dates
                    stat = path.stat()
                    metadata.modified_date = time.ctime(stat.st_mtime)

        except Exception:
            pass

        return metadata

    def _parse_front_matter(self, text: str) -> tuple[Dict[str, Any], str]:
        """
        Parse YAML front matter from Markdown.

        Args:
            text: Full Markdown content

        Returns:
            Tuple of (front_matter_dict, remaining_text)
        """
        match = self.FRONT_MATTER_PATTERN.match(text)

        if not match:
            return {}, text

        front_matter_text = match.group(1)
        remaining_text = text[match.end() :]

        if YAML_AVAILABLE:
            try:
                front_matter = yaml.safe_load(front_matter_text)
                if isinstance(front_matter, dict):
                    return front_matter, remaining_text
            except yaml.YAMLError:
                pass

        return {}, remaining_text

    def _extract_headings(self, text: str) -> List[str]:
        """Extract all headings from Markdown."""
        headings = []

        for match in self.HEADING_PATTERN.finditer(text):
            level = len(match.group(1))
            heading_text = match.group(2).strip()
            prefix = "#" * level
            headings.append(f"{prefix} {heading_text}")

        return headings

    def _extract_links(self, text: str) -> List[tuple[str, str]]:
        """Extract all links from Markdown."""
        links = []

        for match in self.LINK_PATTERN.finditer(text):
            name = match.group(1)
            url = match.group(2)
            # Skip image links and anchors
            if not url.startswith("#") and not url.startswith("data:"):
                links.append((name, url))

        return links

    def clean_text(self, text: str) -> str:
        """
        Clean Markdown text.

        Args:
            text: Raw Markdown content

        Returns:
            Cleaned text
        """
        text = super().clean_text(text)

        if not self.preserve_formatting:
            # Remove Markdown formatting for plain text output
            # Remove code blocks but preserve content
            text = self.CODE_BLOCK_PATTERN.sub(r"\n\2\n", text)

            # Remove emphasis markers
            text = re.sub(r"\*\*(.+?)\*\*", r"\1", text)
            text = re.sub(r"\*(.+?)\*", r"\1", text)
            text = re.sub(r"__(.+?)__", r"\1", text)
            text = re.sub(r"_(.+?)_", r"\1", text)

            # Convert links to plain text
            text = self.LINK_PATTERN.sub(r"\1", text)

            # Remove images
            text = self.IMAGE_PATTERN.sub(r"[Image: \1]", text)

        return text
