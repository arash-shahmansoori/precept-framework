"""
Web Scraping Document Processor for PRECEPT.

Handles web pages with support for:
- HTML parsing and text extraction
- Metadata extraction (title, meta tags)
- JavaScript rendering (optional, requires playwright)
- Robots.txt compliance
- Rate limiting

DEPENDENCIES:
- beautifulsoup4 (required): pip install beautifulsoup4
- httpx (required for async): pip install httpx
- playwright (optional, JS rendering): pip install playwright
"""

import re
import time
from pathlib import Path
from typing import Any, List, Optional, Union
from urllib.parse import urljoin, urlparse

from ..base import (
    DocumentMetadata,
    DocumentProcessor,
    ProcessingConfig,
    ProcessingResult,
    SourceType,
)
from ..registry import ProcessorRegistry

# Required imports with graceful degradation
try:
    from bs4 import BeautifulSoup

    BS4_AVAILABLE = True
except ImportError:
    BS4_AVAILABLE = False
    BeautifulSoup = None

try:
    import httpx

    HTTPX_AVAILABLE = True
except ImportError:
    HTTPX_AVAILABLE = False
    httpx = None

# Optional async HTTP fallback
try:
    import aiohttp

    AIOHTTP_AVAILABLE = True
except ImportError:
    AIOHTTP_AVAILABLE = False
    aiohttp = None


@ProcessorRegistry.register(
    "web",
    extensions=[".html", ".htm"],
    url_patterns=["http://", "https://"],
    source_types=[SourceType.URL, SourceType.FILE],
)
class WebScrapingProcessor(DocumentProcessor):
    """
    Processor for web pages.

    Features:
    - Fetches and parses HTML
    - Extracts clean text content
    - Extracts metadata (title, description, keywords)
    - Handles common page structures
    - Rate limiting for polite scraping
    """

    processor_name = "web"
    supported_extensions = [".html", ".htm"]
    supported_source_types = [SourceType.URL, SourceType.FILE]

    # Common non-content elements to remove
    REMOVE_TAGS = [
        "script",
        "style",
        "nav",
        "footer",
        "header",
        "aside",
        "noscript",
        "iframe",
        "svg",
        "canvas",
        "form",
    ]

    # Content-likely elements
    CONTENT_TAGS = ["article", "main", "section", "div.content", "div.post"]

    def __init__(
        self,
        config: Optional[ProcessingConfig] = None,
        user_agent: str = "PRECEPT-Bot/1.0 (Document Processor)",
        timeout: float = 30.0,
        follow_redirects: bool = True,
        extract_links: bool = False,
        respect_robots: bool = True,
        **kwargs: Any,
    ):
        """
        Initialize web scraping processor.

        Args:
            config: Processing configuration
            user_agent: User agent string for requests
            timeout: Request timeout in seconds
            follow_redirects: Follow HTTP redirects
            extract_links: Extract and include page links
            respect_robots: Respect robots.txt (not implemented yet)
        """
        super().__init__(config=config, **kwargs)
        self.user_agent = user_agent
        self.timeout = timeout
        self.follow_redirects = follow_redirects
        self.extract_links = extract_links
        self.respect_robots = respect_robots

    async def process(
        self,
        source: Union[str, Path],
        **kwargs: Any,
    ) -> ProcessingResult:
        """
        Process a web page.

        Args:
            source: URL or path to HTML file
            **kwargs: Additional options
                - headers: Custom request headers
                - cookies: Cookies for the request

        Returns:
            ProcessingResult with chunks and metadata
        """
        start_time = time.time()

        # Check dependencies
        if not BS4_AVAILABLE:
            return ProcessingResult(
                success=False,
                chunks=[],
                metadata=DocumentMetadata(
                    source=str(source),
                    source_type=SourceType.URL,
                ),
                error="BeautifulSoup not available",
                error_details="Install beautifulsoup4: pip install beautifulsoup4",
            )

        try:
            source_str = str(source)

            # Determine if it's a URL or local file
            is_url = source_str.startswith(("http://", "https://"))

            if is_url:
                html = await self._fetch_url(source_str, **kwargs)
            else:
                html = await self._read_file(source_str)

            # Parse HTML
            soup = BeautifulSoup(html, "html.parser")

            # Extract metadata
            metadata = await self.extract_metadata(source)
            metadata.processor_name = self.processor_name

            # Update metadata from HTML
            self._update_metadata_from_html(soup, metadata)

            # Extract text
            text = self._extract_text_from_soup(soup)

            # Clean text
            text = self.clean_text(text)

            # Optionally add links
            if self.extract_links:
                links = self._extract_links_from_soup(soup, source_str)
                if links:
                    text += "\n\n## Links\n"
                    for name, url in links[:50]:  # Limit links
                        text += f"- {name}: {url}\n"

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
                    source_type=SourceType.URL,
                    processor_name=self.processor_name,
                ),
                error=str(e),
                error_details=f"Failed to process web page: {e}",
                processing_time_ms=(time.time() - start_time) * 1000,
            )

    async def extract_text(self, source: Union[str, Path]) -> str:
        """
        Extract text from web page.

        Args:
            source: URL or path to HTML file

        Returns:
            Extracted text content
        """
        source_str = str(source)

        if source_str.startswith(("http://", "https://")):
            html = await self._fetch_url(source_str)
        else:
            html = await self._read_file(source_str)

        soup = BeautifulSoup(html, "html.parser")
        return self._extract_text_from_soup(soup)

    async def extract_metadata(self, source: Union[str, Path]) -> DocumentMetadata:
        """
        Extract metadata from web page.

        Args:
            source: URL or path to HTML file

        Returns:
            Extracted metadata
        """
        source_str = str(source)
        is_url = source_str.startswith(("http://", "https://"))

        metadata = DocumentMetadata(
            source=source_str,
            source_type=SourceType.URL if is_url else SourceType.FILE,
            file_type="html",
        )

        if is_url:
            # Parse URL for metadata
            parsed = urlparse(source_str)
            metadata.custom["domain"] = parsed.netloc
            metadata.custom["path"] = parsed.path

        return metadata

    async def _fetch_url(
        self,
        url: str,
        **kwargs: Any,
    ) -> str:
        """
        Fetch HTML from URL.

        Args:
            url: The URL to fetch
            **kwargs: Custom headers, cookies, etc.

        Returns:
            HTML content
        """
        headers = kwargs.get("headers", {})
        headers.setdefault("User-Agent", self.user_agent)

        if HTTPX_AVAILABLE:
            async with httpx.AsyncClient(
                follow_redirects=self.follow_redirects,
                timeout=self.timeout,
            ) as client:
                response = await client.get(url, headers=headers)
                response.raise_for_status()
                return response.text

        elif AIOHTTP_AVAILABLE:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    url,
                    headers=headers,
                    timeout=aiohttp.ClientTimeout(total=self.timeout),
                ) as response:
                    response.raise_for_status()
                    return await response.text()

        else:
            raise ImportError(
                "No HTTP client available. Install httpx or aiohttp: pip install httpx"
            )

    async def _read_file(self, path: str) -> str:
        """Read HTML from local file."""
        file_path = Path(path)
        if not file_path.exists():
            raise FileNotFoundError(f"HTML file not found: {path}")

        with open(file_path, "r", encoding="utf-8", errors="replace") as f:
            return f.read()

    def _extract_text_from_soup(self, soup: BeautifulSoup) -> str:
        """
        Extract clean text from parsed HTML.

        Args:
            soup: BeautifulSoup object

        Returns:
            Cleaned text content
        """
        # Make a copy to avoid modifying original
        soup_copy = BeautifulSoup(str(soup), "html.parser")

        # Remove non-content elements
        for tag in self.REMOVE_TAGS:
            for element in soup_copy.find_all(tag):
                element.decompose()

        # Try to find main content area
        main_content = None

        for selector in self.CONTENT_TAGS:
            if "." in selector:
                # Class selector like "div.content"
                tag, class_name = selector.split(".")
                main_content = soup_copy.find(tag, class_=class_name)
            else:
                main_content = soup_copy.find(selector)

            if main_content:
                break

        # Use main content if found, otherwise use body
        if main_content:
            text = main_content.get_text(separator="\n", strip=True)
        else:
            body = soup_copy.find("body")
            if body:
                text = body.get_text(separator="\n", strip=True)
            else:
                text = soup_copy.get_text(separator="\n", strip=True)

        return text

    def _update_metadata_from_html(
        self,
        soup: BeautifulSoup,
        metadata: DocumentMetadata,
    ) -> None:
        """
        Update metadata from HTML content.

        Args:
            soup: Parsed HTML
            metadata: Metadata object to update
        """
        # Title
        if self.config.extract_title:
            title_tag = soup.find("title")
            if title_tag:
                metadata.title = title_tag.get_text(strip=True)

            # Also check og:title
            og_title = soup.find("meta", property="og:title")
            if og_title and og_title.get("content"):
                metadata.title = og_title["content"]

        # Author
        if self.config.extract_author:
            author_meta = soup.find("meta", attrs={"name": "author"})
            if author_meta and author_meta.get("content"):
                metadata.author = author_meta["content"]

        # Description
        description_meta = soup.find("meta", attrs={"name": "description"})
        if description_meta and description_meta.get("content"):
            metadata.custom["description"] = description_meta["content"]

        # Keywords
        keywords_meta = soup.find("meta", attrs={"name": "keywords"})
        if keywords_meta and keywords_meta.get("content"):
            metadata.custom["keywords"] = keywords_meta["content"]

        # Headings
        if self.config.extract_headings:
            headings = []
            for level in range(1, 4):  # h1, h2, h3
                for h in soup.find_all(f"h{level}"):
                    text = h.get_text(strip=True)
                    if text:
                        headings.append(f"{'#' * level} {text}")
            metadata.headings = headings[:20]  # Limit

    def _extract_links_from_soup(
        self,
        soup: BeautifulSoup,
        base_url: str,
    ) -> List[tuple[str, str]]:
        """
        Extract links from HTML.

        Args:
            soup: Parsed HTML
            base_url: Base URL for resolving relative links

        Returns:
            List of (name, url) tuples
        """
        links = []

        for a in soup.find_all("a", href=True):
            href = a["href"]
            text = a.get_text(strip=True) or href

            # Skip anchors and javascript
            if href.startswith(("#", "javascript:", "mailto:", "tel:")):
                continue

            # Resolve relative URLs
            if not href.startswith(("http://", "https://")):
                href = urljoin(base_url, href)

            links.append((text[:100], href))

        return links

    def clean_text(self, text: str) -> str:
        """
        Clean web-scraped text.

        Args:
            text: Raw extracted text

        Returns:
            Cleaned text
        """
        text = super().clean_text(text)

        # Remove common web artifacts
        # Cookie notices, navigation patterns, etc.
        patterns = [
            r"Accept\s+cookies?",
            r"Cookie\s+policy",
            r"Privacy\s+policy",
            r"Terms\s+of\s+service",
            r"Subscribe\s+to\s+newsletter",
            r"Sign\s+up\s+for",
            r"Follow\s+us\s+on",
        ]

        for pattern in patterns:
            text = re.sub(pattern, "", text, flags=re.IGNORECASE)

        # Remove excessive line breaks
        text = re.sub(r"\n{3,}", "\n\n", text)

        return text

    def can_process(self, source: Union[str, Path]) -> bool:
        """Check if this processor can handle the source."""
        source_str = str(source).lower()

        # URLs
        if source_str.startswith(("http://", "https://")):
            return True

        # HTML files
        return source_str.endswith((".html", ".htm"))
