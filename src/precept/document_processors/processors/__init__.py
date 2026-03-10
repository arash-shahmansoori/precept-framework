"""
Concrete Document Processor Implementations.

This module provides ready-to-use processors for common document types:
- PDFProcessor: PDF documents
- TextProcessor: Plain text files
- MarkdownProcessor: Markdown files
- WebScrapingProcessor: Web pages

Each processor is registered with the ProcessorRegistry for auto-detection.
"""

from .markdown import MarkdownProcessor
from .pdf import PDFProcessor
from .text import TextProcessor
from .web import WebScrapingProcessor

__all__ = [
    "PDFProcessor",
    "TextProcessor",
    "MarkdownProcessor",
    "WebScrapingProcessor",
]
