"""
Processor Factory for PRECEPT Document Processors.

This module provides a factory for creating document processors
with automatic type detection.

FACTORY PATTERN:
═══════════════════════════════════════════════════════════════════════════════════
The factory encapsulates processor creation logic:
- Auto-detects document type from source
- Returns the appropriate processor instance
- Supports configuration injection
"""

from pathlib import Path
from typing import Any, Dict, Optional, Type, Union

from .base import DocumentProcessor, ProcessingConfig
from .registry import ProcessorRegistry


class ProcessorFactory:
    """
    Factory for creating document processors.

    Supports:
    - Auto-detection from file extension/URL
    - Explicit processor selection by name
    - Configuration injection
    """

    @classmethod
    def create(
        cls,
        source: Union[str, Path],
        processor_name: Optional[str] = None,
        config: Optional[ProcessingConfig] = None,
        **kwargs: Any,
    ) -> DocumentProcessor:
        """
        Create a processor for the given source.

        Args:
            source: The document source (file path, URL, etc.)
            processor_name: Explicit processor name (auto-detects if not provided)
            config: Processing configuration
            **kwargs: Additional arguments passed to processor constructor

        Returns:
            An initialized processor instance

        Raises:
            ValueError: If no suitable processor is found
        """
        # Auto-detect processor if not specified
        if processor_name is None:
            processor_name = ProcessorRegistry.get_for_source(source)

        if processor_name is None:
            # Try to infer from source
            source_str = str(source).lower()

            # Default mappings for common types
            if source_str.endswith(".pdf"):
                processor_name = "pdf"
            elif source_str.endswith((".txt", ".text")):
                processor_name = "text"
            elif source_str.endswith((".md", ".markdown")):
                processor_name = "markdown"
            elif source_str.startswith(("http://", "https://")):
                processor_name = "web"
            else:
                # Default to text processor
                processor_name = "text"

        # Get processor class
        processor_class = ProcessorRegistry.get(processor_name)

        if processor_class is None:
            available = list(ProcessorRegistry.list_processors().keys())
            raise ValueError(
                f"No processor found for '{processor_name}'. "
                f"Available processors: {available}"
            )

        # Create and return instance
        return processor_class(config=config, **kwargs)

    @classmethod
    def create_by_type(
        cls,
        processor_type: Type[DocumentProcessor],
        config: Optional[ProcessingConfig] = None,
        **kwargs: Any,
    ) -> DocumentProcessor:
        """
        Create a processor by its class type.

        Args:
            processor_type: The processor class
            config: Processing configuration
            **kwargs: Additional arguments

        Returns:
            An initialized processor instance
        """
        return processor_type(config=config, **kwargs)

    @classmethod
    def get_available_processors(cls) -> Dict[str, Any]:
        """
        Get information about all available processors.

        Returns:
            Dict with processor info
        """
        return ProcessorRegistry.list_processors()

    @classmethod
    def supports(cls, source: Union[str, Path]) -> bool:
        """
        Check if there's a processor that supports the given source.

        Args:
            source: The source to check

        Returns:
            True if a processor is available
        """
        processor_name = ProcessorRegistry.get_for_source(source)
        return processor_name is not None

    @classmethod
    def get_processor_for_extension(cls, extension: str) -> Optional[str]:
        """
        Get the processor name for a file extension.

        Args:
            extension: File extension (e.g., ".pdf", "pdf")

        Returns:
            Processor name or None
        """
        return ProcessorRegistry.get_for_extension(extension)
