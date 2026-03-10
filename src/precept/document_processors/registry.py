"""
Processor Registry for PRECEPT Document Processors.

This module implements a plugin-style registry that allows dynamic
registration of document processors without modifying core code.

PLUGIN PATTERN:
═══════════════════════════════════════════════════════════════════════════════════
Third-party processors can be registered using:

1. Decorator-based registration:

    @ProcessorRegistry.register("custom", extensions=[".xyz"])
    class CustomProcessor(DocumentProcessor):
        ...

2. Programmatic registration:

    ProcessorRegistry.add("custom", CustomProcessor, extensions=[".xyz"])

3. Auto-discovery from installed packages (future enhancement)
"""

from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Type, Union

from .base import DocumentProcessor, SourceType


class ProcessorRegistry:
    """
    Registry for document processors.

    Implements the Plugin Pattern for extensibility.
    """

    # Class-level storage for registered processors
    _processors: Dict[str, Type[DocumentProcessor]] = {}
    _extension_map: Dict[str, str] = {}  # extension -> processor_name
    _url_patterns: Dict[str, str] = {}  # pattern -> processor_name
    _source_type_map: Dict[SourceType, List[str]] = {}

    @classmethod
    def register(
        cls,
        name: str,
        extensions: Optional[List[str]] = None,
        url_patterns: Optional[List[str]] = None,
        source_types: Optional[List[SourceType]] = None,
    ) -> Callable[[Type[DocumentProcessor]], Type[DocumentProcessor]]:
        """
        Decorator for registering a processor.

        Usage:
            @ProcessorRegistry.register("pdf", extensions=[".pdf"])
            class PDFProcessor(DocumentProcessor):
                ...

        Args:
            name: Unique name for this processor
            extensions: File extensions this processor handles
            url_patterns: URL patterns this processor handles
            source_types: Source types this processor can handle

        Returns:
            Decorator function
        """

        def decorator(
            processor_class: Type[DocumentProcessor],
        ) -> Type[DocumentProcessor]:
            cls.add(
                name=name,
                processor_class=processor_class,
                extensions=extensions,
                url_patterns=url_patterns,
                source_types=source_types,
            )
            return processor_class

        return decorator

    @classmethod
    def add(
        cls,
        name: str,
        processor_class: Type[DocumentProcessor],
        extensions: Optional[List[str]] = None,
        url_patterns: Optional[List[str]] = None,
        source_types: Optional[List[SourceType]] = None,
    ) -> None:
        """
        Programmatically register a processor.

        Args:
            name: Unique name for this processor
            processor_class: The processor class to register
            extensions: File extensions this processor handles
            url_patterns: URL patterns this processor handles
            source_types: Source types this processor can handle
        """
        # Validate
        if not issubclass(processor_class, DocumentProcessor):
            raise TypeError(
                f"Processor must be a subclass of DocumentProcessor, "
                f"got {processor_class.__name__}"
            )

        # Register processor
        cls._processors[name] = processor_class

        # Map extensions to processor
        if extensions:
            for ext in extensions:
                ext_lower = ext.lower()
                if not ext_lower.startswith("."):
                    ext_lower = f".{ext_lower}"
                cls._extension_map[ext_lower] = name

        # Map URL patterns to processor
        if url_patterns:
            for pattern in url_patterns:
                cls._url_patterns[pattern.lower()] = name

        # Map source types to processor
        source_types = source_types or [SourceType.FILE]
        for source_type in source_types:
            if source_type not in cls._source_type_map:
                cls._source_type_map[source_type] = []
            if name not in cls._source_type_map[source_type]:
                cls._source_type_map[source_type].append(name)

    @classmethod
    def get(cls, name: str) -> Optional[Type[DocumentProcessor]]:
        """
        Get a processor class by name.

        Args:
            name: The processor name

        Returns:
            The processor class or None if not found
        """
        return cls._processors.get(name)

    @classmethod
    def get_for_extension(cls, extension: str) -> Optional[str]:
        """
        Get processor name for a file extension.

        Args:
            extension: The file extension (with or without dot)

        Returns:
            Processor name or None
        """
        ext = extension.lower()
        if not ext.startswith("."):
            ext = f".{ext}"
        return cls._extension_map.get(ext)

    @classmethod
    def get_for_url(cls, url: str) -> Optional[str]:
        """
        Get processor name for a URL.

        Args:
            url: The URL to check

        Returns:
            Processor name or None
        """
        url_lower = url.lower()
        for pattern, name in cls._url_patterns.items():
            if pattern in url_lower:
                return name
        return None

    @classmethod
    def get_for_source(cls, source: Union[str, Path]) -> Optional[str]:
        """
        Auto-detect the right processor for a source.

        Args:
            source: The source (file path, URL, etc.)

        Returns:
            Processor name or None
        """
        source_str = str(source)

        # Check if it's a URL
        if source_str.startswith(("http://", "https://")):
            processor = cls.get_for_url(source_str)
            if processor:
                return processor
            # Default to web scraping for URLs
            return cls._extension_map.get("web") or cls.get_for_url("http")

        # Check file extension
        if "." in source_str:
            ext = "." + source_str.rsplit(".", 1)[-1].lower()
            processor = cls.get_for_extension(ext)
            if processor:
                return processor

        return None

    @classmethod
    def list_processors(cls) -> Dict[str, Dict[str, Any]]:
        """
        List all registered processors with their info.

        Returns:
            Dict mapping processor names to their info
        """
        result = {}
        for name, processor_class in cls._processors.items():
            # Get extensions for this processor
            extensions = [
                ext for ext, proc_name in cls._extension_map.items()
                if proc_name == name
            ]

            # Get URL patterns for this processor
            patterns = [
                pattern for pattern, proc_name in cls._url_patterns.items()
                if proc_name == name
            ]

            result[name] = {
                "class": processor_class.__name__,
                "extensions": extensions,
                "url_patterns": patterns,
                "info": processor_class.get_processor_info()
                if hasattr(processor_class, "get_processor_info")
                else {},
            }

        return result

    @classmethod
    def clear(cls) -> None:
        """Clear all registered processors. Useful for testing."""
        cls._processors.clear()
        cls._extension_map.clear()
        cls._url_patterns.clear()
        cls._source_type_map.clear()

    @classmethod
    def is_registered(cls, name: str) -> bool:
        """Check if a processor is registered."""
        return name in cls._processors

    @classmethod
    def count(cls) -> int:
        """Get the number of registered processors."""
        return len(cls._processors)
