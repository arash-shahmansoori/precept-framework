"""
PRECEPT LLM Clients: Actual API implementations for LLM and Embedding calls.

This module provides the real LLM and embedding implementations used by PRECEPT,
leveraging the same infrastructure as COMPASS.

NO MOCK IMPLEMENTATIONS - All calls go to actual APIs.

Required Dependencies:
- openai
- langchain_openai
- OPENAI_API_KEY environment variable
"""

import os
import sys
from pathlib import Path
from typing import Any, Callable, List, Optional, Type, TypeVar

from pydantic import BaseModel

# Add project root for imports
project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Track if COMPASS model implementations are available
_COMPASS_MODELS_AVAILABLE = False
_IMPORT_ERROR = None

try:
    # Import actual COMPASS model implementations
    from models.openai_api import (
        openai_model_chat_async,
        openai_model_structured_output_async,
        openai_model_context_structured_output_async,
    )
    from models.embedding_models import get_openai_embedding_model
    _COMPASS_MODELS_AVAILABLE = True
except ImportError as e:
    _IMPORT_ERROR = str(e)
    # Define placeholders that will raise clear errors when called
    openai_model_chat_async = None
    openai_model_structured_output_async = None
    openai_model_context_structured_output_async = None
    get_openai_embedding_model = None


def _check_dependencies():
    """Raise clear error if dependencies are not available."""
    if not _COMPASS_MODELS_AVAILABLE:
        raise ImportError(
            f"PRECEPT LLM clients require COMPASS dependencies.\n"
            f"Import error: {_IMPORT_ERROR}\n\n"
            f"Please install required packages:\n"
            f"  pip install openai langchain_openai\n\n"
            f"And ensure OPENAI_API_KEY is set in your environment."
        )

# Type variable for Pydantic models
T = TypeVar('T', bound=BaseModel)


# =============================================================================
# LLM CLIENT - Uses actual OpenAI API
# =============================================================================

async def precept_llm_client(
    system_prompt: str,
    user_prompt: str,
    response_model: Optional[Type[T]] = None,
) -> Any:
    """
    PRECEPT's primary LLM client using actual OpenAI API.

    This is NOT a mock - it makes real API calls to OpenAI.

    Args:
        system_prompt: System instructions for the LLM
        user_prompt: User query/prompt
        response_model: Optional Pydantic model for structured output

    Returns:
        If response_model provided: Parsed Pydantic model instance
        Otherwise: Raw LLM response text

    Raises:
        ImportError: If COMPASS dependencies are not installed
    """
    _check_dependencies()

    if response_model is not None:
        # Use structured output for Pydantic models
        result = await openai_model_structured_output_async(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            response_format=response_model,
        )
        return result
    else:
        # Use regular chat for text responses
        result = await openai_model_chat_async(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
        )
        # Extract content from LangChain response
        if hasattr(result, 'content'):
            return result.content
        return str(result)


async def precept_llm_client_with_context(
    system_prompt: str,
    context: str,
    user_prompt: str,
    response_model: Optional[Type[T]] = None,
) -> Any:
    """
    PRECEPT LLM client with additional context injection.

    Useful for RAG-style prompts where context is retrieved separately.

    Args:
        system_prompt: System instructions
        context: Retrieved context to inject
        user_prompt: User query
        response_model: Optional Pydantic model for structured output

    Returns:
        LLM response (structured or text)

    Raises:
        ImportError: If COMPASS dependencies are not installed
    """
    _check_dependencies()

    if response_model is not None:
        result = await openai_model_context_structured_output_async(
            system_prompt=system_prompt,
            context=context,
            user_prompt=user_prompt,
            response_format=response_model,
        )
        return result
    else:
        # Combine system prompt with context for regular chat
        combined_system = f"{system_prompt}\n\nCONTEXT:\n{context}"
        result = await openai_model_chat_async(
            system_prompt=combined_system,
            user_prompt=user_prompt,
        )
        if hasattr(result, 'content'):
            return result.content
        return str(result)


# =============================================================================
# EMBEDDING CLIENT - Uses actual OpenAI Embeddings API
# =============================================================================

def precept_embedding_fn(text: str) -> List[float]:
    """
    PRECEPT's embedding function using actual OpenAI Embeddings API.

    This is NOT a mock - it makes real API calls to OpenAI.

    Args:
        text: Text to embed

    Returns:
        List of floats representing the embedding vector

    Raises:
        ImportError: If COMPASS dependencies are not installed
    """
    _check_dependencies()
    embedding_model = get_openai_embedding_model()
    embedding = embedding_model.embed_query(text)
    return embedding


def precept_embed_documents(texts: List[str]) -> List[List[float]]:
    """
    Embed multiple documents at once.

    More efficient than calling embed_query multiple times.

    Args:
        texts: List of texts to embed

    Returns:
        List of embedding vectors

    Raises:
        ImportError: If COMPASS dependencies are not installed
    """
    _check_dependencies()
    embedding_model = get_openai_embedding_model()
    embeddings = embedding_model.embed_documents(texts)
    return embeddings


# =============================================================================
# FACTORY FUNCTIONS - Create configured clients
# =============================================================================

def get_precept_llm_client() -> Callable:
    """
    Get the default PRECEPT LLM client.

    Returns:
        Async callable for LLM inference
    """
    return precept_llm_client


def get_precept_embedding_fn() -> Callable:
    """
    Get the default PRECEPT embedding function.

    Returns:
        Callable for generating embeddings
    """
    return precept_embedding_fn


def create_openai_embeddings(model: str = "text-embedding-3-small"):
    """
    Factory function to create OpenAI embeddings with a specific model.

    This is useful when you need embeddings with a different model than the default.
    Uses the same infrastructure as models/embedding_models.py.

    Args:
        model: OpenAI embedding model name (default: text-embedding-3-small)

    Returns:
        LangChain OpenAIEmbeddings object

    Raises:
        ImportError: If langchain_openai is not installed
    """
    try:
        from langchain_openai import OpenAIEmbeddings
        import os

        api_key = os.getenv("OPENAI_API_KEY")
        return OpenAIEmbeddings(model=model, api_key=api_key)
    except ImportError as e:
        raise ImportError(
            f"Cannot create OpenAI embeddings. Please install langchain_openai: {e}"
        )


# =============================================================================
# ALTERNATIVE PROVIDERS (Optional - if other models are configured)
# =============================================================================

try:
    from models.anthropic_api import anthropic_model_chat_async

    async def precept_anthropic_client(
        system_prompt: str,
        user_prompt: str,
        response_model: Optional[Type[T]] = None,
    ) -> Any:
        """
        PRECEPT LLM client using Anthropic Claude API.

        Note: Structured output may not be supported depending on implementation.
        """
        result = await anthropic_model_chat_async(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
        )
        return result

    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False
    precept_anthropic_client = None


try:
    from models.gemini_api import gemini_model_chat_async

    async def precept_gemini_client(
        system_prompt: str,
        user_prompt: str,
        response_model: Optional[Type[T]] = None,
    ) -> Any:
        """
        PRECEPT LLM client using Google Gemini API.
        """
        result = await gemini_model_chat_async(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
        )
        return result

    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False
    precept_gemini_client = None


try:
    from models.groq_api import (
        groq_api_model_chat_async,
        groq_api_model_structured_output_async,
    )

    async def precept_groq_client(
        system_prompt: str,
        user_prompt: str,
        response_model: Optional[Type[T]] = None,
    ) -> Any:
        """
        PRECEPT LLM client using Groq API.

        Groq provides fast inference with Llama, Mixtral, and other models.
        Supports structured output.
        """
        if response_model is not None:
            result = await groq_api_model_structured_output_async(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                response_format=response_model,
            )
            return result
        else:
            result = await groq_api_model_chat_async(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
            )
            if hasattr(result, 'content'):
                return result.content
            return str(result)

    GROQ_AVAILABLE = True
except ImportError:
    GROQ_AVAILABLE = False
    precept_groq_client = None


try:
    from models.together_ai_api_models import (
        together_ai_api_model_chat_async,
        together_ai_api_model_structured_output_async,
    )

    async def precept_together_client(
        system_prompt: str,
        user_prompt: str,
        response_model: Optional[Type[T]] = None,
    ) -> Any:
        """
        PRECEPT LLM client using Together AI API.

        Together AI provides access to various open-source models.
        Supports structured output.
        """
        if response_model is not None:
            result = await together_ai_api_model_structured_output_async(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                response_format=response_model,
            )
            return result
        else:
            result = await together_ai_api_model_chat_async(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
            )
            if hasattr(result, 'content'):
                return result.content
            return str(result)

    TOGETHER_AVAILABLE = True
except ImportError:
    TOGETHER_AVAILABLE = False
    precept_together_client = None


# =============================================================================
# AVAILABILITY CHECK
# =============================================================================

def check_api_availability() -> dict:
    """
    Check which API providers are available and configured.

    Returns:
        Dict with availability status for each provider
    """
    import os

    return {
        "openai": _COMPASS_MODELS_AVAILABLE and bool(os.getenv("OPENAI_API_KEY")),
        "anthropic": ANTHROPIC_AVAILABLE and bool(os.getenv("ANTHROPIC_API_KEY")),
        "gemini": GEMINI_AVAILABLE and bool(os.getenv("GOOGLE_API_KEY")),
        "groq": GROQ_AVAILABLE and bool(os.getenv("GROQ_API_KEY")),
        "together": TOGETHER_AVAILABLE and bool(os.getenv("TOGETHER_API_KEY")),
        "compass_dependencies": _COMPASS_MODELS_AVAILABLE,
        "import_error": _IMPORT_ERROR,
    }


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    # Primary clients (OpenAI)
    "precept_llm_client",
    "precept_llm_client_with_context",
    "precept_embedding_fn",
    "precept_embed_documents",
    # LangChain-compatible embedding object (for vector stores like ChromaDB)
    "get_openai_embedding_model",
    # Factory functions
    "get_precept_llm_client",
    "get_precept_embedding_fn",
    "create_openai_embeddings",
    # Alternative providers
    "precept_anthropic_client",
    "precept_gemini_client",
    "precept_groq_client",
    "precept_together_client",
    # Utilities
    "check_api_availability",
    # Availability flags
    "ANTHROPIC_AVAILABLE",
    "GEMINI_AVAILABLE",
    "GROQ_AVAILABLE",
    "TOGETHER_AVAILABLE",
]
