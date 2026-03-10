"""
OpenAI Embedding Models with Lazy Initialization.

Environment Variables:
- OPENAI_API_KEY: Required for all OpenAI operations
- OPENAI_EMBEDDING_MODEL: Default embedding model (default: text-embedding-3-small)
"""

import os
from typing import List, Optional

# Lazy-initialized embedding model
_openai_embedding = None


def _check_api_key():
    """Check if OpenAI API key is available."""
    if not os.environ.get("OPENAI_API_KEY"):
        raise ValueError(
            "OPENAI_API_KEY environment variable is required. "
            "Set it in your .env file or export it."
        )


def get_openai_embedding_model(model_name: Optional[str] = None):
    """
    Get or create the LangChain OpenAI embedding model.

    Args:
        model_name: Optional model override (defaults to OPENAI_EMBEDDING_MODEL env var)

    Returns:
        LangChain OpenAIEmbeddings instance
    """
    global _openai_embedding

    model = model_name or os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")

    # Only create new instance if model changed or not initialized
    if _openai_embedding is None:
        _check_api_key()
        from langchain_openai import OpenAIEmbeddings

        _openai_embedding = OpenAIEmbeddings(model=model)

    return _openai_embedding


# Note: Use get_openai_embedding_model() for lazy initialization
# The openai_embedding variable is kept for compatibility but is dynamically accessed


def embed_text(text: str, model_name: Optional[str] = None) -> List[float]:
    """
    Embed a single text string.

    Args:
        text: Text to embed
        model_name: Optional model override

    Returns:
        Embedding vector as list of floats
    """
    model = get_openai_embedding_model(model_name)
    return model.embed_query(text)


def embed_texts(
    texts: List[str], model_name: Optional[str] = None
) -> List[List[float]]:
    """
    Embed multiple text strings.

    Args:
        texts: List of texts to embed
        model_name: Optional model override

    Returns:
        List of embedding vectors
    """
    model = get_openai_embedding_model(model_name)
    return model.embed_documents(texts)
