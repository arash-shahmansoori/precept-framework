"""
LLM and Embedding Model Providers.

This module provides unified interfaces to various LLM providers:
- OpenAI (GPT-4, GPT-4o, embeddings)
- Anthropic (Claude)
- Google (Gemini)
- Groq (fast inference)
- Together AI (open source models)

All clients use lazy initialization and environment variables for configuration.
"""

from .anthropic_api import (
    anthropic_model_chat,
    anthropic_model_chat_async,
    anthropic_model_chat_with_prompt_caching,
)
from .embedding_models import (
    embed_text,
    embed_texts,
    get_openai_embedding_model,
)
from .gemini_api import (
    gemini_model_chat,
    gemini_model_chat_async,
)
from .groq_api import (
    groq_api_model_chat,
    groq_api_model_chat_async,
    groq_api_model_chat_structured_output,
)
from .openai_api import (
    openai_model_chat,
    openai_model_chat_async,
    openai_model_chat_structured_output,
    openai_reasoning_structured_output,
)
from .together_ai_api_models import (
    together_ai_api_model_chat,
    together_ai_api_model_chat_async,
    together_ai_api_model_chat_structured_output,
)

__all__ = [
    # OpenAI
    "openai_model_chat",
    "openai_model_chat_async",
    "openai_model_chat_structured_output",
    "openai_reasoning_structured_output",
    # Embeddings
    "get_openai_embedding_model",
    "embed_text",
    "embed_texts",
    # Anthropic
    "anthropic_model_chat",
    "anthropic_model_chat_async",
    "anthropic_model_chat_with_prompt_caching",
    # Gemini
    "gemini_model_chat",
    "gemini_model_chat_async",
    # Groq
    "groq_api_model_chat",
    "groq_api_model_chat_async",
    "groq_api_model_chat_structured_output",
    # Together AI
    "together_ai_api_model_chat",
    "together_ai_api_model_chat_async",
    "together_ai_api_model_chat_structured_output",
]
