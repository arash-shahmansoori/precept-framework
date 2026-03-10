"""
Google Gemini API Wrappers with Lazy Initialization.

Environment Variables:
- GOOGLE_API_KEY: Required for all Gemini operations
- GEMINI_MODEL: Default model (default: gemini-1.5-flash)
"""

import os
from typing import Iterator, List, Optional

# Lazy-initialized model
_chat_model = None


def _check_api_key():
    """Check if Google API key is available."""
    if not os.environ.get("GOOGLE_API_KEY"):
        raise ValueError(
            "GOOGLE_API_KEY environment variable is required. "
            "Set it in your .env file or export it."
        )


def _get_chat_model(model_name: Optional[str] = None):
    """Get or create the LangChain ChatGoogleGenerativeAI model."""
    global _chat_model
    if _chat_model is None:
        _check_api_key()
        from langchain_google_genai import ChatGoogleGenerativeAI
        model = model_name or os.getenv("GEMINI_MODEL", "gemini-1.5-flash")
        _chat_model = ChatGoogleGenerativeAI(model=model, temperature=0)
    return _chat_model


def gemini_model_chat(
    messages: List[dict],
    model_name: Optional[str] = None,
) -> str:
    """
    Synchronous chat with Gemini model.

    Args:
        messages: List of message dicts with 'role' and 'content'
        model_name: Optional model override

    Returns:
        Model response as string
    """
    chat_model = _get_chat_model(model_name)

    from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

    lc_messages = []
    for msg in messages:
        if msg["role"] == "system":
            lc_messages.append(SystemMessage(content=msg["content"]))
        elif msg["role"] == "user":
            lc_messages.append(HumanMessage(content=msg["content"]))
        elif msg["role"] == "assistant":
            lc_messages.append(AIMessage(content=msg["content"]))

    response = chat_model.invoke(lc_messages)
    return response.content


async def gemini_model_chat_async(
    messages: List[dict],
    model_name: Optional[str] = None,
) -> str:
    """
    Asynchronous chat with Gemini model.

    Args:
        messages: List of message dicts with 'role' and 'content'
        model_name: Optional model override

    Returns:
        Model response as string
    """
    chat_model = _get_chat_model(model_name)

    from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

    lc_messages = []
    for msg in messages:
        if msg["role"] == "system":
            lc_messages.append(SystemMessage(content=msg["content"]))
        elif msg["role"] == "user":
            lc_messages.append(HumanMessage(content=msg["content"]))
        elif msg["role"] == "assistant":
            lc_messages.append(AIMessage(content=msg["content"]))

    response = await chat_model.ainvoke(lc_messages)
    return response.content


def gemini_model_chat_stream(
    messages: List[dict],
    model_name: Optional[str] = None,
) -> Iterator[str]:
    """
    Streaming chat with Gemini model.

    Args:
        messages: List of message dicts with 'role' and 'content'
        model_name: Optional model override

    Yields:
        Chunks of model response
    """
    chat_model = _get_chat_model(model_name)

    from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

    lc_messages = []
    for msg in messages:
        if msg["role"] == "system":
            lc_messages.append(SystemMessage(content=msg["content"]))
        elif msg["role"] == "user":
            lc_messages.append(HumanMessage(content=msg["content"]))
        elif msg["role"] == "assistant":
            lc_messages.append(AIMessage(content=msg["content"]))

    for chunk in chat_model.stream(lc_messages):
        if chunk.content:
            yield chunk.content
