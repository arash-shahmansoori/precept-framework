"""
Groq API Wrappers with Lazy Initialization.

Groq provides fast inference for open-source models.

Environment Variables:
- GROQ_API_KEY: Required for all Groq operations
- GROQ_MODEL: Default model (default: llama-3.3-70b-versatile)
- GROQ_STRUCTURED_MODEL: Model for structured output (default: llama-3.3-70b-versatile)
"""

import os
from typing import Any, Iterator, List, Optional

# Lazy-initialized models
_chat_model = None
_chat_model_structured = None


def _check_api_key():
    """Check if Groq API key is available."""
    if not os.environ.get("GROQ_API_KEY"):
        raise ValueError(
            "GROQ_API_KEY environment variable is required. "
            "Set it in your .env file or export it."
        )


def _get_chat_model(model_name: Optional[str] = None):
    """Get or create the LangChain ChatGroq model."""
    global _chat_model
    if _chat_model is None:
        _check_api_key()
        from langchain_groq import ChatGroq
        model = model_name or os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
        _chat_model = ChatGroq(model=model, temperature=0)
    return _chat_model


def _get_chat_model_structured(model_name: Optional[str] = None):
    """Get or create ChatGroq for structured output."""
    global _chat_model_structured
    if _chat_model_structured is None:
        _check_api_key()
        from langchain_groq import ChatGroq
        model = model_name or os.getenv("GROQ_STRUCTURED_MODEL", "llama-3.3-70b-versatile")
        _chat_model_structured = ChatGroq(model=model, temperature=0)
    return _chat_model_structured


def groq_api_model_chat(
    messages: List[dict],
    model_name: Optional[str] = None,
) -> str:
    """
    Synchronous chat with Groq model.

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


async def groq_api_model_chat_async(
    messages: List[dict],
    model_name: Optional[str] = None,
) -> str:
    """
    Asynchronous chat with Groq model.

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


def groq_api_model_chat_stream(
    messages: List[dict],
    model_name: Optional[str] = None,
) -> Iterator[str]:
    """
    Streaming chat with Groq model.

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


def groq_api_model_chat_structured_output(
    messages: List[dict],
    response_format: Any,
    model_name: Optional[str] = None,
) -> Any:
    """
    Chat with structured output using Groq.

    Args:
        messages: List of message dicts with 'role' and 'content'
        response_format: Pydantic model for structured output
        model_name: Optional model override

    Returns:
        Parsed structured response
    """
    chat_model = _get_chat_model_structured(model_name)
    structured_model = chat_model.with_structured_output(response_format)

    from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

    lc_messages = []
    for msg in messages:
        if msg["role"] == "system":
            lc_messages.append(SystemMessage(content=msg["content"]))
        elif msg["role"] == "user":
            lc_messages.append(HumanMessage(content=msg["content"]))
        elif msg["role"] == "assistant":
            lc_messages.append(AIMessage(content=msg["content"]))

    return structured_model.invoke(lc_messages)
