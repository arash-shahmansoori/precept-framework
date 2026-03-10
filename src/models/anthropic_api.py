"""
Anthropic (Claude) API Wrappers with Lazy Initialization.

Environment Variables:
- ANTHROPIC_API_KEY: Required for all Anthropic operations
- ANTHROPIC_MODEL: Default model (default: claude-sonnet-4-20250514)
"""

import os
from typing import Iterator, List, Optional

# Lazy-initialized clients
_anthropic_client = None
_chat_model = None
_chat_model_caching = None


def _check_api_key():
    """Check if Anthropic API key is available."""
    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise ValueError(
            "ANTHROPIC_API_KEY environment variable is required. "
            "Set it in your .env file or export it."
        )


def _get_anthropic_client():
    """Get or create the Anthropic client (lazy initialization)."""
    global _anthropic_client
    if _anthropic_client is None:
        _check_api_key()
        from anthropic import Anthropic

        _anthropic_client = Anthropic()
    return _anthropic_client


def _get_chat_model(model_name: Optional[str] = None):
    """Get or create the LangChain ChatAnthropic model (lazy initialization)."""
    global _chat_model
    if _chat_model is None:
        _check_api_key()
        from langchain_anthropic import ChatAnthropic

        model = model_name or os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-20250514")
        _chat_model = ChatAnthropic(model=model, temperature=0)
    return _chat_model


def _get_chat_model_caching(model_name: Optional[str] = None):
    """Get or create ChatAnthropic with prompt caching enabled."""
    global _chat_model_caching
    if _chat_model_caching is None:
        _check_api_key()
        from langchain_anthropic import ChatAnthropic

        model = model_name or os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-20250514")
        _chat_model_caching = ChatAnthropic(
            model=model,
            temperature=0,
            extra_headers={"anthropic-beta": "prompt-caching-2024-07-31"},
        )
    return _chat_model_caching


def anthropic_model_chat(
    messages: List[dict],
    model_name: Optional[str] = None,
) -> str:
    """
    Synchronous chat with Anthropic model.

    Args:
        messages: List of message dicts with 'role' and 'content'
        model_name: Optional model override

    Returns:
        Model response as string
    """
    client = _get_anthropic_client()
    model = model_name or os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-20250514")

    # Convert messages format
    system_msg = ""
    anthropic_messages = []
    for msg in messages:
        if msg["role"] == "system":
            system_msg = msg["content"]
        else:
            anthropic_messages.append(msg)

    response = client.messages.create(
        model=model,
        max_tokens=4096,
        system=system_msg if system_msg else None,
        messages=anthropic_messages,
    )
    return response.content[0].text


async def anthropic_model_chat_async(
    messages: List[dict],
    model_name: Optional[str] = None,
) -> str:
    """
    Asynchronous chat with Anthropic model using LangChain.

    Args:
        messages: List of message dicts with 'role' and 'content'
        model_name: Optional model override

    Returns:
        Model response as string
    """
    chat_model = _get_chat_model(model_name)

    # Convert to LangChain message format
    from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

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


def anthropic_model_chat_stream(
    messages: List[dict],
    model_name: Optional[str] = None,
) -> Iterator[str]:
    """
    Streaming chat with Anthropic model.

    Args:
        messages: List of message dicts with 'role' and 'content'
        model_name: Optional model override

    Yields:
        Chunks of model response
    """
    client = _get_anthropic_client()
    model = model_name or os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-20250514")

    # Convert messages format
    system_msg = ""
    anthropic_messages = []
    for msg in messages:
        if msg["role"] == "system":
            system_msg = msg["content"]
        else:
            anthropic_messages.append(msg)

    with client.messages.stream(
        model=model,
        max_tokens=4096,
        system=system_msg if system_msg else None,
        messages=anthropic_messages,
    ) as stream:
        for text in stream.text_stream:
            yield text


def anthropic_model_chat_with_prompt_caching(
    messages: List[dict],
    model_name: Optional[str] = None,
) -> str:
    """
    Chat with Anthropic using prompt caching for efficiency.

    Prompt caching reduces latency and cost for repeated prompts.

    Args:
        messages: List of message dicts with 'role' and 'content'
        model_name: Optional model override

    Returns:
        Model response as string
    """
    chat_model = _get_chat_model_caching(model_name)

    from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

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
