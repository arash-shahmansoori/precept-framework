"""
OpenAI API Wrappers with Lazy Initialization.

All model names are configured via environment variables with sensible defaults.
Clients are initialized lazily to avoid import-time failures.

IMPORTANT: Async clients are created per-event-loop to avoid race conditions
when running parallel experiments. Each event loop gets its own client instance.

Environment Variables:
- OPENAI_API_KEY: Required for all OpenAI operations
- OPENAI_CHAT_MODEL: Default chat model (default: gpt-4o-mini)
- OPENAI_STRUCTURED_MODEL: Model for structured output (default: gpt-4o-mini)
- OPENAI_REASONING_MODEL: Model for reasoning tasks (default: gpt-4o-mini)
"""

import asyncio
import os
import threading
from typing import Any, Dict, Iterator, List, Optional

# Lazy-initialized sync client (thread-safe, shared across event loops)
_openai_client = None
_openai_client_lock = threading.Lock()

# Per-event-loop async clients (to prevent event loop binding issues)
_async_clients: dict = {}  # Maps event loop id -> AsyncOpenAI client
_async_client_lock = threading.Lock()

# Chat model (sync, thread-safe)
_chat_model = None
_chat_model_lock = threading.Lock()


def _check_api_key():
    """Check if OpenAI API key is available."""
    if not os.environ.get("OPENAI_API_KEY"):
        raise ValueError(
            "OPENAI_API_KEY environment variable is required. "
            "Set it in your .env file or export it."
        )


def _get_openai_client():
    """Get or create the OpenAI client (lazy initialization, thread-safe)."""
    global _openai_client
    if _openai_client is None:
        with _openai_client_lock:
            if _openai_client is None:
                _check_api_key()
                from openai import OpenAI

                _openai_client = OpenAI()
    return _openai_client


def _get_async_client():
    """
    Get or create the async OpenAI client for the CURRENT event loop.

    CRITICAL: AsyncOpenAI clients bind internal resources (locks, connections)
    to the event loop they're created in. When running parallel experiments,
    each experiment runs in its own process/event loop. If we reuse a client
    created in a different (possibly closed) event loop, we get:
    - RuntimeError: Event loop is closed
    - RuntimeError: Task attached to a different event loop

    Solution: Create a new client for each event loop and track by loop id.
    Old clients are cleaned up when their loops close.
    """
    global _async_clients

    _check_api_key()

    try:
        # Get the current running event loop
        loop = asyncio.get_running_loop()
        loop_id = id(loop)
    except RuntimeError:
        # No running loop - create a fresh client (caller will handle loop)
        from openai import AsyncOpenAI

        return AsyncOpenAI()

    # Check if we have a client for this loop
    with _async_client_lock:
        if loop_id in _async_clients:
            return _async_clients[loop_id]

        # Clean up clients for closed loops
        closed_loops = []
        for cached_loop_id in list(_async_clients.keys()):
            # If we can't find the loop or it's closed, mark for cleanup
            if cached_loop_id != loop_id:
                closed_loops.append(cached_loop_id)

        for closed_id in closed_loops:
            # Silently remove - the loop is closed so we can't await close()
            _async_clients.pop(closed_id, None)

        # Create new client for current loop
        from openai import AsyncOpenAI

        client = AsyncOpenAI()
        _async_clients[loop_id] = client
        return client


def reset_async_client():
    """
    Force reset of async client cache. Call this between experiment runs
    in the same process to ensure clean state.
    """
    global _async_clients
    with _async_client_lock:
        _async_clients.clear()


def _get_chat_model():
    """Get or create the LangChain ChatOpenAI model (lazy initialization, thread-safe)."""
    global _chat_model
    if _chat_model is None:
        with _chat_model_lock:
            if _chat_model is None:
                _check_api_key()
                from langchain_openai import ChatOpenAI

                model_name = os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini")
                _chat_model = ChatOpenAI(model=model_name, temperature=0)
    return _chat_model


def openai_model_chat(
    messages: List[Dict[str, str]],
    model_name: Optional[str] = None,
) -> str:
    """
    Synchronous chat with OpenAI model.

    Args:
        messages: List of message dicts with 'role' and 'content'
        model_name: Optional model override (defaults to OPENAI_CHAT_MODEL env var)

    Returns:
        Model response as string
    """
    client = _get_openai_client()
    model = model_name or os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini")

    response = client.chat.completions.create(
        model=model,
        messages=messages,
    )
    return response.choices[0].message.content


async def openai_model_chat_async(
    messages: List[Dict[str, str]],
    model_name: Optional[str] = None,
) -> str:
    """
    Asynchronous chat with OpenAI model.

    Args:
        messages: List of message dicts with 'role' and 'content'
        model_name: Optional model override

    Returns:
        Model response as string
    """
    client = _get_async_client()
    model = model_name or os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini")

    response = await client.chat.completions.create(
        model=model,
        messages=messages,
    )
    return response.choices[0].message.content


def openai_model_chat_stream(
    messages: List[Dict[str, str]],
    model_name: Optional[str] = None,
) -> Iterator[str]:
    """
    Streaming chat with OpenAI model.

    Args:
        messages: List of message dicts with 'role' and 'content'
        model_name: Optional model override

    Yields:
        Chunks of model response
    """
    client = _get_openai_client()
    model = model_name or os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini")

    response = client.chat.completions.create(
        model=model,
        messages=messages,
        stream=True,
    )

    for chunk in response:
        if chunk.choices[0].delta.content:
            yield chunk.choices[0].delta.content


def openai_model_chat_structured_output(
    messages: List[Dict[str, str]],
    response_format: Any,
    model_name: Optional[str] = None,
) -> Any:
    """
    Chat with structured output using OpenAI's response_format.

    Args:
        messages: List of message dicts with 'role' and 'content'
        response_format: Pydantic model or JSON schema for structured output
        model_name: Optional model override

    Returns:
        Parsed structured response
    """
    client = _get_openai_client()
    model = model_name or os.getenv("OPENAI_STRUCTURED_MODEL", "gpt-4o-mini")

    response = client.beta.chat.completions.parse(
        model=model,
        messages=messages,
        response_format=response_format,
    )
    return response.choices[0].message.parsed


def openai_reasoning_structured_output(
    messages: List[Dict[str, str]],
    response_format: Any,
    model_name: Optional[str] = None,
) -> Any:
    """
    Reasoning-focused chat with structured output.

    Uses a reasoning-optimized model for complex tasks.

    Args:
        messages: List of message dicts with 'role' and 'content'
        response_format: Pydantic model or JSON schema for structured output
        model_name: Optional model override

    Returns:
        Parsed structured response
    """
    client = _get_openai_client()
    model = model_name or os.getenv("OPENAI_REASONING_MODEL", "gpt-4o-mini")

    # For reasoning models, use 'user' role for instructions
    formatted_messages = []
    for msg in messages:
        role = msg.get("role", "user")
        # Convert 'developer' or 'system' to 'user' for reasoning models
        if role in ("developer", "system"):
            role = "user"
        formatted_messages.append({"role": role, "content": msg["content"]})

    response = client.beta.chat.completions.parse(
        model=model,
        messages=formatted_messages,
        response_format=response_format,
    )
    return response.choices[0].message.parsed


async def openai_model_structured_output_async(
    system_prompt: str,
    user_prompt: str,
    response_format: Any,
    model_name: Optional[str] = None,
) -> Any:
    """
    Async structured output using OpenAI's response_format.

    Args:
        system_prompt: System instructions for the LLM
        user_prompt: User query/prompt
        response_format: Pydantic model or JSON schema for structured output
        model_name: Optional model override

    Returns:
        Parsed structured response
    """
    _check_api_key()
    client = _get_async_client()
    model = model_name or os.getenv("OPENAI_STRUCTURED_MODEL", "gpt-4o-mini")

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    response = await client.beta.chat.completions.parse(
        model=model,
        messages=messages,
        response_format=response_format,
    )
    return response.choices[0].message.parsed


async def openai_model_context_structured_output_async(
    system_prompt: str,
    user_prompt: str,
    context: str,
    response_format: Any,
    model_name: Optional[str] = None,
) -> Any:
    """
    Async structured output with additional context.

    Args:
        system_prompt: System instructions for the LLM
        user_prompt: User query/prompt
        context: Additional context to include
        response_format: Pydantic model or JSON schema for structured output
        model_name: Optional model override

    Returns:
        Parsed structured response
    """
    _check_api_key()
    client = _get_async_client()
    model = model_name or os.getenv("OPENAI_STRUCTURED_MODEL", "gpt-4o-mini")

    full_prompt = f"{user_prompt}\n\nContext:\n{context}" if context else user_prompt

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": full_prompt},
    ]

    response = await client.beta.chat.completions.parse(
        model=model,
        messages=messages,
        response_format=response_format,
    )
    return response.choices[0].message.parsed
