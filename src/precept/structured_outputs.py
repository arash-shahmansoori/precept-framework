"""
Structured Output Definitions for PRECEPT.

Provides Pydantic models and JSON schemas for reliable LLM response parsing.
Uses OpenAI's structured output features for guaranteed format compliance.

Usage:
    from precept.structured_outputs import (
        ReasoningResponse,
        BaselineResponse,
        ReflexionResponse,
        parse_structured_response,
    )

    # Parse with guaranteed structure
    response = parse_structured_response(llm_output, ReasoningResponse)
"""

from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Literal, Optional, Union

try:
    from pydantic import BaseModel, Field

    PYDANTIC_AVAILABLE = True
except ImportError:
    PYDANTIC_AVAILABLE = False
    BaseModel = object  # type: ignore
    Field = lambda **kwargs: None  # type: ignore


# =============================================================================
# CONFIDENCE LEVELS
# =============================================================================


class ConfidenceLevel(str, Enum):
    """Confidence levels for LLM suggestions."""

    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


# =============================================================================
# PYDANTIC MODELS FOR STRUCTURED OUTPUTS
# =============================================================================


if PYDANTIC_AVAILABLE:

    class TaskParseResponse(BaseModel):
        """
        Structured response for LLM-assisted task parsing.

        Used when rule-based parsing fails or for complex tasks.
        OpenAI's structured outputs guarantee these fields exist.
        """

        action: str = Field(
            ...,
            description="The primary action to perform (e.g., 'book_shipment', 'deploy', 'trade')",
        )
        entity: str = Field(
            ...,
            description="The main entity involved (e.g., port name, flight ID, package name)",
        )
        source: Optional[str] = Field(
            None,
            description="Source/origin for the action (e.g., origin port, source system)",
        )
        target: Optional[str] = Field(
            None,
            description="Target/destination for the action (e.g., destination port, target env)",
        )
        task_type: str = Field(
            ...,
            description="Category of task (e.g., 'shipment', 'booking', 'deployment')",
        )
        parameters: Optional[Dict[str, Any]] = Field(
            default_factory=dict,
            description="Additional parameters extracted from the task",
        )
        confidence: ConfidenceLevel = Field(
            default=ConfidenceLevel.MEDIUM,
            description="How confident is the parse? high=clear task, low=ambiguous",
        )
        parsing_notes: Optional[str] = Field(
            None,
            description="Any notes about ambiguity or assumptions made during parsing",
        )

        class Config:
            extra = "forbid"

    class ReasoningResponse(BaseModel):
        """Structured response for PRECEPT reasoning."""

        solution: str = Field(
            ...,
            description="The suggested solution, or 'EXPLORE' if unsure, or 'EXHAUSTED' if all options forbidden",
        )
        reasoning: str = Field(
            ...,
            description="One sentence explaining which source (rule/memory/exploration) informed the choice",
        )
        confidence: ConfidenceLevel = Field(
            ...,
            description="Confidence level: high (rule match), medium (memory pattern), low (exploration)",
        )

        class Config:
            """Pydantic config for strict mode."""

            extra = "forbid"  # Reject unknown fields

    class BaselineResponse(BaseModel):
        """Structured response for baseline LLM reasoning."""

        solution: str = Field(
            ...,
            description="The suggested option from available options",
        )
        reasoning: str = Field(
            ...,
            description="Brief explanation for the choice",
        )
        confidence: ConfidenceLevel = Field(
            ...,
            description="Confidence level: high, medium, or low",
        )

        class Config:
            extra = "forbid"

    class ReflexionResponse(BaseModel):
        """Structured response for Reflexion-style reasoning."""

        reflection: Optional[str] = Field(
            None,
            description="What went wrong? Why did the previous option fail?",
        )
        lesson: Optional[str] = Field(
            None,
            description="What should be done differently this time?",
        )
        solution: str = Field(
            ...,
            description="The suggested option from available options",
        )
        reasoning: str = Field(
            ...,
            description="Why this option should work based on reflection",
        )
        confidence: ConfidenceLevel = Field(
            ...,
            description="Confidence level: high, medium, or low",
        )

        class Config:
            extra = "forbid"

    class FullReflexionResponse(BaseModel):
        """Structured response for Full Reflexion with cross-episode memory."""

        reflection: str = Field(
            ...,
            description="What patterns do you notice? What should you avoid?",
        )
        lesson: str = Field(
            ...,
            description="Key insight for this and future episodes",
        )
        solution: str = Field(
            ...,
            description="The suggested option from available options",
        )
        reasoning: str = Field(
            ...,
            description="Why this option, based on reflections",
        )
        confidence: ConfidenceLevel = Field(
            ...,
            description="Confidence level: high, medium, or low",
        )

        class Config:
            extra = "forbid"

    class ExpeL_InsightResponse(BaseModel):
        """Structured response for ExpeL insight extraction."""

        insight: str = Field(
            ...,
            description="The generalizable pattern or rule learned from this experience",
        )
        conditions_covered: List[str] = Field(
            ...,
            description="List of condition codes this insight applies to (e.g., ['R-482', 'C-HZMT'])",
        )
        solution: Optional[str] = Field(
            None,
            description="The working solution for success insights (e.g., 'hamburg')",
        )
        avoid: List[str] = Field(
            default_factory=list,
            description="Options to avoid for failure insights (e.g., ['london', 'singapore'])",
        )
        confidence: ConfidenceLevel = Field(
            ...,
            description="How generalizable is this insight: high, medium, or low",
        )

        class Config:
            extra = "forbid"

    class ExpeL_TaskResponse(BaseModel):
        """Structured response for ExpeL task execution."""

        solution: str = Field(
            ...,
            description="The suggested option from available options",
        )
        insight_applied: Optional[str] = Field(
            None,
            description="Which insight (if any) informed this decision",
        )
        reasoning: str = Field(
            ...,
            description="Why this solution was chosen based on insights",
        )
        confidence: ConfidenceLevel = Field(
            ...,
            description="Confidence level: high, medium, or low",
        )

        class Config:
            extra = "forbid"

else:
    # Fallback dataclasses when Pydantic not available

    @dataclass
    class ReasoningResponse:  # type: ignore
        solution: str
        reasoning: str
        confidence: str

    @dataclass
    class BaselineResponse:  # type: ignore
        solution: str
        reasoning: str
        confidence: str

    @dataclass
    class ReflexionResponse:  # type: ignore
        reflection: Optional[str]
        lesson: Optional[str]
        solution: str
        reasoning: str
        confidence: str

    @dataclass
    class FullReflexionResponse:  # type: ignore
        reflection: str
        lesson: str
        solution: str
        reasoning: str
        confidence: str

    @dataclass
    class ExpeL_InsightResponse:  # type: ignore
        insight: str
        conditions_covered: List[str]
        solution: Optional[str]
        avoid: List[str]
        confidence: str

    @dataclass
    class ExpeL_TaskResponse:  # type: ignore
        solution: str
        insight_applied: Optional[str]
        reasoning: str
        confidence: str


# =============================================================================
# JSON SCHEMAS FOR OPENAI STRUCTURED OUTPUTS
# =============================================================================


def get_reasoning_schema() -> Dict[str, Any]:
    """Get JSON schema for reasoning response (OpenAI structured outputs)."""
    return {
        "type": "object",
        "properties": {
            "solution": {
                "type": "string",
                "description": "The suggested solution, 'EXPLORE' if unsure, or 'EXHAUSTED' if all options forbidden",
            },
            "reasoning": {
                "type": "string",
                "description": "One sentence explaining the decision source",
            },
            "confidence": {
                "type": "string",
                "enum": ["high", "medium", "low"],
                "description": "Confidence level",
            },
        },
        "required": ["solution", "reasoning", "confidence"],
        "additionalProperties": False,
    }


def get_baseline_schema() -> Dict[str, Any]:
    """Get JSON schema for baseline response."""
    return {
        "type": "object",
        "properties": {
            "solution": {
                "type": "string",
                "description": "The suggested option",
            },
            "reasoning": {
                "type": "string",
                "description": "Brief explanation",
            },
            "confidence": {
                "type": "string",
                "enum": ["high", "medium", "low"],
            },
        },
        "required": ["solution", "reasoning", "confidence"],
        "additionalProperties": False,
    }


def get_reflexion_schema() -> Dict[str, Any]:
    """Get JSON schema for reflexion response."""
    return {
        "type": "object",
        "properties": {
            "reflection": {
                "type": ["string", "null"],
                "description": "What went wrong? Why did it fail? (null if first attempt)",
            },
            "lesson": {
                "type": ["string", "null"],
                "description": "What to do differently (null if first attempt)",
            },
            "solution": {
                "type": "string",
                "description": "The suggested option",
            },
            "reasoning": {
                "type": "string",
                "description": "Why this option should work",
            },
            "confidence": {
                "type": "string",
                "enum": ["high", "medium", "low"],
            },
        },
        "required": ["reflection", "lesson", "solution", "reasoning", "confidence"],
        "additionalProperties": False,
    }


def get_full_reflexion_schema() -> Dict[str, Any]:
    """Get JSON schema for Full Reflexion response."""
    return {
        "type": "object",
        "properties": {
            "reflection": {
                "type": "string",
                "description": "What patterns do you notice? What should you avoid?",
            },
            "lesson": {
                "type": "string",
                "description": "Key insight for this and future episodes",
            },
            "solution": {
                "type": "string",
                "description": "The suggested option from available options",
            },
            "reasoning": {
                "type": "string",
                "description": "Why this option, based on reflections",
            },
            "confidence": {
                "type": "string",
                "enum": ["high", "medium", "low"],
            },
        },
        "required": ["reflection", "lesson", "solution", "reasoning", "confidence"],
        "additionalProperties": False,
    }


def get_expel_insight_schema() -> Dict[str, Any]:
    """Get JSON schema for ExpeL insight extraction."""
    return {
        "type": "object",
        "properties": {
            "insight": {
                "type": "string",
                "description": "The generalizable pattern or rule learned",
            },
            "conditions_covered": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Condition codes this insight applies to",
            },
            "solution": {
                "type": ["string", "null"],
                "description": "The working solution (for success insights), or null for failures",
            },
            "avoid": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Options to avoid (for failure insights)",
            },
            "confidence": {
                "type": "string",
                "enum": ["high", "medium", "low"],
            },
        },
        "required": ["insight", "conditions_covered", "solution", "avoid", "confidence"],
        "additionalProperties": False,
    }


def get_expel_task_schema() -> Dict[str, Any]:
    """Get JSON schema for ExpeL task execution."""
    return {
        "type": "object",
        "properties": {
            "solution": {
                "type": "string",
                "description": "The suggested option from available options",
            },
            "insight_applied": {
                "type": ["string", "null"],
                "description": "Which insight informed this decision, or null if none",
            },
            "reasoning": {
                "type": "string",
                "description": "Why this solution was chosen",
            },
            "confidence": {
                "type": "string",
                "enum": ["high", "medium", "low"],
            },
        },
        "required": ["solution", "insight_applied", "reasoning", "confidence"],
        "additionalProperties": False,
    }


# =============================================================================
# PARSING FUNCTIONS
# =============================================================================


def parse_structured_response(
    response_text: str,
    response_type: type,
) -> Optional[Union[ReasoningResponse, BaselineResponse, ReflexionResponse]]:
    """
    Parse a structured response from LLM output.

    Works with both:
    1. JSON responses (from structured output mode)
    2. Text responses (fallback regex parsing)

    Args:
        response_text: Raw response from LLM
        response_type: Expected response type (Pydantic model)

    Returns:
        Parsed response object or None if parsing fails
    """
    import json
    import re

    # Try JSON parsing first (structured output mode)
    try:
        # Handle potential markdown code blocks
        json_text = response_text
        if "```json" in response_text:
            json_match = re.search(r"```json\s*(.*?)\s*```", response_text, re.DOTALL)
            if json_match:
                json_text = json_match.group(1)
        elif "```" in response_text:
            json_match = re.search(r"```\s*(.*?)\s*```", response_text, re.DOTALL)
            if json_match:
                json_text = json_match.group(1)

        data = json.loads(json_text)

        if PYDANTIC_AVAILABLE and hasattr(response_type, "model_validate"):
            return response_type.model_validate(data)
        else:
            return response_type(**data)  # type: ignore

    except (json.JSONDecodeError, Exception):
        pass

    # Fallback: Regex-based parsing (for non-structured outputs)
    result = {}

    # Extract fields using regex
    patterns = {
        "solution": r"SOLUTION[:\s]+([^\n]+)",
        "reasoning": r"REASONING[:\s]+([^\n]+)",
        "confidence": r"CONFIDENCE[:\s]+(\w+)",
        "reflection": r"REFLECTION[:\s]+(.+?)(?=LESSON:|SOLUTION:|$)",
        "lesson": r"LESSON[:\s]+(.+?)(?=SOLUTION:|$)",
    }

    for field, pattern in patterns.items():
        match = re.search(pattern, response_text, re.IGNORECASE | re.DOTALL)
        if match:
            result[field] = match.group(1).strip()

    # Validate required fields
    if "solution" not in result:
        return None

    # Normalize confidence
    if "confidence" in result:
        result["confidence"] = result["confidence"].lower()
        if result["confidence"] not in ("high", "medium", "low"):
            result["confidence"] = "medium"

    try:
        if PYDANTIC_AVAILABLE and hasattr(response_type, "model_validate"):
            return response_type.model_validate(result)
        else:
            # For dataclasses, filter to known fields
            import dataclasses

            if dataclasses.is_dataclass(response_type):
                field_names = {f.name for f in dataclasses.fields(response_type)}
                filtered = {k: v for k, v in result.items() if k in field_names}
                return response_type(**filtered)  # type: ignore
            return None
    except Exception:
        return None


def create_structured_output_params(
    schema: Dict[str, Any],
    strict: bool = True,
) -> Dict[str, Any]:
    """
    Create parameters for OpenAI structured output API call.

    Usage with OpenAI client:
        params = create_structured_output_params(get_reasoning_schema())
        response = await client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[...],
            **params,
        )

    Args:
        schema: JSON schema for the response
        strict: Whether to use strict mode (guaranteed compliance)

    Returns:
        Dict with response_format parameter
    """
    return {
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": "response",
                "strict": strict,
                "schema": schema,
            },
        },
    }


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    # Enums
    "ConfidenceLevel",
    # Response models
    "ReasoningResponse",
    "BaselineResponse",
    "ReflexionResponse",
    "FullReflexionResponse",
    "ExpeL_InsightResponse",
    "ExpeL_TaskResponse",
    "TaskParseResponse",
    # Schema functions
    "get_reasoning_schema",
    "get_baseline_schema",
    "get_reflexion_schema",
    "get_full_reflexion_schema",
    "get_expel_insight_schema",
    "get_expel_task_schema",
    # Parsing
    "parse_structured_response",
    "create_structured_output_params",
    # Constants
    "PYDANTIC_AVAILABLE",
]
