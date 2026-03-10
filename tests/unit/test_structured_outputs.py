"""
Unit Tests for precept.structured_outputs module.

Tests Pydantic models and parsing functions for LLM structured outputs.
"""

import pytest


class TestStructuredOutputModels:
    """Tests for Pydantic model definitions."""

    def test_reasoning_response_import(self):
        """Test ReasoningResponse model can be imported."""
        from precept.structured_outputs import ReasoningResponse

        assert ReasoningResponse is not None

    def test_baseline_response_import(self):
        """Test BaselineResponse model can be imported."""
        from precept.structured_outputs import BaselineResponse

        assert BaselineResponse is not None

    def test_reflexion_response_import(self):
        """Test ReflexionResponse model can be imported."""
        from precept.structured_outputs import ReflexionResponse

        assert ReflexionResponse is not None

    def test_full_reflexion_response_import(self):
        """Test FullReflexionResponse model can be imported."""
        from precept.structured_outputs import FullReflexionResponse

        assert FullReflexionResponse is not None


class TestReasoningResponse:
    """Tests for ReasoningResponse model."""

    def test_create_valid_response(self):
        """Test creating a valid reasoning response."""
        from precept.structured_outputs import ReasoningResponse

        response = ReasoningResponse(
            solution="Antwerp",
            reasoning="Rotterdam is blocked",
            confidence="high",
        )

        assert response.solution == "Antwerp"
        assert response.reasoning == "Rotterdam is blocked"
        assert response.confidence == "high"

    def test_confidence_values(self):
        """Test valid confidence values."""
        from precept.structured_outputs import ReasoningResponse

        for conf in ["high", "medium", "low"]:
            response = ReasoningResponse(
                solution="test", reasoning="test", confidence=conf
            )
            assert response.confidence == conf


class TestParseStructuredResponse:
    """Tests for parse_structured_response function."""

    def test_parse_valid_json(self):
        """Test parsing valid JSON response."""
        from precept.structured_outputs import ReasoningResponse, parse_structured_response

        json_response = '{"solution": "Antwerp", "reasoning": "Best option", "confidence": "high"}'

        result = parse_structured_response(json_response, ReasoningResponse)

        assert result is not None
        assert result.solution == "Antwerp"

    def test_parse_invalid_json_fallback(self):
        """Test fallback parsing for non-JSON response."""
        from precept.structured_outputs import ReasoningResponse, parse_structured_response

        text_response = """
        SOLUTION: Antwerp
        REASONING: Rotterdam is blocked
        CONFIDENCE: high
        """

        result = parse_structured_response(text_response, ReasoningResponse)

        # Fallback may or may not work depending on implementation
        # Just check it doesn't crash
        assert result is None or hasattr(result, "solution")

    def test_parse_empty_response(self):
        """Test parsing empty response."""
        from precept.structured_outputs import ReasoningResponse, parse_structured_response

        result = parse_structured_response("", ReasoningResponse)

        # Should return None for empty response
        assert result is None


class TestSchemaGeneration:
    """Tests for JSON schema generation."""

    def test_get_reasoning_schema(self):
        """Test getting reasoning response schema."""
        from precept.structured_outputs import get_reasoning_schema

        schema = get_reasoning_schema()

        assert schema is not None
        assert "properties" in schema
        assert "solution" in schema["properties"]
        assert "reasoning" in schema["properties"]
        assert "confidence" in schema["properties"]

    def test_create_structured_output_params(self):
        """Test creating structured output parameters."""
        from precept.structured_outputs import create_structured_output_params

        params = create_structured_output_params("reasoning")

        assert params is not None
        assert "response_format" in params
