"""
Unit Tests for precept.agent_functions module.

Tests pure functions for PRECEPT agent operations including:
- LLM response parsing
- Prompt building
- Task result helpers
- Statistics computation
"""

from precept.agent_functions import (
    apply_llm_suggestion,
    apply_procedure_hint,
    build_llm_reasoning_result,
    build_reasoning_prompt,
    build_task_record,
    build_task_result,
    compute_average,
    compute_success_rate,
    compute_usefulness_feedback,
    format_error_feedback,
    format_failure_context,
    increment_counters,
    parse_llm_response,
    parse_reflexion_response,
    should_trigger_compass_evolution,
    should_trigger_consolidation,
    update_failure_counter,
    update_llm_stats,
)


class TestParseLLMResponse:
    """Tests for parse_llm_response function."""

    def test_parse_valid_response(self):
        """Test parsing a valid LLM response with solution."""
        response = """
        SOLUTION: Antwerp
        REASONING: Rotterdam is blocked
        CONFIDENCE: high
        """
        result = parse_llm_response(response)

        assert result is not None
        assert result.suggested_solution == "Antwerp"
        assert "Rotterdam" in result.reasoning
        assert result.confidence == "high"

    def test_parse_explore_response(self):
        """Test parsing EXPLORE response returns None."""
        response = "SOLUTION: EXPLORE\nREASONING: Need more info"
        result = parse_llm_response(response)

        assert result is None

    def test_parse_exhausted_response(self):
        """Test parsing EXHAUSTED response."""
        response = "SOLUTION: EXHAUSTED\nREASONING: All options tried"
        result = parse_llm_response(response)

        assert result is not None
        assert result.suggested_solution == "EXHAUSTED"

    def test_parse_empty_response(self):
        """Test parsing empty response returns None."""
        result = parse_llm_response("")
        assert result is None

        result = parse_llm_response(None)
        assert result is None

    def test_parse_case_insensitive(self):
        """Test parsing is case insensitive."""
        response = "solution: hamburg\nreasoning: best option\nconfidence: medium"
        result = parse_llm_response(response)

        assert result is not None
        assert result.suggested_solution == "hamburg"

    def test_parse_with_quotes(self):
        """Test parsing solution with quotes."""
        response = 'SOLUTION: "Antwerp"\nREASONING: Clean port'
        result = parse_llm_response(response)

        assert result is not None
        assert result.suggested_solution == "Antwerp"


class TestParseReflexionResponse:
    """Tests for parse_reflexion_response function."""

    def test_parse_full_reflexion(self):
        """Test parsing full reflexion response."""
        response = """
        REFLECTION: Rotterdam failed due to port closure
        LESSON: Use alternative ports for blocked routes
        SOLUTION: Antwerp
        REASONING: Nearest alternative
        CONFIDENCE: high
        """
        options = ["rotterdam", "hamburg", "antwerp"]
        result = parse_reflexion_response(response, options)

        assert result["reflection"] is not None
        assert "Rotterdam" in result["reflection"]
        assert result["lesson"] is not None
        assert result["solution"] == "antwerp"

    def test_parse_with_valid_option_matching(self):
        """Test that solution matches valid options."""
        response = "SOLUTION: antwerp"
        options = ["rotterdam", "hamburg", "antwerp"]
        result = parse_reflexion_response(response, options)

        assert result["solution"] == "antwerp"

    def test_fallback_to_text_search(self):
        """Test fallback when SOLUTION not found."""
        response = "I recommend using Antwerp as the alternative."
        options = ["rotterdam", "hamburg", "antwerp"]
        result = parse_reflexion_response(response, options)

        assert result["solution"] == "antwerp"


class TestBuildReasoningPrompt:
    """Tests for build_reasoning_prompt function."""

    def test_build_basic_prompt(self, mock_parsed_task):
        """Test building a basic reasoning prompt."""
        prompt = build_reasoning_prompt(
            task="Book shipment Rotterdam to Boston",
            parsed_task=mock_parsed_task,
            memories="Previous: Rotterdam blocked",
            learned_rules="R-482: Use Antwerp",
        )

        assert "Rotterdam" in prompt
        assert "Boston" in prompt

    def test_build_prompt_with_forbidden(self, mock_parsed_task):
        """Test prompt includes forbidden section."""
        prompt = build_reasoning_prompt(
            task="Book shipment",
            parsed_task=mock_parsed_task,
            memories="",
            learned_rules="",
            forbidden_section="FORBIDDEN: rotterdam, hamburg",
        )

        assert "FORBIDDEN" in prompt

    def test_build_prompt_with_error_feedback(self, mock_parsed_task):
        """Test prompt includes error feedback."""
        prompt = build_reasoning_prompt(
            task="Book shipment",
            parsed_task=mock_parsed_task,
            memories="",
            learned_rules="",
            error_feedback="Previous attempt failed: R-482",
        )

        assert "R-482" in prompt


class TestStatisticsFunctions:
    """Tests for statistics computation functions."""

    def test_compute_success_rate(self):
        """Test success rate computation."""
        assert compute_success_rate(5, 10) == 0.5
        assert compute_success_rate(10, 10) == 1.0
        assert compute_success_rate(0, 10) == 0.0
        assert compute_success_rate(0, 0) == 0.0  # Edge case

    def test_compute_average(self):
        """Test average computation."""
        assert compute_average([1.0, 2.0, 3.0]) == 2.0
        assert compute_average([10.0]) == 10.0
        assert compute_average([]) == 0.0  # Edge case

    def test_update_failure_counter(self):
        """Test failure counter updates."""
        assert update_failure_counter(3, True) == 0  # Reset on success
        assert update_failure_counter(3, False) == 4  # Increment on failure
        assert update_failure_counter(0, False) == 1

    def test_increment_counters(self):
        """Test counter incrementing."""
        result = increment_counters(5, 3)
        assert result["tasks_since_consolidation"] == 6
        assert result["tasks_since_compass"] == 4


class TestLearningTriggers:
    """Tests for learning trigger functions."""

    def test_should_trigger_consolidation(self):
        """Test consolidation trigger logic."""
        assert should_trigger_consolidation(3, 3) is True
        assert should_trigger_consolidation(2, 3) is False
        assert should_trigger_consolidation(5, 3) is True

    def test_should_trigger_compass_evolution(self):
        """Test COMPASS evolution trigger logic."""
        # Trigger by interval
        assert (
            should_trigger_compass_evolution(
                tasks_since_compass=5,
                consecutive_failures=0,
                compass_evolution_interval=5,
                failure_threshold=3,
                enable_compass_optimization=True,
            )
            is True
        )

        # Trigger by failures
        assert (
            should_trigger_compass_evolution(
                tasks_since_compass=1,
                consecutive_failures=3,
                compass_evolution_interval=5,
                failure_threshold=3,
                enable_compass_optimization=True,
            )
            is True
        )

        # Disabled
        assert (
            should_trigger_compass_evolution(
                tasks_since_compass=10,
                consecutive_failures=5,
                compass_evolution_interval=5,
                failure_threshold=3,
                enable_compass_optimization=False,
            )
            is False
        )


class TestTaskResultHelpers:
    """Tests for task result helper functions."""

    def test_build_task_result(self):
        """Test building task result dict."""
        result = build_task_result(
            success=True,
            task_steps=3,
            overhead_steps=1,
            duration=2.5,
            response="Booking successful",
            strategy="learned_rule",
            complexity="medium",
            domain="logistics",
        )

        assert result["success"] is True
        assert result["steps"] == 3
        assert result["overhead"] == 1
        assert result["duration"] == 2.5
        assert result["domain"] == "logistics"

    def test_build_task_record(self):
        """Test building task record for scoring."""
        record = build_task_record(
            task="Book shipment",
            success=True,
            steps=3,
            overhead=1,
            duration=2.0,
            strategy="pivot",
        )

        assert record["task"] == "Book shipment"
        assert record["success"] is True
        assert record["strategy"] == "pivot"

    def test_apply_procedure_hint(self, mock_parsed_task):
        """Test applying procedure hint."""
        # With valid procedure
        applied = apply_procedure_hint(mock_parsed_task, "Step 1: Check port")
        assert applied is True
        assert "procedure_hint" in mock_parsed_task.parameters

        # With no procedure
        mock_parsed_task.parameters = {}
        applied = apply_procedure_hint(mock_parsed_task, "No procedure found")
        assert applied is False

    def test_apply_llm_suggestion(self, mock_parsed_task):
        """Test applying LLM suggestion."""
        mock_parsed_task.parameters = {}

        # With valid suggestion
        suggestion = {"suggested_solution": "Antwerp", "reasoning": "Best option"}
        applied, strategy = apply_llm_suggestion(mock_parsed_task, suggestion)
        assert applied is True
        assert "Antwerp" in mock_parsed_task.parameters.get("preferred_solution", "")
        assert "LLM-Reasoned" in strategy

        # With None
        applied, strategy = apply_llm_suggestion(mock_parsed_task, None)
        assert applied is False
        assert strategy == ""


class TestLLMStatsHelpers:
    """Tests for LLM statistics helper functions."""

    def test_format_error_feedback(self):
        """Test error feedback formatting."""
        feedback = format_error_feedback("R-482: Port blocked")
        assert "R-482" in feedback
        assert "Previous attempt failed" in feedback

        # Empty feedback
        assert format_error_feedback("") == ""

    def test_build_llm_reasoning_result(self):
        """Test building LLM reasoning result."""
        from precept.agent_functions import LLMSuggestion

        suggestion = LLMSuggestion(
            suggested_solution="Antwerp",
            reasoning="Best port",
            confidence="high",
        )
        result = build_llm_reasoning_result(suggestion)

        assert result["suggested_solution"] == "Antwerp"
        assert result["reasoning"] == "Best port"
        assert result["confidence"] == "high"

        # None input
        assert build_llm_reasoning_result(None) is None

    def test_update_llm_stats(self):
        """Test LLM stats updating."""
        # Success case
        stats = update_llm_stats(
            calls=5, successes=3, failures=2, result={"suggested_solution": "Antwerp"}
        )
        assert stats["calls"] == 6
        assert stats["successes"] == 4
        assert stats["failures"] == 2

        # Failure case
        stats = update_llm_stats(calls=5, successes=3, failures=2, result=None)
        assert stats["calls"] == 6
        assert stats["successes"] == 3
        assert stats["failures"] == 3


class TestUsefulnessFeedback:
    """Tests for usefulness feedback computation."""

    def test_compute_usefulness_feedback(self):
        """Test usefulness feedback values."""
        assert compute_usefulness_feedback(True) == 0.5
        assert compute_usefulness_feedback(False) == -0.3


class TestFormatFailureContext:
    """Tests for failure context formatting."""

    def test_format_failure_context_on_failure(self):
        """Test formatting when task failed."""
        context = format_failure_context(
            task="Book shipment", response="R-482 error", success=False
        )
        assert "Book shipment" in context
        assert "R-482" in context

    def test_format_failure_context_on_success(self):
        """Test formatting when task succeeded."""
        context = format_failure_context(
            task="Book shipment", response="Success", success=True
        )
        assert context == ""


# =============================================================================
# CONTEXT FETCH TESTS
# =============================================================================


class TestContextFetchResult:
    """Tests for ContextFetchResult dataclass."""

    def test_create_context_result(self):
        """Test creating a ContextFetchResult."""
        from precept.agent_functions import ContextFetchResult

        result = ContextFetchResult(
            memories="Previous Rotterdam failure",
            procedure="Step 1: Check port status",
            rules="R-482: Rotterdam blocked → Use Antwerp",
        )

        assert result.memories == "Previous Rotterdam failure"
        assert result.procedure == "Step 1: Check port status"
        assert result.rules == "R-482: Rotterdam blocked → Use Antwerp"

    def test_context_result_fields(self):
        """Test ContextFetchResult has all expected fields."""
        from precept.agent_functions import ContextFetchResult

        result = ContextFetchResult(
            memories="",
            procedure="",
            rules="",
        )

        assert hasattr(result, "memories")
        assert hasattr(result, "procedure")
        assert hasattr(result, "rules")


class TestFetchContext:
    """Tests for fetch_context function."""

    def test_fetch_context_import(self):
        """Test fetch_context can be imported."""
        from precept.agent_functions import fetch_context

        assert fetch_context is not None

    def test_context_fetch_result_is_dataclass(self):
        """Test ContextFetchResult is a dataclass."""
        from precept.agent_functions import ContextFetchResult
        from dataclasses import is_dataclass

        assert is_dataclass(ContextFetchResult)


class TestMCPClientProtocol:
    """Tests for MCPClientProtocol interface."""

    def test_protocol_has_standard_retrieval(self):
        """Test MCPClientProtocol has retrieve_memories method."""
        from precept.agent_functions import MCPClientProtocol

        # Check if retrieve_memories is defined
        assert hasattr(MCPClientProtocol, "retrieve_memories")

    def test_protocol_has_standard_methods(self):
        """Test MCPClientProtocol has standard methods."""
        from precept.agent_functions import MCPClientProtocol

        standard_methods = [
            "retrieve_memories",
            "get_procedure",
            "get_learned_rules",
            "record_error",
            "record_solution",
            "store_experience",
        ]

        for method in standard_methods:
            assert hasattr(MCPClientProtocol, method)
