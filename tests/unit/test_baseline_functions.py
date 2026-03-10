"""
Unit Tests for precept.baseline_functions module.

Tests pure functions for baseline agent operations including:
- Response parsing
- Context building
- Reflection memory management
- Statistics computation
"""

import pytest

from precept.baseline_functions import (
    add_reflection,
    build_baseline_llm_prompt,
    build_core_stats,
    build_current_episode_context,
    build_error_context,
    build_reflection_section,
    clear_reflection_memory,
    compute_average_steps,
    compute_llm_accuracy,
    compute_per_task_rate,
    compute_success_rate,
    create_reflection_record,
    extract_solution_from_response,
    find_option_in_text,
    format_accumulated_reflections,
    get_memory_stats,
    get_reflection_memory,
    match_option,
    parse_baseline_llm_response,
)


class TestResponseParsing:
    """Tests for LLM response parsing functions."""

    def test_extract_solution_from_response(self):
        """Test extracting SOLUTION field."""
        response = "SOLUTION: Antwerp\nREASONING: Best option"
        result = extract_solution_from_response(response)
        assert result == "antwerp"

    def test_extract_solution_not_found(self):
        """Test when SOLUTION not in response."""
        response = "I recommend using Antwerp"
        result = extract_solution_from_response(response)
        assert result is None

    def test_match_option(self):
        """Test matching suggested to valid options."""
        options = ["rotterdam", "hamburg", "antwerp"]

        # Exact match
        assert match_option("antwerp", options) == "antwerp"

        # Partial match is intentionally unsupported (exact matching only)
        assert match_option("ant", options) is None

        # No match
        assert match_option("london", options) is None

    def test_find_option_in_text(self):
        """Test finding any option in response text."""
        options = ["rotterdam", "hamburg", "antwerp"]

        response = "The best choice would be Antwerp due to availability"
        assert find_option_in_text(response, options) == "antwerp"

        response = "No valid port mentioned"
        assert find_option_in_text(response, options) is None

    def test_parse_baseline_llm_response(self):
        """Test full response parsing."""
        options = ["rotterdam", "hamburg", "antwerp"]

        # With SOLUTION field
        response = "SOLUTION: antwerp"
        assert parse_baseline_llm_response(response, options) == "antwerp"

        # Fallback to text search
        response = "I suggest using Hamburg for this route"
        assert parse_baseline_llm_response(response, options) == "hamburg"

        # No match
        response = "Unable to determine"
        assert parse_baseline_llm_response(response, options) is None


class TestContextBuilding:
    """Tests for context building functions."""

    def test_build_error_context(self):
        """Test error context building."""
        context = build_error_context(
            failed_options=["rotterdam", "hamburg"], last_error="R-482: Port blocked"
        )

        assert "rotterdam" in context
        assert "hamburg" in context
        assert "R-482" in context
        assert "PREVIOUS ATTEMPT FAILED" in context

    def test_build_error_context_empty(self):
        """Test empty error context."""
        assert build_error_context([], None) == ""
        assert build_error_context([], "error") == ""
        assert build_error_context(["opt"], None) == ""

    def test_build_reflection_section(self):
        """Test reflection section building."""
        attempts = [
            {
                "option": "rotterdam",
                "error": "R-482",
                "reflection": "Port is blocked",
            },
            {
                "option": "hamburg",
                "error": "H-903",
                "reflection": "US routes blocked",
            },
        ]

        section = build_reflection_section(attempts)

        assert "Attempt 1" in section
        assert "Attempt 2" in section
        assert "rotterdam" in section
        assert "R-482" in section
        assert "MUST reflect" in section

    def test_build_reflection_section_empty(self):
        """Test empty reflection section."""
        assert build_reflection_section([]) == ""

    def test_build_current_episode_context(self):
        """Test current episode context building."""
        attempts = [
            {"option": "rotterdam", "error": "R-482"},
            {"option": "hamburg", "error": "H-903"},
        ]

        context = build_current_episode_context(attempts)

        assert "CURRENT EPISODE" in context
        assert "rotterdam" in context
        assert "DIFFERENT option" in context

    def test_build_current_episode_context_empty(self):
        """Test empty episode context."""
        assert build_current_episode_context([]) == ""


class TestReflectionMemory:
    """Tests for reflection memory management."""

    def setup_method(self):
        """Clear memory before each test."""
        clear_reflection_memory()

    def test_add_and_get_reflection(self):
        """Test adding and retrieving reflections."""
        reflection = {
            "episode": 1,
            "task": "Book shipment",
            "outcome": "success",
            "reflection": "Rotterdam blocked",
        }

        add_reflection("logistics:book", reflection)
        memories = get_reflection_memory("logistics:book")

        assert len(memories) == 1
        assert memories[0]["task"] == "Book shipment"

    def test_reflection_memory_max_size(self):
        """Test memory pruning at max size."""
        for i in range(25):
            add_reflection(
                "test_type",
                {"episode": i, "task": f"Task {i}"},
                max_size=20,
            )

        memories = get_reflection_memory("test_type")
        assert len(memories) == 20
        # Should keep most recent
        assert memories[0]["episode"] == 5
        assert memories[-1]["episode"] == 24

    def test_clear_specific_memory(self):
        """Test clearing specific task type memory."""
        add_reflection("type_a", {"episode": 1})
        add_reflection("type_b", {"episode": 1})

        clear_reflection_memory("type_a")

        assert len(get_reflection_memory("type_a")) == 0
        assert len(get_reflection_memory("type_b")) == 1

    def test_clear_all_memory(self):
        """Test clearing all memory."""
        add_reflection("type_a", {"episode": 1})
        add_reflection("type_b", {"episode": 1})

        clear_reflection_memory()

        assert get_memory_stats() == {}

    def test_get_memory_stats(self):
        """Test memory statistics."""
        add_reflection("type_a", {"episode": 1})
        add_reflection("type_a", {"episode": 2})
        add_reflection("type_b", {"episode": 1})

        stats = get_memory_stats()

        assert stats["type_a"] == 2
        assert stats["type_b"] == 1

    def test_format_accumulated_reflections(self):
        """Test formatting accumulated reflections."""
        add_reflection(
            "logistics:book",
            {
                "episode": 1,
                "task": "Book Rotterdam to Boston",
                "outcome": "success",
                "reflection": "Used Antwerp",
                "lesson": "Rotterdam blocked",
                "failed_options": ["rotterdam"],
                "successful_option": "antwerp",
            },
        )

        formatted = format_accumulated_reflections("logistics:book")

        assert "Episode 1" in formatted
        assert "Rotterdam" in formatted or "rotterdam" in formatted
        assert "antwerp" in formatted

    def test_format_accumulated_reflections_empty(self):
        """Test formatting when no reflections."""
        formatted = format_accumulated_reflections("nonexistent")
        assert "first episode" in formatted.lower()


class TestStatisticsFunctions:
    """Tests for statistics computation functions."""

    def test_compute_success_rate(self):
        """Test success rate computation."""
        assert compute_success_rate(8, 10) == 0.8
        assert compute_success_rate(0, 10) == 0.0
        assert compute_success_rate(0, 0) == 0.0

    def test_compute_average_steps(self):
        """Test average steps computation."""
        assert compute_average_steps([2, 3, 4]) == 3.0
        assert compute_average_steps([5]) == 5.0
        assert compute_average_steps([]) == 0.0

    def test_compute_per_task_rate(self):
        """Test per-task rate computation."""
        assert compute_per_task_rate(10, 5) == 2.0
        assert compute_per_task_rate(0, 5) == 0.0
        assert compute_per_task_rate(10, 0) == 0.0

    def test_compute_llm_accuracy(self):
        """Test LLM accuracy computation."""
        assert compute_llm_accuracy(8, 10) == 0.8
        assert compute_llm_accuracy(0, 10) == 0.0
        assert compute_llm_accuracy(0, 0) == 0.0

    def test_build_core_stats(self):
        """Test building core statistics."""
        stats = build_core_stats(
            total_tasks=10,
            successful_tasks=8,
            steps_per_task=[2, 3, 4, 3, 2, 3, 4, 3, 2, 3],
            llm_calls=20,
            llm_suggestions_followed=15,
            llm_suggestions_failed=5,
            domain="logistics",
            baseline_type="adapted_react",
        )

        assert stats["total_tasks"] == 10
        assert stats["successful_tasks"] == 8
        assert stats["success_rate"] == 0.8
        assert stats["avg_steps"] == 2.9
        assert stats["llm_calls"] == 20
        assert stats["llm_accuracy"] == 0.75
        assert stats["domain"] == "logistics"
        assert stats["baseline_type"] == "adapted_react"


class TestReflectionRecord:
    """Tests for reflection record creation."""

    def test_create_reflection_record(self):
        """Test creating a reflection record."""
        record = create_reflection_record(
            episode=1,
            task="Book shipment Rotterdam to Boston",
            success=True,
            reflection="Rotterdam was blocked",
            lesson="Use Antwerp for Rotterdam routes",
            failed_options=["rotterdam"],
            successful_option="antwerp",
            attempts=2,
        )

        assert record["episode"] == 1
        assert record["task"] == "Book shipment Rotterdam to Boston"
        assert record["outcome"] == "success"
        assert record["reflection"] == "Rotterdam was blocked"
        assert record["lesson"] == "Use Antwerp for Rotterdam routes"
        assert record["failed_options"] == ["rotterdam"]
        assert record["successful_option"] == "antwerp"
        assert record["attempts"] == 2
        assert "timestamp" in record

    def test_create_reflection_record_failure(self):
        """Test creating failure reflection record."""
        record = create_reflection_record(
            episode=2,
            task="Failed task",
            success=False,
            reflection="All options failed",
            lesson=None,
            failed_options=["a", "b", "c"],
            successful_option=None,
            attempts=3,
        )

        assert record["outcome"] == "failure"
        assert record["successful_option"] is None


class TestPromptBuilding:
    """Tests for prompt building functions."""

    def test_build_baseline_llm_prompt(self, mock_parsed_task):
        """Test building baseline prompt.
        
        Note: Options are intentionally NOT included in the prompt for fair
        comparison with PRECEPT (which also doesn't get options).
        """
        prompt = build_baseline_llm_prompt(
            task="Book shipment Rotterdam to Boston",
            parsed_task=mock_parsed_task,
            options=["rotterdam", "hamburg", "antwerp"],  # Not included in prompt
            memories="Previous: Rotterdam blocked",
        )

        assert "Rotterdam" in prompt or "rotterdam" in prompt.lower()
        assert "Boston" in prompt or "boston" in prompt.lower()
        # Options are NOT included in prompt for fair comparison
        # The prompt should include task context and memories

    def test_build_baseline_llm_prompt_with_error(self, mock_parsed_task):
        """Test prompt with error context."""
        error_context = "Previous attempt failed: R-482"
        prompt = build_baseline_llm_prompt(
            task="Book shipment",
            parsed_task=mock_parsed_task,
            options=["antwerp"],
            memories="",
            error_context=error_context,
        )

        assert "R-482" in prompt
