"""
LLM Baseline Agents for Fair Comparison with PRECEPT.

This module provides THREE fair LLM baselines that use the SAME LLM as PRECEPT
but do NOT learn across tasks. This isolates PRECEPT's learning advantage.

Architecture follows software engineering best practices:
- Dependency injection for configuration
- Pure functions for core logic (in baseline_functions.py)
- Configuration-driven behavior (via BaselineConfig)

═══════════════════════════════════════════════════════════════════════════════
BASELINE 1: LLMBaselineAgent (Adapted ReAct)
═══════════════════════════════════════════════════════════════════════════════
- Simple error feedback loop
- On failure: pass error message to next LLM call
- LLM suggests solution based on error context
- NO explicit reflection or lesson extraction

═══════════════════════════════════════════════════════════════════════════════
BASELINE 2: ReflexionBaselineAgent (Adapted Reflexion)
═══════════════════════════════════════════════════════════════════════════════
- Explicit reflection after failures
- LLM generates: "What went wrong? What should I try differently?"
- Reflection is passed to next attempt (within-task only)
- Still NO cross-task learning (reflections are forgotten between tasks)

═══════════════════════════════════════════════════════════════════════════════
BASELINE 3: FullReflexionBaselineAgent (Full Reflexion Paper)
═══════════════════════════════════════════════════════════════════════════════
- Cross-episode memory for the same task type
- Reflections persist across episodes
- Implements the full Reflexion paper (Shinn et al., 2023)

FAIR COMPARISON GUARANTEE:
All baselines use the SAME LLM call budget as PRECEPT per attempt.
The only difference is WHAT we ask the LLM to do in that call.

Usage:
    from precept import LLMBaselineAgent, ReflexionBaselineAgent
    from precept.config import BaselineConfig

    # With default config
    baseline = LLMBaselineAgent(baseline_strategy=LogisticsBaselineStrategy())

    # With custom config
    config = BaselineConfig(max_attempts=5, verbose=True)
    baseline = LLMBaselineAgent(baseline_strategy=strategy, config=config)
"""

import asyncio
import os
import random
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from .agent_functions import parse_reflexion_response
from .baseline_functions import (
    # ExpeL functions
    add_expel_insight,
    add_expel_stats,
    add_reflection,
    build_baseline_llm_prompt,
    build_core_stats,
    build_current_episode_context,
    build_error_context,
    build_expel_insight_extraction_prompt,
    build_expel_task_prompt,
    build_full_reflexion_llm_prompt,
    build_reflection_section,
    build_reflexion_llm_prompt,
    clear_expel_insights,
    clear_reflection_memory,
    compute_per_task_rate,
    compute_success_rate,
    create_reflection_record,
    extract_conditions_from_task,
    format_accumulated_reflections,
    get_expel_insights,
    get_expel_stats,
    get_memory_stats,
    get_reflection_memory,
    parse_baseline_llm_response,
    parse_expel_insight_response,
    parse_expel_task_response,
    retrieve_expel_insights_by_task,
)
from .config import (
    BaselineConfig,
    PromptTemplates,
    get_baseline_logger,
    get_default_config,
)
from .domain_strategies.base import BaselineDomainStrategy, ParsedTask
from .domain_strategies.logistics import LogisticsBaselineStrategy
from .structured_outputs import (
    create_structured_output_params,
    get_baseline_schema,
    get_expel_insight_schema,
    get_expel_task_schema,
    get_full_reflexion_schema,
    get_reflexion_schema,
)

# Module-level logger (lazy initialization to avoid stdout logging during import)
# This is critical for MCP server which uses stdout for JSONRPC
_logger = None


def _get_logger():
    """Get or create the module logger (lazy initialization)."""
    global _logger
    if _logger is None:
        _logger = get_baseline_logger()
    return _logger


# MCP imports
try:
    from mcp import StdioServerParameters

    MCP_AVAILABLE = True
except ImportError:
    MCP_AVAILABLE = False
    StdioServerParameters = None

# OpenAI imports for LLM baseline
try:
    from openai import AsyncOpenAI

    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    AsyncOpenAI = None


class LLMBaselineAgent:
    """
    LLM-enabled baseline agent - Uses LLM reasoning but NO CROSS-TASK LEARNING!

    This baseline has access to EVERYTHING PRECEPT has WITHIN a single task:
    ✓ Same LLM (GPT-4o-mini) for reasoning
    ✓ Same MCP tools for execution
    ✓ Same memory retrieval mechanism
    ✓ Same error feedback from MCP server
    ✓ Same ability to re-reason with error context
    ✓ Same retry budget (MAX_ATTEMPTS)

    But DOES NOT have PRECEPT's cross-task capabilities:
    ✗ NO rule learning from PREVIOUS tasks
    ✗ NO memory consolidation across tasks
    ✗ NO COMPASS prompt evolution
    ✗ NO pattern recognition across tasks

    Usage:
        # With default config
        baseline = LLMBaselineAgent(baseline_strategy=LogisticsBaselineStrategy())

        # With custom config
        config = BaselineConfig(max_attempts=5, verbose=True)
        baseline = LLMBaselineAgent(baseline_strategy=strategy, config=config)
    """

    def __init__(
        self,
        baseline_strategy: Optional[BaselineDomainStrategy] = None,
        config: Optional[BaselineConfig] = None,
        # Legacy parameters for backward compatibility
        model: Optional[str] = None,
        server_script: Optional[Path] = None,
        verbose: Optional[bool] = None,
        max_internal_workers: Optional[int] = None,
    ):
        """
        Initialize the LLM baseline agent.

        Args:
            baseline_strategy: The baseline strategy to use (defaults to logistics)
            config: Baseline configuration (preferred)
            model: The OpenAI model to use (legacy, use config.model)
            server_script: Path to the MCP server script (legacy)
            verbose: Whether to print LLM reasoning (legacy)
            max_internal_workers: Max concurrent internal operations (legacy)
        """
        if not OPENAI_AVAILABLE:
            raise ImportError(
                "OpenAI library not available. Install with: pip install openai"
            )

        # Initialize configuration
        self.config = config or get_default_config().baseline

        # Apply legacy parameter overrides
        if model is not None:
            self.config.model = model
        if verbose is not None:
            self.config.verbose = verbose
        if max_internal_workers is not None:
            self.config.max_internal_workers = max_internal_workers

        # Store strategy
        self.strategy = baseline_strategy or LogisticsBaselineStrategy()

        # Server script
        if server_script is None:
            self.server_script = Path(__file__).parent / "precept_mcp_server.py"
        else:
            self.server_script = server_script

        # Clients (initialized in connect())
        self.mcp_client = None
        self.llm_client: Optional[AsyncOpenAI] = None

        # Internal concurrency control
        self._internal_semaphore: Optional[asyncio.Semaphore] = None

        # Statistics
        self.total_tasks = 0
        self.successful_tasks = 0
        self.steps_per_task: List[int] = []
        self.llm_calls = 0
        self.llm_suggestions_followed = 0
        self.llm_suggestions_failed = 0

        # Prompt templates
        self.prompts = PromptTemplates()

    # =========================================================================
    # PROPERTY ACCESSORS
    # =========================================================================

    @property
    def model(self) -> str:
        """Get the LLM model name."""
        return self.config.model

    @property
    def verbose(self) -> bool:
        """Check if verbose logging is enabled."""
        return self.config.verbose

    @property
    def max_internal_workers(self) -> int:
        """Get max internal workers."""
        return self.config.max_internal_workers

    @property
    def MAX_ATTEMPTS(self) -> int:
        """Get max attempts."""
        return self.config.max_attempts

    # =========================================================================
    # CONNECTION METHODS
    # =========================================================================

    async def connect(self) -> None:
        """Connect to MCP server and initialize LLM client."""
        if not MCP_AVAILABLE:
            raise ImportError(
                "MCP library not available. Install with: pip install mcp"
            )

        # Initialize internal semaphore
        self._internal_semaphore = asyncio.Semaphore(self.max_internal_workers)

        # Initialize LLM client
        self.llm_client = AsyncOpenAI()

        # Import here to avoid circular imports
        from .precept_mcp_client import LogisticsMCPClient

        # Get project root for PYTHONPATH
        project_root = self.server_script.parent.parent.parent

        self.mcp_client = LogisticsMCPClient(
            server_params=StdioServerParameters(
                command="python3",
                args=[str(self.server_script)],
                env={**os.environ, "PYTHONPATH": str(project_root / "src")},
            )
        )
        await self.mcp_client.connect()

        _get_logger().info(
            f"  ✓ LLM Baseline connected [{self.strategy.domain_name}] (model: {self.model})"
        )
        _get_logger().info(
            f"    Internal concurrency: {self.max_internal_workers} workers"
        )
        _get_logger().info("    • Uses LLM reasoning (stateless - no learning)")
        _get_logger().info("    • Same MCP tools as PRECEPT")
        _get_logger().info("    • NO rule learning, NO memory consolidation")

    async def disconnect(self) -> None:
        """Disconnect from server with proper async cleanup."""
        # Close OpenAI client first to release HTTP connections
        # This prevents "Event loop is closed" errors during parallel execution
        if hasattr(self, "llm_client") and self.llm_client:
            try:
                await self.llm_client.close()
            except (RuntimeError, asyncio.CancelledError):
                # Event loop may be closing - this is expected during shutdown
                pass
            except Exception:
                pass  # Suppress all cleanup errors
            finally:
                self.llm_client = None

        # Then close MCP client
        if self.mcp_client:
            try:
                await self.mcp_client.disconnect()
            except (RuntimeError, asyncio.CancelledError):
                pass  # Event loop may be closing
            except Exception:
                pass  # Suppress cleanup errors

    # =========================================================================
    # LLM REASONING
    # =========================================================================

    async def _ask_llm(
        self,
        task: str,
        parsed_task: ParsedTask,
        memories: str,
        options: List[str],
        error_feedback: Optional[str] = None,
        failed_options: Optional[List[str]] = None,
    ) -> Optional[str]:
        """
        Ask LLM for a solution suggestion (stateless - no cross-task learning).

        Uses OpenAI structured outputs for reliable JSON parsing instead of regex.

        Args:
            task: The original task string
            parsed_task: Parsed task information
            memories: Retrieved memories (may be empty)
            options: Available options for this task
            error_feedback: Error message from previous attempt
            failed_options: Options that already failed

        Returns:
            Suggested option from LLM, or None if parsing fails
        """
        import json

        self.llm_calls += 1

        # Build error context
        error_context = build_error_context(failed_options or [], error_feedback)

        # Build prompt
        prompt = build_baseline_llm_prompt(
            task=task,
            parsed_task=parsed_task,
            options=options,
            memories=memories,
            error_context=error_context,
            prompts=self.prompts,
        )

        # Add structured output instruction to prompt
        structured_instruction = """

Respond with a JSON object containing:
- "solution": your suggested solution (a single word/phrase)
- "reasoning": brief explanation for your choice
- "confidence": "high", "medium", or "low"
"""
        prompt = prompt + structured_instruction

        if self.verbose:
            _get_logger().debug(
                f"[LLM Baseline] _ask_llm called with {len(options)} options..."
            )

        try:
            if self.verbose:
                _get_logger().debug(
                    "[LLM Baseline] Calling LLM API with structured output..."
                )

            # Use structured output for guaranteed JSON format
            structured_params = create_structured_output_params(get_baseline_schema())

            response = await self.llm_client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a helpful assistant for domain-specific tasks. Always respond with valid JSON.",
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
                **structured_params,
            )

            response_text = response.choices[0].message.content or ""

            if self.verbose:
                _get_logger().debug(
                    f"[LLM Baseline] Response: {response_text[:150]}..."
                )

            # Parse JSON response (guaranteed valid by structured output)
            try:
                parsed = json.loads(response_text)
                suggested = parsed.get("solution", "").strip().lower()

                # Match to valid options
                for opt in options:
                    if opt.lower() == suggested:
                        if self.verbose:
                            _get_logger().debug(
                                f"[LLM Baseline] Parsed solution: {opt}"
                            )
                        return opt

                # If no exact match, return the raw suggestion (might still work)
                if self.verbose:
                    _get_logger().debug(
                        f"[LLM Baseline] Parsed solution (raw): {suggested}"
                    )
                return suggested if suggested else None

            except json.JSONDecodeError:
                # Fallback to regex parsing if JSON fails
                _get_logger().warning(
                    "[LLM Baseline] JSON parse failed, falling back to regex"
                )
                solution = parse_baseline_llm_response(response_text, options)
                return solution

        except Exception as e:
            _get_logger().error(f"[LLM Baseline] ERROR: {type(e).__name__}: {e}")
            return None

    # =========================================================================
    # TASK EXECUTION
    # =========================================================================

    async def run_task(self, task: str) -> Dict[str, Any]:
        """
        Run task using LLM reasoning with error feedback - NO CROSS-TASK LEARNING.

        Args:
            task: The task string to execute

        Returns:
            Dict with success, steps, duration, attempts, llm_suggestions, domain
        """
        self.total_tasks += 1
        start_time = time.time()

        if self.verbose:
            _get_logger().debug(f"[LLM Baseline] Starting run_task for: {task[:50]}...")

        # Parse task
        parsed_task = self.strategy.parse_task(task)

        steps = 0
        success = False
        attempts = 0
        tried_options: List[str] = []
        llm_suggestions: List[str] = []
        last_error: Optional[str] = None

        try:
            # Memory retrieval
            query = f"{parsed_task.action} {parsed_task.target or parsed_task.entity}"
            memories = await self.mcp_client.retrieve_memories(query)
            memories_str = str(memories) if memories else ""
            steps += 1

            # Get available options
            available_options = self.strategy.get_options_for_task(parsed_task)

            if self.verbose:
                _get_logger().debug(
                    f"[LLM Baseline] Available options: {available_options[:5]}..."
                )

            # Retry loop (stateless - no memory of what was tried)
            while attempts < self.MAX_ATTEMPTS:
                if self.verbose:
                    _get_logger().debug(
                        f"[LLM Baseline] Attempt {attempts + 1}/{self.MAX_ATTEMPTS}"
                    )

                # Baseline sees ALL options every time (no memory!)
                remaining_options = available_options.copy()

                if self.verbose:
                    _get_logger().debug(
                        f"[LLM Baseline] Available options: {remaining_options}"
                    )

                # Ask LLM for suggestion
                llm_suggestion = await self._ask_llm(
                    task=task,
                    parsed_task=parsed_task,
                    memories=memories_str,
                    options=remaining_options,
                    error_feedback=last_error,
                    failed_options=None,  # NO MEMORY of what failed!
                )
                steps += 1

                # Determine option to try
                if llm_suggestion and llm_suggestion in remaining_options:
                    option_to_try = llm_suggestion
                    llm_suggestions.append(llm_suggestion)
                else:
                    option_to_try = random.choice(remaining_options)

                # Execute action
                tried_options.append(option_to_try)
                exec_success, result = await self.strategy.execute_action(
                    self.mcp_client,
                    option_to_try,
                    parsed_task,
                )
                steps += 1
                attempts += 1

                if exec_success:
                    success = True
                    if llm_suggestion == option_to_try:
                        self.llm_suggestions_followed += 1
                    break
                else:
                    last_error = str(result) if result else "Operation failed"
                    if llm_suggestion == option_to_try:
                        self.llm_suggestions_failed += 1

        except Exception as e:
            if self.verbose:
                _get_logger().error(f"Error: {e}")

        duration = time.time() - start_time

        if success:
            self.successful_tasks += 1
        self.steps_per_task.append(steps)

        return {
            "success": success,
            "steps": steps,
            "duration": duration,
            "attempts": attempts,
            "tried_options": tried_options,
            "llm_suggestions": llm_suggestions,
            "llm_suggestion": llm_suggestions[0] if llm_suggestions else None,
            "domain": self.strategy.domain_name,
        }

    # =========================================================================
    # STATISTICS
    # =========================================================================

    def get_success_rate(self) -> float:
        """Get the current success rate using pure function."""
        return compute_success_rate(self.successful_tasks, self.total_tasks)

    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive statistics using pure functions."""
        return build_core_stats(
            total_tasks=self.total_tasks,
            successful_tasks=self.successful_tasks,
            steps_per_task=self.steps_per_task,
            llm_calls=self.llm_calls,
            llm_suggestions_followed=self.llm_suggestions_followed,
            llm_suggestions_failed=self.llm_suggestions_failed,
            domain=self.strategy.domain_name,
            baseline_type="adapted_react",
        )


class ReflexionBaselineAgent:
    """
    Adapted Reflexion baseline - Uses EXPLICIT REFLECTION but NO CROSS-TASK LEARNING.

    This baseline adds EXPLICIT REFLECTION after failures:
    1. Parse task
    2. Ask LLM for suggestion
    3. Execute action
    4. On failure: Ask LLM to REFLECT on what went wrong
    5. Pass reflection to next attempt
    6. Repeat until success or MAX_ATTEMPTS

    KEY DIFFERENCES FROM LLMBaselineAgent:
    - LLMBaselineAgent: Just passes error message to next call
    - ReflexionBaselineAgent: Asks LLM to explicitly reflect

    Usage:
        reflexion = ReflexionBaselineAgent(baseline_strategy=LogisticsBaselineStrategy())
        await reflexion.connect()
        result = await reflexion.run_task("Book shipment Rotterdam to Boston")
    """

    def __init__(
        self,
        baseline_strategy: Optional[BaselineDomainStrategy] = None,
        config: Optional[BaselineConfig] = None,
        model: Optional[str] = None,
        server_script: Optional[Path] = None,
        verbose: Optional[bool] = None,
        max_internal_workers: Optional[int] = None,
    ):
        """Initialize the Reflexion baseline agent."""
        if not OPENAI_AVAILABLE:
            raise ImportError(
                "OpenAI library not available. Install with: pip install openai"
            )

        self.config = config or get_default_config().baseline

        if model is not None:
            self.config.model = model
        if verbose is not None:
            self.config.verbose = verbose
        if max_internal_workers is not None:
            self.config.max_internal_workers = max_internal_workers

        self.strategy = baseline_strategy or LogisticsBaselineStrategy()

        if server_script is None:
            self.server_script = Path(__file__).parent / "precept_mcp_server.py"
        else:
            self.server_script = server_script

        self.mcp_client = None
        self.llm_client: Optional[AsyncOpenAI] = None
        self._internal_semaphore: Optional[asyncio.Semaphore] = None

        # Statistics
        self.total_tasks = 0
        self.successful_tasks = 0
        self.steps_per_task: List[int] = []
        self.llm_calls = 0
        self.reflections_generated = 0
        self.llm_suggestions_followed = 0
        self.llm_suggestions_failed = 0

        self.prompts = PromptTemplates()

    @property
    def model(self) -> str:
        return self.config.model

    @property
    def verbose(self) -> bool:
        return self.config.verbose

    @property
    def max_internal_workers(self) -> int:
        return self.config.max_internal_workers

    @property
    def MAX_ATTEMPTS(self) -> int:
        return self.config.max_attempts

    async def connect(self) -> None:
        """Connect to MCP server and initialize LLM client."""
        if not MCP_AVAILABLE:
            raise ImportError("MCP library not available.")

        self._internal_semaphore = asyncio.Semaphore(self.max_internal_workers)
        self.llm_client = AsyncOpenAI()

        from .precept_mcp_client import LogisticsMCPClient

        project_root = self.server_script.parent.parent.parent

        self.mcp_client = LogisticsMCPClient(
            server_params=StdioServerParameters(
                command="python3",
                args=[str(self.server_script)],
                env={**os.environ, "PYTHONPATH": str(project_root / "src")},
            )
        )
        await self.mcp_client.connect()

        _get_logger().info(
            f"  ✓ Reflexion Baseline connected [{self.strategy.domain_name}] (model: {self.model})"
        )
        _get_logger().info("    • Uses explicit reflection after failures")
        _get_logger().info("    • Same MCP tools as PRECEPT")
        _get_logger().info(
            "    • NO cross-task learning (reflections forgotten between tasks)"
        )

    async def disconnect(self) -> None:
        """Disconnect from server with proper async cleanup."""
        # Close OpenAI client first to release HTTP connections
        if hasattr(self, "llm_client") and self.llm_client:
            try:
                await self.llm_client.close()
            except (RuntimeError, asyncio.CancelledError):
                pass  # Event loop may be closing
            except Exception:
                pass
            finally:
                self.llm_client = None

        # Then close MCP client
        if self.mcp_client:
            try:
                await self.mcp_client.disconnect()
            except (RuntimeError, asyncio.CancelledError):
                pass
            except Exception:
                pass

    async def _ask_llm_with_reflection(
        self,
        task: str,
        parsed_task: ParsedTask,
        memories: str,
        options: List[str],
        previous_attempts: List[Dict[str, str]],
    ) -> Dict[str, Optional[str]]:
        """
        Ask LLM for solution with EXPLICIT REFLECTION on previous failures.

        Uses OpenAI structured outputs for reliable JSON parsing.
        """
        import json

        self.llm_calls += 1

        # Build reflection section
        reflection_section = build_reflection_section(previous_attempts)

        # Build prompt
        prompt = build_reflexion_llm_prompt(
            task=task,
            parsed_task=parsed_task,
            options=options,
            memories=memories,
            reflection_section=reflection_section,
            prompts=self.prompts,
        )

        # Add structured output instruction
        structured_instruction = """

Respond with a JSON object containing:
- "reflection": what went wrong in previous attempts (or null if first attempt)
- "lesson": what to do differently (or null if first attempt)
- "solution": your suggested solution (a single word/phrase)
- "reasoning": why this solution should work
- "confidence": "high", "medium", or "low"
"""
        prompt = prompt + structured_instruction

        if self.verbose:
            _get_logger().debug(
                f"[Reflexion] Calling LLM with {len(previous_attempts)} previous attempts..."
            )

        try:
            # Use structured output for guaranteed JSON format
            structured_params = create_structured_output_params(get_reflexion_schema())

            response = await self.llm_client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a reflective AI that learns from failures within a task. Always respond with valid JSON.",
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=self.config.temperature,
                max_tokens=self.config.reflection_max_tokens,
                **structured_params,
            )

            response_text = response.choices[0].message.content or ""

            if self.verbose:
                _get_logger().debug(f"[Reflexion] Response: {response_text[:200]}...")

            # Parse JSON response
            try:
                parsed = json.loads(response_text)

                result: Dict[str, Optional[str]] = {
                    "solution": None,
                    "reflection": parsed.get("reflection"),
                    "lesson": parsed.get("lesson"),
                    "reasoning": parsed.get("reasoning"),
                    "confidence": parsed.get("confidence"),
                }

                # Match solution to valid options
                suggested = (parsed.get("solution") or "").strip().lower()
                for opt in options:
                    if opt.lower() == suggested:
                        result["solution"] = opt
                        break
                if not result["solution"] and suggested:
                    result["solution"] = suggested

                if result.get("reflection"):
                    self.reflections_generated += 1

                return result

            except json.JSONDecodeError:
                # Fallback to regex parsing if JSON fails
                _get_logger().warning(
                    "[Reflexion] JSON parse failed, falling back to regex"
                )
                result = parse_reflexion_response(response_text, options)
                if result.get("reflection"):
                    self.reflections_generated += 1
                return result

        except Exception as e:
            _get_logger().error(f"[Reflexion] ERROR: {type(e).__name__}: {e}")
            return {"solution": None, "reflection": None, "lesson": None}

    async def run_task(self, task: str) -> Dict[str, Any]:
        """Run task using Reflexion-style reasoning."""
        self.total_tasks += 1
        start_time = time.time()

        if self.verbose:
            _get_logger().debug(f"[Reflexion] Starting run_task for: {task[:50]}...")

        parsed_task = self.strategy.parse_task(task)

        steps = 0
        success = False
        attempts = 0
        previous_attempts: List[Dict[str, str]] = []
        llm_suggestions: List[str] = []
        reflections: List[str] = []

        try:
            query = f"{parsed_task.action} {parsed_task.target or parsed_task.entity}"
            memories = await self.mcp_client.retrieve_memories(query)
            memories_str = str(memories) if memories else ""
            steps += 1

            available_options = self.strategy.get_options_for_task(parsed_task)

            while attempts < self.MAX_ATTEMPTS:
                if self.verbose:
                    _get_logger().debug(
                        f"[Reflexion] Attempt {attempts + 1}/{self.MAX_ATTEMPTS}"
                    )

                llm_result = await self._ask_llm_with_reflection(
                    task=task,
                    parsed_task=parsed_task,
                    memories=memories_str,
                    options=available_options,
                    previous_attempts=previous_attempts,
                )
                steps += 1

                llm_suggestion = llm_result.get("solution")
                reflection = llm_result.get("reflection")

                if reflection:
                    reflections.append(reflection)

                if llm_suggestion and llm_suggestion in available_options:
                    option_to_try = llm_suggestion
                    llm_suggestions.append(llm_suggestion)
                else:
                    option_to_try = random.choice(available_options)

                exec_success, result = await self.strategy.execute_action(
                    self.mcp_client,
                    option_to_try,
                    parsed_task,
                )
                steps += 1
                attempts += 1

                if exec_success:
                    success = True
                    if llm_suggestion == option_to_try:
                        self.llm_suggestions_followed += 1
                    break
                else:
                    previous_attempts.append(
                        {
                            "option": option_to_try,
                            "error": str(result) if result else "Operation failed",
                            "reflection": reflection,
                        }
                    )
                    if llm_suggestion == option_to_try:
                        self.llm_suggestions_failed += 1

        except Exception as e:
            if self.verbose:
                _get_logger().error(f"[Reflexion] Error: {e}")

        duration = time.time() - start_time

        if success:
            self.successful_tasks += 1
        self.steps_per_task.append(steps)

        return {
            "success": success,
            "steps": steps,
            "duration": duration,
            "attempts": attempts,
            "llm_suggestions": llm_suggestions,
            "llm_suggestion": llm_suggestions[0] if llm_suggestions else None,
            "reflections": reflections,
            "reflection_count": len(reflections),
            "domain": self.strategy.domain_name,
        }

    def get_success_rate(self) -> float:
        """Get the current success rate using pure function."""
        return compute_success_rate(self.successful_tasks, self.total_tasks)

    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive statistics using pure functions."""
        stats = build_core_stats(
            total_tasks=self.total_tasks,
            successful_tasks=self.successful_tasks,
            steps_per_task=self.steps_per_task,
            llm_calls=self.llm_calls,
            llm_suggestions_followed=self.llm_suggestions_followed,
            llm_suggestions_failed=self.llm_suggestions_failed,
            domain=self.strategy.domain_name,
            baseline_type="adapted_reflexion",
        )
        # Add reflexion-specific stats
        stats["reflections_generated"] = self.reflections_generated
        stats["reflections_per_task"] = compute_per_task_rate(
            self.reflections_generated, self.total_tasks
        )
        return stats


class FullReflexionBaselineAgent:
    """
    Full Reflexion baseline with CROSS-EPISODE MEMORY (faithful to original paper).

    This baseline implements the FULL Reflexion paper (Shinn et al., 2023):
    1. Reflections are stored in a PERSISTENT memory buffer
    2. Memory persists across EPISODES of the SAME task type
    3. Each new episode can access ALL previous reflections

    Usage:
        reflexion = FullReflexionBaselineAgent(baseline_strategy=LogisticsBaselineStrategy())
        await reflexion.connect()
        result1 = await reflexion.run_task("Book shipment Rotterdam to Boston")
        result2 = await reflexion.run_task("Book shipment Rotterdam to NYC")
    """

    def __init__(
        self,
        baseline_strategy: Optional[BaselineDomainStrategy] = None,
        config: Optional[BaselineConfig] = None,
        model: Optional[str] = None,
        server_script: Optional[Path] = None,
        verbose: Optional[bool] = None,
        max_reflections_per_type: Optional[int] = None,
        max_internal_workers: Optional[int] = None,
        condition_enhanced_retrieval: bool = False,  # Ablation: strict filtering by conditions
        hybrid_retrieval: bool = False,  # Ablation: use BM25 + semantic hybrid retrieval
        improved_baselines: bool = False,  # IMPROVED: strict condition-based filtering
    ):
        """Initialize the Full Reflexion baseline agent.

        Args:
            condition_enhanced_retrieval: If True, strictly filter reflections by conditions (ablation).
                                          This tests if condition-based retrieval fixes the confusion.
                                          Also shows conditions in reflections.
            hybrid_retrieval: If True, use BM25 + semantic hybrid retrieval (ablation).
            improved_baselines: If True, use strict condition-based filtering for O(1)-like lookup.
                                This prioritizes exact condition matches over semantic similarity.
                                Not faithful to the Reflexion paper but tests if structured matching helps.
        """
        if not OPENAI_AVAILABLE:
            raise ImportError(
                "OpenAI library not available. Install with: pip install openai"
            )

        self.config = config or get_default_config().baseline

        if model is not None:
            self.config.model = model
        if verbose is not None:
            self.config.verbose = verbose
        if max_reflections_per_type is not None:
            self.config.max_reflections_per_type = max_reflections_per_type
        if max_internal_workers is not None:
            self.config.max_internal_workers = max_internal_workers

        self.strategy = baseline_strategy or LogisticsBaselineStrategy()
        self.max_reflections = self.config.max_reflections_per_type
        self.condition_enhanced_retrieval = (
            condition_enhanced_retrieval  # Retrieval ablation flag
        )
        self.hybrid_retrieval = hybrid_retrieval  # BM25 + semantic ablation flag
        self.improved_baselines = (
            improved_baselines  # IMPROVED: strict condition matching
        )

        if server_script is None:
            self.server_script = Path(__file__).parent / "precept_mcp_server.py"
        else:
            self.server_script = server_script

        self.mcp_client = None
        self.llm_client: Optional[AsyncOpenAI] = None
        self._internal_semaphore: Optional[asyncio.Semaphore] = None

        # Statistics
        self.total_tasks = 0
        self.successful_tasks = 0
        self.total_episodes = 0
        self.steps_per_task: List[int] = []
        self.llm_calls = 0
        self.reflections_generated = 0
        self.reflections_reused = 0
        self.llm_suggestions_followed = 0
        self.llm_suggestions_failed = 0

        self.prompts = PromptTemplates()

    # =========================================================================
    # CLASS-LEVEL MEMORY MANAGEMENT (delegates to module functions)
    # =========================================================================

    @classmethod
    def get_reflection_memory(cls, task_type: str) -> List[Dict[str, Any]]:
        """Get accumulated reflections for a task type."""
        return get_reflection_memory(task_type)

    @classmethod
    def add_reflection(
        cls,
        task_type: str,
        reflection: Dict[str, Any],
        max_size: int = 20,
        enable_vector_store: bool = False,
    ) -> None:
        """Add a reflection to the memory buffer."""
        add_reflection(task_type, reflection, max_size, enable_vector_store)

    @classmethod
    def clear_memory(cls, task_type: Optional[str] = None) -> None:
        """Clear reflection memory (for testing/reset)."""
        clear_reflection_memory(task_type)

    @classmethod
    def get_memory_stats(cls) -> Dict[str, int]:
        """Get memory statistics."""
        return get_memory_stats()

    # =========================================================================
    # PROPERTY ACCESSORS
    # =========================================================================

    @property
    def model(self) -> str:
        return self.config.model

    @property
    def verbose(self) -> bool:
        return self.config.verbose

    @property
    def max_internal_workers(self) -> int:
        return self.config.max_internal_workers

    @property
    def MAX_ATTEMPTS(self) -> int:
        return self.config.max_attempts

    # =========================================================================
    # CONNECTION METHODS
    # =========================================================================

    async def connect(self) -> None:
        """Connect to MCP server and initialize LLM client."""
        if not MCP_AVAILABLE:
            raise ImportError("MCP library not available.")

        self._internal_semaphore = asyncio.Semaphore(self.max_internal_workers)
        self.llm_client = AsyncOpenAI()

        from .precept_mcp_client import LogisticsMCPClient

        project_root = self.server_script.parent.parent.parent

        self.mcp_client = LogisticsMCPClient(
            server_params=StdioServerParameters(
                command="python3",
                args=[str(self.server_script)],
                env={**os.environ, "PYTHONPATH": str(project_root / "src")},
            )
        )
        await self.mcp_client.connect()

        _get_logger().info(
            f"  ✓ Full Reflexion Baseline connected [{self.strategy.domain_name}]"
        )
        _get_logger().info(
            f"    Internal concurrency: {self.max_internal_workers} workers"
        )
        _get_logger().info("    • Cross-episode memory (same task type)")
        _get_logger().info("    • Verbal reflections persist across episodes")
        _get_logger().info("    • NO rule compilation, NO deterministic pruning")
        _get_logger().info(f"    • Memory stats: {self.get_memory_stats()}")

    async def disconnect(self) -> None:
        """Disconnect from server with proper async cleanup."""
        # Close OpenAI client first to release HTTP connections
        if hasattr(self, "llm_client") and self.llm_client:
            try:
                await self.llm_client.close()
            except (RuntimeError, asyncio.CancelledError):
                pass  # Event loop may be closing
            except Exception:
                pass
            finally:
                self.llm_client = None

        # Then close MCP client
        if self.mcp_client:
            try:
                await self.mcp_client.disconnect()
            except (RuntimeError, asyncio.CancelledError):
                pass
            except Exception:
                pass

    # =========================================================================
    # LLM REASONING
    # =========================================================================

    async def _ask_llm_with_full_reflexion(
        self,
        task: str,
        parsed_task: ParsedTask,
        memories: str,
        options: List[str],
        task_type: str,
        current_attempts: List[Dict[str, str]],
        conditions: Optional[List[str]] = None,  # For condition_aware ablation
    ) -> Dict[str, Optional[str]]:
        """Ask LLM with FULL Reflexion (accumulated cross-episode reflections)."""
        self.llm_calls += 1

        # Get accumulated reflections (pass condition_enhanced_retrieval for both filtering and display)
        # Also pass hybrid_retrieval for BM25 + condition matching ablation
        # Pass improved_baselines for strict condition-based O(1)-like lookup
        accumulated = format_accumulated_reflections(
            task_type,
            condition_aware=self.condition_enhanced_retrieval,  # Display conditions
            current_conditions=conditions
            if (
                self.condition_enhanced_retrieval
                or self.hybrid_retrieval
                or self.improved_baselines
            )
            else None,
            condition_enhanced_retrieval=self.condition_enhanced_retrieval,  # Filter by conditions
            hybrid_retrieval=self.hybrid_retrieval,  # BM25 + condition matching
            task_description=task,  # For BM25 relevance scoring
            improved_baselines=self.improved_baselines,  # IMPROVED: strict condition matching
        )
        has_prior_reflections = (
            accumulated
            != "No previous reflections. This is your first episode of this task type."
        )
        if has_prior_reflections:
            self.reflections_reused += 1

        # Build current episode context
        current_context = build_current_episode_context(current_attempts)

        # Build prompt
        prompt = build_full_reflexion_llm_prompt(
            task=task,
            parsed_task=parsed_task,
            options=options,
            task_type=task_type,
            accumulated_reflections=accumulated,
            current_episode_context=current_context,
            prompts=self.prompts,
            conditions=conditions,
            include_options_conditions=self.improved_baselines,
        )

        # ═══════════════════════════════════════════════════════════════════════
        # DETAILED INVESTIGATION LOGGING - Full Reflexion Behavior Analysis
        # ═══════════════════════════════════════════════════════════════════════
        if self.verbose:
            _get_logger().info("═" * 70)
            _get_logger().info("🔍 FULL REFLEXION INVESTIGATION")
            _get_logger().info("═" * 70)
            _get_logger().info(f"📋 TASK: {task[:100]}...")
            _get_logger().info(f"📂 TASK TYPE: {task_type}")
            _get_logger().info(f"🎯 AVAILABLE OPTIONS: {options}")
            _get_logger().info(f"📚 HAS PRIOR REFLECTIONS: {has_prior_reflections}")
            _get_logger().info(
                f"📝 NUM ACCUMULATED REFLECTIONS: {len(self.get_reflection_memory(task_type))}"
            )
            _get_logger().info("─" * 70)
            _get_logger().info("📖 ACCUMULATED REFLECTIONS (what LLM sees):")
            for line in accumulated.split("\n")[:15]:  # Limit to 15 lines
                _get_logger().info(f"   {line}")
            _get_logger().info("─" * 70)
            if current_context:
                _get_logger().info(
                    "⚠️ CURRENT EPISODE CONTEXT (failed attempts this episode):"
                )
                _get_logger().info(f"   {current_context[:200]}...")
            _get_logger().info("─" * 70)

        # Add structured output instruction to prompt
        structured_instruction = """

Respond with a JSON object containing:
- "reflection": what patterns do you notice from past experiences
- "lesson": key insight for this and future episodes
- "solution": your suggested solution (a single word/phrase from available options)
- "reasoning": why this solution based on reflections
- "confidence": "high", "medium", or "low"
"""
        prompt = prompt + structured_instruction

        try:
            import json

            # Use structured output for guaranteed JSON format
            structured_params = create_structured_output_params(
                get_full_reflexion_schema()
            )

            response = await self.llm_client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a reflective AI that learns from past episodes. Always respond with valid JSON.",
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=self.config.temperature,
                max_tokens=self.config.full_reflexion_max_tokens,
                **structured_params,
            )

            response_text = response.choices[0].message.content or ""

            # Parse JSON response
            try:
                parsed = json.loads(response_text)

                parsed_result: Dict[str, Optional[str]] = {
                    "solution": None,
                    "reflection": parsed.get("reflection"),
                    "lesson": parsed.get("lesson"),
                    "reasoning": parsed.get("reasoning"),
                    "confidence": parsed.get("confidence"),
                }

                # Match solution to valid options
                suggested = (parsed.get("solution") or "").strip().lower()
                for opt in options:
                    if opt.lower() == suggested:
                        parsed_result["solution"] = opt
                        break
                if not parsed_result["solution"] and suggested:
                    parsed_result["solution"] = suggested

            except json.JSONDecodeError:
                # Fallback to regex parsing if JSON fails
                _get_logger().warning(
                    "[FullReflexion] JSON parse failed, falling back to regex"
                )
                parsed_result = parse_reflexion_response(response_text, options)

            # ═══════════════════════════════════════════════════════════════════════
            # DETAILED INVESTIGATION LOGGING - LLM Response Analysis
            # ═══════════════════════════════════════════════════════════════════════
            if self.verbose:
                _get_logger().info("🤖 LLM RESPONSE ANALYSIS:")
                _get_logger().info(
                    f"   Raw Response (first 300 chars): {response_text[:300]}..."
                )
                _get_logger().info("─" * 70)
                _get_logger().info(
                    f"   🎯 PARSED SOLUTION: {parsed_result.get('solution')}"
                )
                _get_logger().info(
                    f"   💭 PARSED REFLECTION: {parsed_result.get('reflection', 'None')[:100] if parsed_result.get('reflection') else 'None'}..."
                )
                _get_logger().info(
                    f"   📚 PARSED LESSON: {parsed_result.get('lesson', 'None')[:100] if parsed_result.get('lesson') else 'None'}..."
                )
                _get_logger().info("─" * 70)
                solution = parsed_result.get("solution")
                if solution and solution in options:
                    _get_logger().info(
                        f"   ✅ SOLUTION '{solution}' IS VALID (in options)"
                    )
                elif solution:
                    _get_logger().info(
                        f"   ❌ SOLUTION '{solution}' IS INVALID (not in options: {options})"
                    )
                else:
                    _get_logger().info(
                        f"   ⚠️ NO SOLUTION PARSED - will use RANDOM from {options}"
                    )
                _get_logger().info("═" * 70)

            return parsed_result

        except Exception as e:
            _get_logger().error(f"[FullReflexion] ERROR: {type(e).__name__}: {e}")
            return {"solution": None, "reflection": None, "lesson": None}

    # =========================================================================
    # TASK EXECUTION
    # =========================================================================

    async def run_task(
        self,
        task: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Run task using FULL Reflexion with cross-episode memory.

        Args:
            task: The task string to execute
            metadata: Optional dict with scenario metadata (e.g., condition_key for multi-condition)
        """
        self.total_tasks += 1
        self.total_episodes += 1
        start_time = time.time()

        parsed_task = self.strategy.parse_task(task)

        # ═══════════════════════════════════════════════════════════════════
        # INJECT MULTI-CONDITION METADATA (if provided)
        # This allows condition_key to be passed from scenario without being
        # visible in task description (fair comparison with PRECEPT)
        # ═══════════════════════════════════════════════════════════════════
        if metadata:
            if parsed_task.parameters is None:
                parsed_task.parameters = {}
            # Inject condition_key from metadata (for multi-condition enforcement)
            if "condition_key" in metadata and metadata["condition_key"]:
                parsed_task.parameters["condition_key"] = metadata["condition_key"]
            elif "conditions" in metadata and metadata["conditions"]:
                # BUGFIX: Derive condition_key from conditions list when not
                # explicitly provided (mirrors fix in precept_agent.py)
                conditions_list = metadata["conditions"]
                parsed_task.parameters["condition_key"] = "+".join(
                    sorted(str(c) for c in conditions_list)
                )
            # Inject conditions list if available
            if "conditions" in metadata:
                parsed_task.parameters["conditions"] = metadata["conditions"]

        task_type = f"{self.strategy.domain_name}:{parsed_task.action}"

        # Extract conditions for ablation mode
        conditions: List[str] = []
        if metadata and "conditions" in metadata:
            conditions = metadata["conditions"]
        elif hasattr(parsed_task, "parameters") and parsed_task.parameters:
            conditions = parsed_task.parameters.get("conditions", [])
        if not conditions:
            # Fallback: extract from task string
            conditions = extract_conditions_from_task(task)

        if self.verbose:
            _get_logger().debug(f"[FullReflexion] Task type: {task_type}")
            _get_logger().debug(
                f"[FullReflexion] Accumulated reflections: {len(self.get_reflection_memory(task_type))}"
            )
            if self.condition_enhanced_retrieval:
                _get_logger().debug(
                    f"[FullReflexion] Condition-enhanced mode: conditions={conditions}"
                )

        steps = 0
        success = False
        attempts = 0
        current_attempts: List[Dict[str, str]] = []
        llm_suggestions: List[str] = []
        reflections: List[str] = []
        failed_options: List[str] = []
        successful_option: Optional[str] = None
        llm_result: Dict[str, Optional[str]] = {}

        try:
            query = f"{parsed_task.action} {parsed_task.target or parsed_task.entity}"
            memories = await self.mcp_client.retrieve_memories(query)
            memories_str = str(memories) if memories else ""
            steps += 1

            available_options = self.strategy.get_options_for_task(parsed_task)

            while attempts < self.MAX_ATTEMPTS:
                if self.verbose:
                    _get_logger().debug(
                        f"[FullReflexion] Attempt {attempts + 1}/{self.MAX_ATTEMPTS}"
                    )

                llm_result = await self._ask_llm_with_full_reflexion(
                    task=task,
                    parsed_task=parsed_task,
                    memories=memories_str,
                    options=available_options,
                    task_type=task_type,
                    current_attempts=current_attempts,
                    conditions=conditions,  # Pass conditions for ablation mode
                )
                steps += 1

                llm_suggestion = llm_result.get("solution")
                reflection = llm_result.get("reflection")

                if reflection:
                    reflections.append(reflection)
                    self.reflections_generated += 1

                if llm_suggestion and llm_suggestion in available_options:
                    option_to_try = llm_suggestion
                    llm_suggestions.append(llm_suggestion)
                    used_llm_suggestion = True
                else:
                    option_to_try = random.choice(available_options)
                    used_llm_suggestion = False

                # ═══════════════════════════════════════════════════════════════════════
                # DETAILED INVESTIGATION LOGGING - Action Selection
                # ═══════════════════════════════════════════════════════════════════════
                if self.verbose:
                    _get_logger().info("⚡ ACTION SELECTION:")
                    if used_llm_suggestion:
                        _get_logger().info(
                            f"   📌 USING LLM SUGGESTION: {option_to_try}"
                        )
                    else:
                        _get_logger().info(
                            f"   🎲 USING RANDOM CHOICE: {option_to_try} (LLM suggestion '{llm_suggestion}' was invalid or None)"
                        )

                exec_success, result = await self.strategy.execute_action(
                    self.mcp_client,
                    option_to_try,
                    parsed_task,
                )
                steps += 1
                attempts += 1

                # ═══════════════════════════════════════════════════════════════════════
                # DETAILED INVESTIGATION LOGGING - Execution Result
                # ═══════════════════════════════════════════════════════════════════════
                if self.verbose:
                    if exec_success:
                        _get_logger().info(
                            f"   ✅ SUCCESS with '{option_to_try}' on attempt {attempts}/{self.MAX_ATTEMPTS}"
                        )
                        if used_llm_suggestion:
                            _get_logger().info(
                                f"   🎯 LLM CORRECTLY PREDICTED: {option_to_try}"
                            )
                    else:
                        _get_logger().info(
                            f"   ❌ FAILED with '{option_to_try}' on attempt {attempts}/{self.MAX_ATTEMPTS}"
                        )
                        _get_logger().info(f"   📝 Error: {str(result)[:100]}...")
                        if used_llm_suggestion:
                            _get_logger().info(
                                f"   ⚠️ LLM SUGGESTION WAS WRONG: {llm_suggestion}"
                            )
                    _get_logger().info("─" * 70)

                if exec_success:
                    success = True
                    successful_option = option_to_try
                    if llm_suggestion == option_to_try:
                        self.llm_suggestions_followed += 1
                    break
                else:
                    failed_options.append(option_to_try)
                    current_attempts.append(
                        {
                            "option": option_to_try,
                            "error": str(result) if result else "Operation failed",
                        }
                    )
                    if llm_suggestion == option_to_try:
                        self.llm_suggestions_failed += 1

        except Exception as e:
            if self.verbose:
                _get_logger().error(f"[FullReflexion] Error: {e}")

        duration = time.time() - start_time

        # Store reflection for future episodes
        final_reflection = reflections[-1] if reflections else None
        reflection_record = create_reflection_record(
            episode=self.total_episodes,
            task=task,
            success=success,
            reflection=final_reflection,
            lesson=llm_result.get("lesson") if llm_result else None,
            failed_options=failed_options,
            successful_option=successful_option,
            attempts=attempts,
            conditions=conditions,  # Store conditions for ablation mode
        )
        # Enable vector store when hybrid_retrieval is active (for BM25+Vector ensemble)
        self.add_reflection(
            task_type,
            reflection_record,
            self.max_reflections,
            enable_vector_store=self.hybrid_retrieval,  # Create embeddings for hybrid retrieval
        )

        if success:
            self.successful_tasks += 1
        self.steps_per_task.append(steps)

        # ═══════════════════════════════════════════════════════════════════════
        # DETAILED INVESTIGATION LOGGING - Task Summary
        # ═══════════════════════════════════════════════════════════════════════
        if self.verbose:
            _get_logger().info("╔" + "═" * 68 + "╗")
            _get_logger().info("║" + " FULL REFLEXION TASK SUMMARY ".center(68) + "║")
            _get_logger().info("╠" + "═" * 68 + "╣")
            _get_logger().info(f"║ Task: {task[:60]}...".ljust(69) + "║")
            _get_logger().info(f"║ Task Type: {task_type}".ljust(69) + "║")
            _get_logger().info(f"║ SUCCESS: {success}".ljust(69) + "║")
            _get_logger().info(f"║ Steps: {steps}".ljust(69) + "║")
            _get_logger().info(
                f"║ Attempts: {attempts}/{self.MAX_ATTEMPTS}".ljust(69) + "║"
            )
            _get_logger().info(
                f"║ First-Try Success: {attempts == 1 and success}".ljust(69) + "║"
            )
            _get_logger().info(f"║ Failed Options: {failed_options}".ljust(69) + "║")
            _get_logger().info(
                f"║ Successful Option: {successful_option}".ljust(69) + "║"
            )
            _get_logger().info(
                f"║ LLM Suggestions Made: {llm_suggestions}".ljust(69) + "║"
            )
            _get_logger().info(
                f"║ Accumulated Reflections After: {len(self.get_reflection_memory(task_type))}".ljust(
                    69
                )
                + "║"
            )
            _get_logger().info("╠" + "═" * 68 + "╣")
            if not success or attempts > 1:
                _get_logger().info("║" + " 🔍 WHY FIRST-TRY FAILED? ".center(68) + "║")
                if llm_suggestions:
                    _get_logger().info(
                        f"║ LLM suggested '{llm_suggestions[0]}' but it was WRONG".ljust(
                            69
                        )
                        + "║"
                    )
                else:
                    _get_logger().info(
                        "║ LLM provided NO valid suggestion → used RANDOM".ljust(69)
                        + "║"
                    )
                _get_logger().info(
                    "║ Reflections may not have matched the multi-condition key".ljust(
                        69
                    )
                    + "║"
                )
            _get_logger().info("╚" + "═" * 68 + "╝")
            _get_logger().info("")

        return {
            "success": success,
            "steps": steps,
            "duration": duration,
            "attempts": attempts,
            "first_try": success and attempts == 1,  # Success without error recovery
            "llm_suggestions": llm_suggestions,
            "llm_suggestion": llm_suggestions[0] if llm_suggestions else None,
            "reflections": reflections,
            "reflection_count": len(reflections),
            "accumulated_reflections": len(self.get_reflection_memory(task_type)),
            "task_type": task_type,
            "domain": self.strategy.domain_name,
        }

    # =========================================================================
    # STATISTICS
    # =========================================================================

    def get_success_rate(self) -> float:
        """Get the current success rate using pure function."""
        return compute_success_rate(self.successful_tasks, self.total_tasks)

    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive statistics using pure functions."""
        stats = build_core_stats(
            total_tasks=self.total_tasks,
            successful_tasks=self.successful_tasks,
            steps_per_task=self.steps_per_task,
            llm_calls=self.llm_calls,
            llm_suggestions_followed=self.llm_suggestions_followed,
            llm_suggestions_failed=self.llm_suggestions_failed,
            domain=self.strategy.domain_name,
            baseline_type="full_reflexion",
        )
        # Add full reflexion-specific stats
        stats["total_episodes"] = self.total_episodes
        stats["reflections_generated"] = self.reflections_generated
        stats["reflections_reused"] = self.reflections_reused
        stats["memory_stats"] = self.get_memory_stats()
        return stats


class ExpeL_BaselineAgent:
    """
    ExpeL (Experiential Learning) baseline - Zhao et al., 2023.

    Faithful implementation of "ExpeL: LLM Agents Are Experiential Learners"
    https://arxiv.org/abs/2308.10144

    ═══════════════════════════════════════════════════════════════════════════
    KEY PRINCIPLES FROM THE PAPER:
    ═══════════════════════════════════════════════════════════════════════════
    1. Learn from BOTH successes AND failures (not just failures like Reflexion)
    2. Extract GENERALIZABLE insights in natural language
    3. Use SIMILARITY-BASED retrieval for relevant past experiences
    4. Apply insights to new tasks with similar characteristics

    ═══════════════════════════════════════════════════════════════════════════
    ARCHITECTURE (faithful to paper):
    ═══════════════════════════════════════════════════════════════════════════
    1. EXPERIENCE COLLECTION: Execute tasks, record full trajectories
    2. INSIGHT EXTRACTION: Use LLM to extract generalizable patterns from trajectories
    3. INSIGHT STORAGE: Store insights with condition codes and trajectory metadata
    4. RETRIEVAL: Find relevant insights via condition similarity (Jaccard + exact match priority)

    ═══════════════════════════════════════════════════════════════════════════
    FAIR COMPARISON GUARANTEE:
    ═══════════════════════════════════════════════════════════════════════════
    - Same LLM (GPT-4o-mini) as PRECEPT and other baselines
    - Same MCP tools for task execution
    - Same retry budget (MAX_ATTEMPTS)
    - Same training/testing protocol

    ═══════════════════════════════════════════════════════════════════════════
    KEY THEORETICAL DIFFERENCE FROM PRECEPT:
    ═══════════════════════════════════════════════════════════════════════════
    - ExpeL:   Stores natural language insights, retrieves via SIMILARITY MATCHING
    - PRECEPT: Stores structured rules, retrieves via EXACT COMPOSITE KEY HASH

    Why ExpeL should degrade with multi-conditions (n):
    - Similarity matching may retrieve insights with PARTIAL condition overlap
    - LLM must interpret multiple conditions and find exact matches
    - Error rate compounds: P(all_conditions_matched) ≈ 0.70^n

    Why PRECEPT maintains ~100% with multi-conditions:
    - Composite key lookup is DETERMINISTIC (hash-based)
    - Exact match guaranteed regardless of condition count
    - No interpretation needed: condition_key → solution directly

    ═══════════════════════════════════════════════════════════════════════════
    USAGE:
    ═══════════════════════════════════════════════════════════════════════════

        expel = ExpeL_BaselineAgent(baseline_strategy=LogisticsBaselineStrategy())
        await expel.connect()

        # Training phase: Execute tasks and extract insights
        for task in training_tasks:
            result = await expel.run_task(task, training=True)
            # Insights are automatically extracted and stored

        # Testing phase: Retrieve and apply learned insights
        for task in test_tasks:
            result = await expel.run_task(task, training=False)
            # Relevant insights are retrieved and applied
    """

    def __init__(
        self,
        baseline_strategy: Optional[BaselineDomainStrategy] = None,
        config: Optional[BaselineConfig] = None,
        model: Optional[str] = None,
        server_script: Optional[Path] = None,
        verbose: Optional[bool] = None,
        max_internal_workers: Optional[int] = None,
        top_k_insights: int = 10,  # Standardized to match other methods
        condition_enhanced_retrieval: bool = False,  # Ablation: use conditions in retrieval + display
        hybrid_retrieval: bool = False,  # Ablation: use BM25 + semantic hybrid retrieval
        improved_baselines: bool = False,  # IMPROVED: metadata-based filtering for O(1) lookup
    ):
        """Initialize the ExpeL baseline agent.

        Args:
            condition_enhanced_retrieval: If True, include conditions in vector search (ablation).
                                          Tests if embeddings can distinguish conditions.
                                          Also shows conditions in insights.
            hybrid_retrieval: If True, use BM25 + semantic hybrid retrieval (ablation).
                              Tests if hybrid retrieval improves performance.
            improved_baselines: If True, store conditions as metadata and use pre-filtering.
                                This gives O(1)-like lookup using ChromaDB's 'where' clause.
                                Not faithful to ExpeL paper but tests if structured filtering helps.
        """
        if not OPENAI_AVAILABLE:
            raise ImportError(
                "OpenAI library not available. Install with: pip install openai"
            )

        self.config = config or get_default_config().baseline

        if model is not None:
            self.config.model = model
        if verbose is not None:
            self.config.verbose = verbose
        if max_internal_workers is not None:
            self.config.max_internal_workers = max_internal_workers

        self.strategy = baseline_strategy or LogisticsBaselineStrategy()
        self.top_k_insights = top_k_insights
        self.condition_enhanced_retrieval = condition_enhanced_retrieval
        self.hybrid_retrieval = hybrid_retrieval
        self.improved_baselines = improved_baselines  # IMPROVED: metadata filtering

        if server_script is None:
            self.server_script = Path(__file__).parent / "precept_mcp_server.py"
        else:
            self.server_script = server_script

        self.mcp_client = None
        self.llm_client: Optional[AsyncOpenAI] = None
        self._internal_semaphore: Optional[asyncio.Semaphore] = None

        # Statistics
        self.total_tasks = 0
        self.successful_tasks = 0
        self.total_episodes = 0
        self.steps_per_task: List[int] = []
        self.llm_calls = 0
        self.llm_suggestions_followed = 0
        self.llm_suggestions_failed = 0

        # ExpeL-specific stats
        self.insights_extracted = 0
        self.insights_retrieved = 0
        self.insights_applied = 0

        self.prompts = PromptTemplates()

    # =========================================================================
    # CLASS-LEVEL INSIGHT MANAGEMENT
    # =========================================================================

    @classmethod
    def get_insights(cls) -> List[Dict[str, Any]]:
        """Get all stored insights."""
        return get_expel_insights()

    @classmethod
    def add_insight(cls, insight: Dict[str, Any]) -> None:
        """Add an insight to storage."""
        add_expel_insight(insight)

    @classmethod
    def clear_insights(cls) -> None:
        """Clear all insights (for testing/reset)."""
        clear_expel_insights()

    @classmethod
    def get_insight_stats(cls) -> Dict[str, Any]:
        """Get insight statistics."""
        return get_expel_stats()

    # =========================================================================
    # PROPERTY ACCESSORS
    # =========================================================================

    @property
    def model(self) -> str:
        return self.config.model

    @property
    def verbose(self) -> bool:
        return self.config.verbose

    @property
    def max_internal_workers(self) -> int:
        return self.config.max_internal_workers

    @property
    def MAX_ATTEMPTS(self) -> int:
        return self.config.max_attempts

    # =========================================================================
    # CONNECTION METHODS
    # =========================================================================

    async def connect(self) -> None:
        """Connect to MCP server and initialize LLM client."""
        if not MCP_AVAILABLE:
            raise ImportError("MCP library not available.")

        self._internal_semaphore = asyncio.Semaphore(self.max_internal_workers)
        self.llm_client = AsyncOpenAI()

        from .precept_mcp_client import LogisticsMCPClient

        project_root = self.server_script.parent.parent.parent

        self.mcp_client = LogisticsMCPClient(
            server_params=StdioServerParameters(
                command="python3",
                args=[str(self.server_script)],
                env={**os.environ, "PYTHONPATH": str(project_root / "src")},
            )
        )
        await self.mcp_client.connect()

        _get_logger().info(
            f"  ✓ ExpeL Baseline connected [{self.strategy.domain_name}]"
        )
        _get_logger().info(
            f"    Internal concurrency: {self.max_internal_workers} workers"
        )
        _get_logger().info("    • Experiential Learning (Zhao et al., 2023)")
        _get_logger().info("    • Extracts generalizable insights from experiences")
        _get_logger().info("    • Retrieves insights by condition similarity")
        _get_logger().info(f"    • Top-k insights: {self.top_k_insights}")
        _get_logger().info(f"    • Insight store: {self.get_insight_stats()}")

    async def disconnect(self) -> None:
        """Disconnect from server with proper async cleanup."""
        if hasattr(self, "llm_client") and self.llm_client:
            try:
                await self.llm_client.close()
            except (RuntimeError, asyncio.CancelledError):
                pass
            except Exception:
                pass
            finally:
                self.llm_client = None

        if self.mcp_client:
            try:
                await self.mcp_client.disconnect()
            except (RuntimeError, asyncio.CancelledError):
                pass
            except Exception:
                pass

    # =========================================================================
    # INSIGHT EXTRACTION (ExpeL Core Component)
    # =========================================================================

    async def _extract_insight(
        self,
        task: str,
        success: bool,
        attempts: int,
        failed_options: List[str],
        successful_option: Optional[str],
        conditions: List[str],
        errors: Optional[List[str]] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Extract a generalizable insight from a task execution.

        This is the core of ExpeL - after each task, we ask the LLM to
        extract a reusable insight that can help with similar future tasks.

        Uses OpenAI structured outputs for reliable JSON parsing.

        Args:
            task: The task string
            success: Whether the task succeeded
            attempts: Number of attempts
            failed_options: Options that failed
            successful_option: The successful option (if any)
            conditions: Conditions present in the task
            errors: Error messages received

        Returns:
            Extracted insight dictionary or None
        """
        import json

        self.llm_calls += 1

        prompt = build_expel_insight_extraction_prompt(
            task=task,
            success=success,
            attempts=attempts,
            failed_options=failed_options,
            successful_option=successful_option,
            conditions=conditions,
            errors=errors,
            prompts=self.prompts,
            improved_baselines=self.improved_baselines,  # Use PRECEPT-like prompts
        )

        # Add structured output instruction
        structured_instruction = """

Respond with a JSON object containing:
- "insight": the generalizable pattern or rule learned from this experience
- "conditions_covered": list of condition codes this applies to (e.g., ["R-482", "C-HZMT"])
- "solution": the working solution (for success) or null (for failure)
- "avoid": list of options to avoid (for failure) or empty list (for success)
- "confidence": "high", "medium", or "low"
"""
        prompt = prompt + structured_instruction

        if self.verbose:
            _get_logger().info("═" * 70)
            _get_logger().info("🔬 ExpeL INSIGHT EXTRACTION")
            _get_logger().info("═" * 70)
            _get_logger().info(f"   Task: {task[:80]}...")
            _get_logger().info(f"   Success: {success}")
            _get_logger().info(f"   Conditions: {conditions}")

        try:
            # Use structured output for guaranteed JSON format
            structured_params = create_structured_output_params(
                get_expel_insight_schema()
            )

            response = await self.llm_client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are an AI that extracts generalizable insights from task executions. Always respond with valid JSON.",
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=self.config.temperature,
                max_tokens=self.config.reflection_max_tokens,
                **structured_params,
            )

            response_text = response.choices[0].message.content or ""

            # Parse JSON response
            try:
                parsed = json.loads(response_text)

                insight: Dict[str, Any] = {
                    "insight": parsed.get("insight"),
                    "conditions": parsed.get("conditions_covered", []),
                    "solution": parsed.get("solution"),
                    "avoid": parsed.get("avoid", []),
                    "confidence": parsed.get("confidence", "low"),
                    "type": "success" if success else "failure",
                }

            except json.JSONDecodeError:
                # Fallback to regex parsing if JSON fails
                _get_logger().warning(
                    "[ExpeL] JSON parse failed, falling back to regex"
                )
                insight = parse_expel_insight_response(response_text, success)

            if insight.get("insight"):
                self.insights_extracted += 1

                if self.verbose:
                    _get_logger().info("─" * 70)
                    _get_logger().info(
                        f"   ✅ Extracted insight: {insight.get('insight', '')[:100]}..."
                    )
                    _get_logger().info(
                        f"   Conditions covered: {insight.get('conditions', [])}"
                    )
                    if success:
                        _get_logger().info(f"   Solution: {insight.get('solution')}")
                    else:
                        _get_logger().info(f"   Avoid: {insight.get('avoid', [])}")
                    _get_logger().info("═" * 70)

                return insight

        except Exception as e:
            _get_logger().error(
                f"[ExpeL] Insight extraction error: {type(e).__name__}: {e}"
            )

        return None

    # =========================================================================
    # LLM REASONING WITH INSIGHTS
    # =========================================================================

    async def _ask_llm_with_insights(
        self,
        task: str,
        parsed_task: Any,
        options: List[str],
        conditions: List[str],
        current_attempts: List[Dict[str, str]],
    ) -> Dict[str, Optional[str]]:
        """
        Ask LLM for a solution using retrieved insights.

        Args:
            task: The task string
            parsed_task: Parsed task information
            options: Available options
            conditions: Conditions in the task
            current_attempts: Failed attempts in current episode

        Returns:
            Dictionary with solution, insight_applied, reasoning, confidence
        """
        self.llm_calls += 1

        # Retrieve relevant insights
        # Choose retrieval method based on mode:
        # - IMPROVED: Metadata filtering with O(1)-like condition lookup (highest priority)
        # - Hybrid + Condition-enhanced: Combined BM25 + condition-enhanced embeddings
        # - Hybrid only: BM25 + semantic (ablation to test if hybrid helps)
        # - Condition-enhanced only: Include conditions in vector search
        # - Default: Vector similarity only (per ExpeL paper)
        if self.improved_baselines:
            # IMPROVED BASELINES: Use metadata-filtered retrieval for O(1)-like lookup
            # When hybrid_retrieval is also True, use BM25+vector for ranking within filtered set
            from .baseline_functions import retrieve_expel_insights_with_metadata_filter

            retrieved_insights = retrieve_expel_insights_with_metadata_filter(
                task=task,
                task_conditions=conditions,
                top_k=self.top_k_insights,
                hybrid_retrieval=self.hybrid_retrieval,  # Enable BM25+vector ranking
            )
        elif self.hybrid_retrieval:
            # Use hybrid BM25 + semantic retrieval
            # When condition_enhanced is also True, include conditions in the query
            from .baseline_functions import retrieve_expel_insights_hybrid

            retrieved_insights = retrieve_expel_insights_hybrid(
                task=task,
                task_conditions=conditions,
                top_k=self.top_k_insights,
                condition_enhanced=self.condition_enhanced_retrieval,  # Combined mode
            )
        else:
            # Default: Vector similarity only (per ExpeL paper)
            # condition_enhanced affects whether conditions are embedded in query
            retrieved_insights = retrieve_expel_insights_by_task(
                task=task,
                task_conditions=conditions,
                top_k=self.top_k_insights,
                condition_enhanced=self.condition_enhanced_retrieval,  # Ablation mode
            )

        # Track retrieval statistics
        self.insights_retrieved += len(retrieved_insights)

        # Build current episode context
        current_context = ""
        if current_attempts:
            lines = [
                "\n═══════════════════════════════════════════════════════════════════════════════",
                "CURRENT EPISODE ATTEMPTS:",
            ]
            for i, attempt in enumerate(current_attempts, 1):
                lines.append(f"\n  Attempt {i}:")
                lines.append(f"    Option: {attempt['option']}")
                lines.append(f"    Error: {attempt['error']}")
            lines.append("\n⚠️ Choose a DIFFERENT option!")
            lines.append(
                "═══════════════════════════════════════════════════════════════════════════════"
            )
            current_context = "\n".join(lines)

        # Build prompt (pass condition_enhanced_retrieval for ablation mode)
        prompt = build_expel_task_prompt(
            task=task,
            parsed_task=parsed_task,
            options=options,
            conditions=conditions,
            insights=retrieved_insights,
            current_episode_context=current_context,
            prompts=self.prompts,
            condition_aware=(
                self.condition_enhanced_retrieval or self.improved_baselines
            ),  # Show conditions in insights for improved baselines
            include_options_conditions=self.improved_baselines,
        )

        # Add structured output instruction
        structured_instruction = """

Respond with a JSON object containing:
- "solution": your suggested solution (a single word/phrase from available options)
- "insight_applied": which insight (if any) informed this decision, or null
- "reasoning": why this solution was chosen based on insights
- "confidence": "high", "medium", or "low"
"""
        prompt = prompt + structured_instruction

        if self.verbose:
            _get_logger().info("═" * 70)
            _get_logger().info("🎯 ExpeL TASK EXECUTION")
            _get_logger().info("═" * 70)
            _get_logger().info(f"   Task: {task[:80]}...")
            _get_logger().info(f"   Conditions: {conditions}")
            _get_logger().info(f"   Retrieved {len(retrieved_insights)} insights")
            for i, insight in enumerate(retrieved_insights[:3], 1):
                _get_logger().info(
                    f"   Insight {i}: {insight.get('insight', '')[:60]}..."
                )
            _get_logger().info("─" * 70)

        try:
            import json

            # Use structured output for guaranteed JSON format
            structured_params = create_structured_output_params(get_expel_task_schema())

            response = await self.llm_client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are an AI agent that learns from past experiences. Always respond with valid JSON.",
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
                **structured_params,
            )

            response_text = response.choices[0].message.content or ""

            # Parse JSON response
            try:
                parsed = json.loads(response_text)

                result: Dict[str, Optional[str]] = {
                    "solution": None,
                    "insight_applied": parsed.get("insight_applied"),
                    "reasoning": parsed.get("reasoning"),
                    "confidence": parsed.get("confidence", "low"),
                }

                # Match solution to valid options
                suggested = (parsed.get("solution") or "").strip().lower()
                for opt in options:
                    if opt.lower() == suggested:
                        result["solution"] = opt
                        break
                if not result["solution"] and suggested:
                    result["solution"] = suggested

            except json.JSONDecodeError:
                # Fallback to regex parsing if JSON fails
                _get_logger().warning(
                    "[ExpeL] JSON parse failed, falling back to regex"
                )
                result = parse_expel_task_response(response_text, options)

            if self.verbose:
                _get_logger().info(f"   LLM Response: {response_text[:150]}...")
                _get_logger().info(f"   Parsed solution: {result.get('solution')}")
                _get_logger().info(
                    f"   Insight applied: {result.get('insight_applied')}"
                )
                _get_logger().info("═" * 70)

            # Track if insight was applied
            insight_applied = result.get("insight_applied")
            if insight_applied and insight_applied.lower() not in ["none", "n/a", ""]:
                self.insights_applied += 1

            return result

        except Exception as e:
            _get_logger().error(f"[ExpeL] ERROR: {type(e).__name__}: {e}")
            return {
                "solution": None,
                "insight_applied": None,
                "reasoning": None,
                "confidence": "low",
            }

    # =========================================================================
    # TASK EXECUTION
    # =========================================================================

    async def run_task(
        self,
        task: str,
        training: bool = True,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Run task using ExpeL methodology.

        Args:
            task: The task string to execute
            training: If True, extract insights after execution (for learning)
                     If False, only retrieve and apply existing insights (for testing)
            metadata: Optional dict with scenario metadata (e.g., condition_key for multi-condition)

        Returns:
            Dict with success, steps, duration, attempts, insights, domain
        """
        self.total_tasks += 1
        self.total_episodes += 1
        start_time = time.time()

        parsed_task = self.strategy.parse_task(task)

        # ═══════════════════════════════════════════════════════════════════
        # INJECT MULTI-CONDITION METADATA (if provided)
        # This allows condition_key to be passed from scenario without being
        # visible in task description (fair comparison with PRECEPT)
        # ═══════════════════════════════════════════════════════════════════
        if metadata:
            if parsed_task.parameters is None:
                parsed_task.parameters = {}
            # Inject condition_key from metadata (for multi-condition enforcement)
            if "condition_key" in metadata and metadata["condition_key"]:
                parsed_task.parameters["condition_key"] = metadata["condition_key"]
            elif "conditions" in metadata and metadata["conditions"]:
                # BUGFIX: Derive condition_key from conditions list when not
                # explicitly provided (mirrors fix in precept_agent.py)
                conditions_list = metadata["conditions"]
                parsed_task.parameters["condition_key"] = "+".join(
                    sorted(str(c) for c in conditions_list)
                )
            # Inject conditions list if available
            if "conditions" in metadata:
                parsed_task.parameters["conditions"] = metadata["conditions"]

        # Extract conditions: prefer metadata, fallback to task string
        conditions = []
        if metadata and metadata.get("conditions"):
            conditions = metadata["conditions"]
        elif parsed_task.parameters and parsed_task.parameters.get("conditions"):
            conditions = parsed_task.parameters["conditions"]
        else:
            # Fallback: extract from task string
            conditions = extract_conditions_from_task(task)

        if self.verbose:
            _get_logger().info(f"[ExpeL] Task: {task[:60]}...")
            _get_logger().info(f"[ExpeL] Conditions: {conditions}")
            _get_logger().info(f"[ExpeL] Training mode: {training}")

        steps = 0
        success = False
        attempts = 0
        current_attempts: List[Dict[str, str]] = []
        llm_suggestions: List[str] = []
        errors_received: List[str] = []
        failed_options: List[str] = []
        successful_option: Optional[str] = None

        try:
            # Get available options
            available_options = self.strategy.get_options_for_task(parsed_task)

            while attempts < self.MAX_ATTEMPTS:
                if self.verbose:
                    _get_logger().debug(
                        f"[ExpeL] Attempt {attempts + 1}/{self.MAX_ATTEMPTS}"
                    )

                # Ask LLM with retrieved insights
                llm_result = await self._ask_llm_with_insights(
                    task=task,
                    parsed_task=parsed_task,
                    options=available_options,
                    conditions=conditions,
                    current_attempts=current_attempts,
                )
                steps += 1

                llm_suggestion = llm_result.get("solution")

                if llm_suggestion and llm_suggestion in available_options:
                    option_to_try = llm_suggestion
                    llm_suggestions.append(llm_suggestion)
                    used_llm_suggestion = True
                else:
                    option_to_try = random.choice(available_options)
                    used_llm_suggestion = False

                if self.verbose:
                    if used_llm_suggestion:
                        _get_logger().info(
                            f"   📌 Using LLM suggestion: {option_to_try}"
                        )
                    else:
                        _get_logger().info(f"   🎲 Using random: {option_to_try}")

                # Execute action
                exec_success, result = await self.strategy.execute_action(
                    self.mcp_client,
                    option_to_try,
                    parsed_task,
                )
                steps += 1
                attempts += 1

                if exec_success:
                    success = True
                    successful_option = option_to_try
                    if llm_suggestion == option_to_try:
                        self.llm_suggestions_followed += 1
                    break
                else:
                    error_msg = str(result) if result else "Operation failed"
                    errors_received.append(error_msg)
                    failed_options.append(option_to_try)
                    current_attempts.append(
                        {
                            "option": option_to_try,
                            "error": error_msg,
                        }
                    )
                    if llm_suggestion == option_to_try:
                        self.llm_suggestions_failed += 1

        except Exception as e:
            if self.verbose:
                _get_logger().error(f"[ExpeL] Error: {e}")

        duration = time.time() - start_time

        # ExpeL Core: Extract insight during training
        # Following the paper, we extract insights from both successes and failures
        if training and conditions:  # Only extract insights if there are conditions
            insight = await self._extract_insight(
                task=task,
                success=success,
                attempts=attempts,
                failed_options=failed_options,
                successful_option=successful_option,
                conditions=conditions,
                errors=errors_received,
            )
            if insight and insight.get("insight"):
                # Store trajectory information with the insight (per ExpeL paper)
                insight["task"] = task
                insight["trajectory"] = {
                    "attempts": attempts,
                    "failed_options": failed_options,
                    "successful_option": successful_option,
                    "errors": errors_received,
                }
                # Store insight with improved_baselines flag for metadata storage
                add_expel_insight(
                    insight,
                    condition_enhanced=self.condition_enhanced_retrieval,
                    improved_baselines=self.improved_baselines,
                )

        if success:
            self.successful_tasks += 1
        self.steps_per_task.append(steps)

        # Summary logging
        if self.verbose:
            _get_logger().info("╔" + "═" * 68 + "╗")
            _get_logger().info("║" + " ExpeL TASK SUMMARY ".center(68) + "║")
            _get_logger().info("╠" + "═" * 68 + "╣")
            _get_logger().info(f"║ Task: {task[:58]}...".ljust(69) + "║")
            _get_logger().info(f"║ SUCCESS: {success}".ljust(69) + "║")
            _get_logger().info(
                f"║ Attempts: {attempts}/{self.MAX_ATTEMPTS}".ljust(69) + "║"
            )
            _get_logger().info(
                f"║ First-Try Success: {attempts == 1 and success}".ljust(69) + "║"
            )
            _get_logger().info(f"║ Conditions: {conditions}".ljust(69) + "║")
            _get_logger().info(
                f"║ Total Insights: {len(self.get_insights())}".ljust(69) + "║"
            )
            _get_logger().info("╚" + "═" * 68 + "╝")

        return {
            "success": success,
            "steps": steps,
            "duration": duration,
            "attempts": attempts,
            "first_try": success and attempts == 1,  # Success without error recovery
            "llm_suggestions": llm_suggestions,
            "llm_suggestion": llm_suggestions[0] if llm_suggestions else None,
            "conditions": conditions,
            "insights_retrieved": self.insights_retrieved,
            "insights_applied": self.insights_applied,
            "total_insights": len(self.get_insights()),
            "domain": self.strategy.domain_name,
        }

    # =========================================================================
    # STATISTICS
    # =========================================================================

    def get_success_rate(self) -> float:
        """Get the current success rate."""
        return compute_success_rate(self.successful_tasks, self.total_tasks)

    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive statistics."""
        stats = build_core_stats(
            total_tasks=self.total_tasks,
            successful_tasks=self.successful_tasks,
            steps_per_task=self.steps_per_task,
            llm_calls=self.llm_calls,
            llm_suggestions_followed=self.llm_suggestions_followed,
            llm_suggestions_failed=self.llm_suggestions_failed,
            domain=self.strategy.domain_name,
            baseline_type="expel",
        )
        # Add ExpeL-specific stats
        stats = add_expel_stats(
            stats,
            insights_extracted=self.insights_extracted,
            insights_retrieved=self.insights_retrieved,
            insights_applied=self.insights_applied,
            total_tasks=self.total_tasks,
        )
        stats["total_episodes"] = self.total_episodes
        return stats
