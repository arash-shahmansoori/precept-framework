"""
AutoGen PRECEPT Agent with COMPASS Advantages.

This module provides a generic AutoGen agent with PRECEPT learning capabilities
and COMPASS advantages (ML complexity analysis, smart rollouts, etc.).

Uses the Strategy Pattern for domain-specific behavior.
Works with ANY black swan category by injecting the appropriate strategy.

Architecture:
- Configuration-driven via PreceptConfig
- Dependency injection for MCP clients and strategies
- Pure functions for core logic (testable)
- Constraint handling via RefineInterceptor

Features:
- ML-based complexity analysis (COMPASS)
- Smart rollout allocation (COMPASS)
- Dynamic rule learning (PRECEPT)
- Multi-strategy coordination
- Docker-based code execution for coding domain (optional)

Usage:
    from precept import PRECEPTAgent
    from precept.domain_strategies import LogisticsDomainStrategy
    from precept.config import PreceptConfig

    # With default config
    agent = PRECEPTAgent(domain_strategy=LogisticsDomainStrategy())

    # With custom config
    config = PreceptConfig()
    config.agent.max_attempts = 5
    agent = PRECEPTAgent(domain_strategy=LogisticsDomainStrategy(), config=config)

    await agent.connect()
    result = await agent.run_task("Book shipment from Rotterdam to Boston")
"""

import asyncio
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Type

# Structured outputs for robust LLM parsing
from .structured_outputs import ConfidenceLevel, ReasoningResponse, TaskParseResponse

# AutoGen imports
try:
    from autogen_agentchat.agents import AssistantAgent
    from autogen_ext.models.openai import OpenAIChatCompletionClient
    from mcp import StdioServerParameters

    AUTOGEN_AVAILABLE = True
except ImportError:
    AUTOGEN_AVAILABLE = False
    AssistantAgent = None
    OpenAIChatCompletionClient = None
    StdioServerParameters = None

# Local imports
from .agent_functions import (
    add_error_constraint,
    apply_llm_suggestion,
    apply_procedure_hint,
    build_constraint_stack_prompt,
    build_llm_reasoning_result,
    build_reasoning_prompt,
    build_task_record,
    build_task_result,
    extract_and_store_atomic_precepts,
    fetch_context,
    fetch_context_compositional,
    fetch_context_with_hybrid,
    format_error_feedback,
    format_failure_context,
    parallel_fetch,
    parse_llm_response,
    record_error,
    record_successful_solution,
    store_experience_and_trigger_learning,
    update_failure_counter,
    update_llm_stats,
)
from .compass_controller import (
    COMPASSAction,
    COMPASSController,
    create_compass_controller,
)
from .config import (
    AgentConfig,
    PreceptConfig,
    get_agent_logger,
    get_default_config,
)
from .constraints import RefineInterceptor, create_refine_interceptor
from .domain_strategies.base import DomainStrategy
from .precept_orchestrator import PRECEPTConfig
from .scoring import compute_scores_from_task_results

# Module-level logger (lazy initialization to avoid stdout logging during import)
# This is critical for MCP server which uses stdout for JSONRPC
_logger = None


def _get_logger():
    """Get or create the module logger (lazy initialization)."""
    global _logger
    if _logger is None:
        _logger = get_agent_logger()
    return _logger


class PRECEPTAgent:
    """
    Generic AutoGen agent with PRECEPT learning + COMPASS advantages.

    Uses the Strategy Pattern for domain-specific behavior.
    Works with ANY black swan category by injecting the appropriate strategy.

    Architecture follows software engineering best practices:
    - Dependency injection for configuration and dependencies
    - Pure functions for core logic (in agent_functions.py)
    - Separation of concerns (constraints in constraints.py)
    - Configuration-driven behavior (via PreceptConfig)

    Features:
    - ML-based complexity analysis (COMPASS)
    - Smart rollout allocation (COMPASS)
    - Dynamic rule learning (PRECEPT)
    - Multi-strategy coordination
    - Docker-based code execution for coding domain (optional)
    - Dynamic configuration updates from real execution feedback
    """

    def __init__(
        self,
        domain_strategy: DomainStrategy,
        config: Optional[PreceptConfig] = None,
        # Legacy parameters for backward compatibility
        model: Optional[str] = None,
        precept_config: Optional[PRECEPTConfig] = None,
        server_script: Optional[Path] = None,
        enable_llm_reasoning: Optional[bool] = None,
        force_llm_reasoning: Optional[bool] = None,
        verbose_llm: Optional[bool] = None,
        max_internal_workers: Optional[int] = None,
    ):
        """
        Initialize the AutoGen PRECEPT agent.

        Args:
            domain_strategy: The domain strategy to use (determines domain behavior)
            config: Full PRECEPT configuration (preferred)
            model: The OpenAI model to use (legacy, use config.llm.model)
            precept_config: Optional PRECEPT configuration (legacy)
            server_script: Path to MCP server script (legacy, use config.server_script)
            enable_llm_reasoning: Enable LLM reasoning (legacy, use config.agent)
            force_llm_reasoning: Force LLM reasoning (legacy, use config.agent)
            verbose_llm: Verbose LLM logging (legacy, use config.agent)
            max_internal_workers: Max internal workers (legacy, use config.agent)
        """
        if not AUTOGEN_AVAILABLE:
            raise ImportError(
                "AutoGen dependencies not available. "
                "Install with: pip install autogen-agentchat autogen-ext mcp"
            )

        # Initialize configuration
        self.config = config or get_default_config()

        # Apply legacy parameter overrides
        self._apply_legacy_overrides(
            model=model,
            precept_config=precept_config,
            server_script=server_script,
            enable_llm_reasoning=enable_llm_reasoning,
            force_llm_reasoning=force_llm_reasoning,
            verbose_llm=verbose_llm,
            max_internal_workers=max_internal_workers,
        )

        # Store strategy
        self.strategy = domain_strategy

        # MCP client and AutoGen agent (initialized in connect())
        self.mcp_client = None
        self.agent: Optional[AssistantAgent] = None
        self.model_client: Optional[OpenAIChatCompletionClient] = None

        # Internal concurrency control
        self._internal_semaphore: Optional[asyncio.Semaphore] = None

        # ─── Statistics ───
        self.total_tasks = 0
        self.successful_tasks = 0
        self.steps_per_task: List[int] = []

        # COMPASS tracking
        self.tasks_since_consolidation = 0
        self.tasks_since_compass = 0
        self.consecutive_failures = 0

        # Task results for COMPASS scoring
        self.task_results: List[Dict] = []
        self.learning_events: List[str] = []

        # ─── COMPASS PROMPT EVOLUTION ───
        self._current_system_prompt: str = ""
        self._prompt_generation: int = 0
        self._prompt_updated_at_task: int = 0

        # ─── LLM Reasoning Statistics ───
        self._llm_reasoning_calls: int = 0
        self._llm_reasoning_successes: int = 0
        self._llm_reasoning_failures: int = 0

        # ─── DETERMINISTIC PRUNING ───
        # Pass the soft_constraints_retriable flag from config
        # When True, SOFT errors are not permanently forbidden (more exhaustive search)
        self._refine_interceptor: RefineInterceptor = create_refine_interceptor(
            soft_constraints_retriable=self.config.agent.soft_constraints_retriable,
        )

        # ─── COMPASS CONTROLLER ───
        # The "System 2" executive that implements:
        # 1. Hierarchical Constraint Prioritization (Physics > Policy > Instruction)
        # 2. Fast-path routing for trivial tasks
        # 3. Epistemic detour triggers (using domain-provided probes)
        # 4. Constraint-aware LLM context injection
        from .compass_controller import COMPASSConfig

        compass_config = COMPASSConfig(
            enable_fast_path=getattr(self.config.agent, "enable_fast_path", True),
            enable_epistemic_probing=getattr(
                self.config.agent, "enable_epistemic_probing", True
            ),
            enable_constraint_hierarchy=getattr(
                self.config.agent, "enable_constraint_hierarchy", True
            ),
            trivial_confidence_threshold=getattr(
                self.config.agent, "trivial_confidence_threshold", 0.8
            ),
            max_probes_per_task=getattr(self.config.agent, "max_probes_per_task", 3),
            learn_from_probes=True,
        )
        self._compass_controller: COMPASSController = create_compass_controller(
            config=compass_config,
            domain_strategy=self.strategy,  # Inject domain for probes
        )

    def _apply_legacy_overrides(
        self,
        model: Optional[str] = None,
        precept_config: Optional[PRECEPTConfig] = None,
        server_script: Optional[Path] = None,
        enable_llm_reasoning: Optional[bool] = None,
        force_llm_reasoning: Optional[bool] = None,
        verbose_llm: Optional[bool] = None,
        max_internal_workers: Optional[int] = None,
    ) -> None:
        """Apply legacy parameter overrides to configuration."""
        if model is not None:
            self.config.llm.model = model
        if server_script is not None:
            self.config.server_script = server_script
        if enable_llm_reasoning is not None:
            self.config.agent.enable_llm_reasoning = enable_llm_reasoning
        if force_llm_reasoning is not None:
            self.config.agent.force_llm_reasoning = force_llm_reasoning
        if verbose_llm is not None:
            self.config.agent.verbose_llm = verbose_llm
        if max_internal_workers is not None:
            self.config.agent.max_internal_workers = max_internal_workers

        # Apply PRECEPTConfig if provided
        if precept_config is not None:
            self.config.agent.consolidation_interval = (
                precept_config.consolidation_interval
            )
            self.config.agent.compass_evolution_interval = (
                precept_config.compass_evolution_interval
            )
            self.config.agent.max_memories = precept_config.max_memories
            self.config.agent.enable_compass_optimization = (
                precept_config.enable_compass_optimization
            )

    # =========================================================================
    # PROPERTY ACCESSORS (for backward compatibility)
    # =========================================================================

    @property
    def model(self) -> str:
        """Get the LLM model name."""
        return self.config.llm.model

    @property
    def server_script(self) -> Path:
        """Get the MCP server script path."""
        return self.config.server_script

    @property
    def enable_llm_reasoning(self) -> bool:
        """Check if LLM reasoning is enabled."""
        return self.config.agent.enable_llm_reasoning

    @property
    def force_llm_reasoning(self) -> bool:
        """Check if LLM reasoning is forced."""
        return self.config.agent.force_llm_reasoning

    @property
    def verbose_llm(self) -> bool:
        """Check if verbose LLM logging is enabled."""
        return self.config.agent.verbose_llm

    @property
    def max_internal_workers(self) -> int:
        """Get max internal workers."""
        return self.config.agent.max_internal_workers

    @property
    def consolidation_interval(self) -> int:
        """Get consolidation interval."""
        return self.config.agent.consolidation_interval

    @property
    def compass_evolution_interval(self) -> int:
        """Get COMPASS evolution interval."""
        return self.config.agent.compass_evolution_interval

    @property
    def failure_threshold(self) -> int:
        """Get failure threshold."""
        return self.config.agent.failure_threshold

    @property
    def precept_config(self) -> AgentConfig:
        """Get agent configuration (legacy compatibility)."""
        return self.config.agent

    @property
    def max_retries(self) -> int:
        """Get max retries from config (single source of truth)."""
        return self.config.agent.max_retries

    @property
    def _pruning_stats(self) -> Dict[str, int]:
        """Get pruning stats from interceptor."""
        return self._refine_interceptor.get_stats()

    # =========================================================================
    # CONNECTION METHODS
    # =========================================================================

    async def connect(self) -> None:
        """Connect to MCP server and initialize agent."""
        from .compass_mcp_client import PRECEPTMCPClientWithCOMPASS

        # Initialize internal semaphore for concurrent operations
        self._internal_semaphore = asyncio.Semaphore(self.max_internal_workers)

        _get_logger().info(
            f"🚀 Initializing AutoGen PRECEPT Agent for [{self.strategy.domain_name}]..."
        )
        _get_logger().info(
            f"    Internal concurrency: {self.max_internal_workers} workers"
        )

        # Get project root for PYTHONPATH
        project_root = self.server_script.parent.parent.parent

        # Connect using the COMPASS-enhanced client
        self.mcp_client = PRECEPTMCPClientWithCOMPASS(
            server_params=StdioServerParameters(
                command="python3",
                args=[str(self.server_script)],
                env={
                    **os.environ,
                    "PYTHONPATH": str(project_root / "src"),
                    "PRECEPT_ENABLE_COMPASS": os.environ.get("PRECEPT_ENABLE_COMPASS", "1"),
                    "PRECEPT_INCLUDE_RULES_IN_PROMPT": os.environ.get("PRECEPT_INCLUDE_RULES_IN_PROMPT", "1"),
                },
            )
        )
        await self.mcp_client.connect()

        # ═══════════════════════════════════════════════════════════════════
        # DYNAMIC PROBE DISCOVERY
        # Discover available diagnostic probes from MCP server at runtime.
        # The agent doesn't know a priori what probes exist - it discovers them.
        # ═══════════════════════════════════════════════════════════════════
        if (
            self.config.agent.enable_epistemic_probing
            if hasattr(self.config.agent, "enable_epistemic_probing")
            else True
        ):
            try:
                await self._compass_controller.discover_probes(self.mcp_client)
                _get_logger().info("  ✓ Discovered diagnostic probes from MCP server")
            except Exception as e:
                _get_logger().warning(f"  ⚠ Probe discovery failed: {e}")

        # Create AutoGen tools using domain strategy
        autogen_tools = self.strategy.create_autogen_tools(self.mcp_client)

        # Create model client
        self.model_client = OpenAIChatCompletionClient(model=self.model)

        # Create AutoGen agent with domain-specific prompt
        self._current_system_prompt = self.strategy.get_system_prompt()
        self.agent = AssistantAgent(
            name=f"PRECEPT_{self.strategy.category.value}_Agent",
            model_client=self.model_client,
            tools=autogen_tools,
            system_message=self._current_system_prompt,
        )

        # ═══════════════════════════════════════════════════════════════════
        # REGISTER EXECUTION CALLBACK FOR VERIFIED COMPASS/GEPA EVOLUTION
        # ═══════════════════════════════════════════════════════════════════
        # This enables COMPASS/GEPA to use REAL agent execution for
        # evaluating candidate prompts, instead of heuristic scoring.
        # ═══════════════════════════════════════════════════════════════════
        await self._register_compass_execution_callback()

        self._print_connection_info()

    async def _register_compass_execution_callback(self) -> None:
        """
        Register this agent as the execution callback for COMPASS/GEPA evolution.

        ═══════════════════════════════════════════════════════════════════════════
        VERIFIED EVOLUTION: Real agent execution for candidate prompt evaluation
        ═══════════════════════════════════════════════════════════════════════════

        When COMPASS/GEPA evolution evaluates a candidate prompt, instead of using
        heuristic LLM simulation, it calls this agent's run_task() method.

        The signal flow:
            Candidate Prompt → This Agent.run_task() → Predicted Solution
                                                     ↓
                            Environment (MCP Tools) verifies internally
                            (predicted == expected? - agent NEVER sees expected)
                                                     ↓
                            Returns: {success: bool, error_code, error_message}
                                                     ↓
                            COMPASS/GEPA uses ONLY these signals for evolution

        This ensures:
        - No "cheating" by seeing expected solutions
        - Binary verifiable scoring (success/failure from environment)
        - Honest feedback loop based on actual agent predictions
        - Applicable to ALL verifiable tasks (Black Swan CSP, compositional, etc.)
        ═══════════════════════════════════════════════════════════════════════════
        """
        try:
            # Create the execution callback that COMPASS/GEPA will use
            async def execute_with_prompt(prompt: str, task: dict) -> dict:
                """
                Execute a task with a candidate prompt for COMPASS/GEPA evaluation.

                Args:
                    prompt: The candidate system prompt to evaluate
                    task: {"task": str, "goal": str, "metadata": dict}

                Returns:
                    {
                        "success": bool,           # From environment verification
                        "error_code": str | None,
                        "error_message": str | None,
                        "predicted_solution": Any,
                        "steps": int,
                    }
                """
                # Save current prompt
                original_prompt = self._current_system_prompt

                try:
                    # Temporarily switch to candidate prompt
                    self._current_system_prompt = prompt

                    # Recreate agent with candidate prompt
                    if self.agent and self.model_client:
                        autogen_tools = self.strategy.create_autogen_tools(
                            self.mcp_client
                        )
                        self.agent = AssistantAgent(
                            name=f"PRECEPT_{self.strategy.category.value}_Agent",
                            model_client=self.model_client,
                            tools=autogen_tools,
                            system_message=prompt,
                        )

                    # Execute the task - environment verifies internally
                    task_str = task.get("task", "")
                    metadata = task.get("metadata", {})
                    result = await self.run_task(task_str, metadata=metadata)

                    return {
                        "success": result.get("success", False),
                        "error_code": None,  # Not exposed in run_task result
                        "error_message": result.get("response", "")
                        if not result.get("success")
                        else None,
                        "predicted_solution": result.get("strategy", ""),
                        "steps": result.get("steps", 0),
                    }

                finally:
                    # Restore original prompt
                    self._current_system_prompt = original_prompt
                    if self.agent and self.model_client:
                        autogen_tools = self.strategy.create_autogen_tools(
                            self.mcp_client
                        )
                        self.agent = AssistantAgent(
                            name=f"PRECEPT_{self.strategy.category.value}_Agent",
                            model_client=self.model_client,
                            tools=autogen_tools,
                            system_message=original_prompt,
                        )

            # Register with MCP server
            await self.mcp_client.call_tool(
                "register_compass_execution_callback",
                {"callback_id": f"agent_{id(self)}"},
            )

            # Store callback for direct use by local COMPASS components
            self._compass_execute_callback = execute_with_prompt

            if self.verbose_llm:
                _get_logger().info(
                    "  ✓ Registered COMPASS execution callback (verified evolution)"
                )

        except Exception as e:
            if self.verbose_llm:
                _get_logger().warning(
                    f"  ⚠ Failed to register COMPASS execution callback: {e}"
                )

    def _print_connection_info(self) -> None:
        """Log connection information."""
        _get_logger().info("  ✓ Connected to MCP server")
        _get_logger().info(f"  ✓ Domain: {self.strategy.domain_name}")
        _get_logger().info(f"  ✓ Actions: {self.strategy.get_available_actions()}")
        _get_logger().info("  ✓ COMPASS Advantages enabled:")
        _get_logger().info("    • PRECEPTComplexityAnalyzer (ML-based)")
        _get_logger().info("    • SmartRolloutStrategy (adaptive)")
        _get_logger().info("    • MultiStrategyCoordinator")
        _get_logger().info("    • Dynamic Prompt Evolution (COMPASS)")

        if self.force_llm_reasoning:
            _get_logger().info(
                "    • LLM Reasoning (FORCED - always calls LLM for comparison)"
            )
        elif self.enable_llm_reasoning:
            _get_logger().info(
                "    • LLM Reasoning (Tier 2 - when no programmatic match)"
            )
        else:
            _get_logger().info("    • LLM Reasoning (disabled - programmatic only)")

        # Show Docker execution status for coding domain
        if self.strategy.domain_name == "coding":
            self._print_coding_domain_info()

        _get_logger().info(
            f"✅ AutoGen PRECEPT Agent [{self.strategy.domain_name}] ready"
        )

    def _print_coding_domain_info(self) -> None:
        """Log coding domain-specific info."""
        if (
            hasattr(self.strategy, "enable_docker_execution")
            and self.strategy.enable_docker_execution
        ):
            docker_available = getattr(self.strategy, "is_docker_available", False)
            if callable(docker_available):
                docker_available = docker_available
            else:
                docker_available = bool(docker_available)

            if hasattr(self.strategy, "is_docker_available"):
                docker_available = self.strategy.is_docker_available

            if docker_available:
                _get_logger().info("    • Docker Code Execution (sandboxed)")
            else:
                _get_logger().info("    • Subprocess Fallback (Docker unavailable)")
            _get_logger().info("    • Dynamic Learning from Execution")
        else:
            _get_logger().info("    • Simulated Execution (MCP-based)")

    async def disconnect(self) -> None:
        """Disconnect from MCP server with proper async cleanup."""
        # Close OpenAI client first if present
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
    # PROMPT EVOLUTION
    # =========================================================================

    async def refresh_evolved_prompt(self) -> bool:
        """
        Refresh the system prompt with evolved version from COMPASS.

        Returns:
            True if prompt was updated, False otherwise
        """
        try:
            evolved_prompt = await self.mcp_client.get_evolved_prompt(
                include_rules=True
            )

            if evolved_prompt and "No evolved prompt" not in evolved_prompt:
                if evolved_prompt != self._current_system_prompt:
                    self._current_system_prompt = evolved_prompt
                    self._prompt_generation += 1
                    self._prompt_updated_at_task = self.total_tasks

                    if self.verbose_llm:
                        self._log_evolved_prompt(evolved_prompt)

                    # Recreate agent with evolved prompt
                    if self.agent and self.model_client:
                        autogen_tools = self.strategy.create_autogen_tools(
                            self.mcp_client
                        )
                        self.agent = AssistantAgent(
                            name=f"PRECEPT_{self.strategy.category.value}_Agent",
                            model_client=self.model_client,
                            tools=autogen_tools,
                            system_message=self._current_system_prompt,
                        )

                    return True

            return False

        except Exception as e:
            if self.verbose_llm:
                _get_logger().warning(f"Prompt refresh failed: {e}")
            return False

    def _log_evolved_prompt(self, evolved_prompt: str) -> None:
        """Log the evolved prompt for debugging."""
        _get_logger().debug(f"🧬 EVOLVED PROMPT (Gen {self._prompt_generation}):")
        preview = evolved_prompt[:500].replace("\n", "\n       ")
        _get_logger().debug(f"   {preview}...")
        if "LEARNED RULES" in evolved_prompt:
            rules_start = evolved_prompt.find("LEARNED RULES")
            if rules_start > 0:
                rules_section = evolved_prompt[rules_start : rules_start + 300]
                _get_logger().debug(
                    f"   ... {rules_section.replace(chr(10), chr(10) + '       ')}..."
                )

    def get_current_prompt(self) -> str:
        """Get the current (potentially evolved) system prompt."""
        return self._current_system_prompt

    def get_prompt_stats(self) -> Dict[str, Any]:
        """Get statistics about prompt evolution."""
        return {
            "prompt_generation": self._prompt_generation,
            "prompt_updated_at_task": self._prompt_updated_at_task,
            "prompt_length": len(self._current_system_prompt),
            "has_evolved": self._prompt_generation > 0,
        }

    # =========================================================================
    # INTERNAL CONCURRENCY HELPERS
    # =========================================================================

    async def _run_with_semaphore(self, coro):
        """Run an async operation with internal semaphore for concurrency control."""
        if self._internal_semaphore is None:
            return await coro
        async with self._internal_semaphore:
            return await coro

    async def _parallel_fetch(self, *coros):
        """Run multiple coroutines in parallel, limited by internal semaphore."""
        return await parallel_fetch(*coros, semaphore=self._internal_semaphore)

    # =========================================================================
    # LLM REASONING
    # =========================================================================

    async def _llm_reason_with_evolved_prompt(
        self,
        task: str,
        parsed_task: Any,
        memories: str,
        learned_rules: str = "",
        error_feedback: str = "",
        forbidden_section: str = "",
        procedure: str = "",
    ) -> Optional[Dict[str, Any]]:
        """
        Use LLM to reason about task using evolved prompt with learned rules.

        This method delegates to pure functions for testability:
        - format_error_feedback() - Pure
        - build_reasoning_prompt() - Pure
        - parse_llm_response() - Pure
        - build_llm_reasoning_result() - Pure
        - update_llm_stats() - Pure

        Only _call_llm() performs I/O.
        """
        if not self.model_client:
            if self.verbose_llm:
                _get_logger().warning("LLM reasoning skipped: no model_client")
            return None

        if self.verbose_llm:
            _get_logger().debug(f"🧠 LLM reasoning for: {task[:50]}...")

        try:
            # Pure: Build prompt
            # NOTE: We intentionally do NOT pass available_options here.
            # This would defeat the black swan scenario where the agent
            # must learn through experience, not be given a cheat sheet.
            formatted_feedback = format_error_feedback(error_feedback)
            reasoning_prompt = build_reasoning_prompt(
                task=task,
                parsed_task=parsed_task,
                memories=memories,
                learned_rules=learned_rules,
                forbidden_section=forbidden_section,
                error_feedback=formatted_feedback,
                prompts=self.config.prompts,
                # Smart rule filtering config from AgentConfig
                max_rules_chars=self.config.agent.max_rules_chars,
                max_memories_chars=self.config.agent.max_memories_chars,
                enable_smart_rule_filtering=self.config.agent.enable_smart_rule_filtering,
                procedure=procedure,
            )

            # ═══════════════════════════════════════════════════════════════════
            # STRATEGY: Try structured output first (robust), fall back to regex
            # OpenAI's structured outputs GUARANTEE the response schema
            # ═══════════════════════════════════════════════════════════════════

            result = None

            # ATTEMPT 1: OpenAI Structured Output (preferred - guaranteed schema)
            structured_response = await self._call_llm_structured(reasoning_prompt)

            if structured_response:
                # Convert Pydantic model to dict for compatibility
                result = {
                    "suggested_solution": structured_response.solution,
                    "reasoning": structured_response.reasoning,
                    "confidence": structured_response.confidence.value
                    if hasattr(structured_response.confidence, "value")
                    else str(structured_response.confidence),
                }

                # Handle special cases
                if structured_response.solution.upper() == "EXPLORE":
                    result = None  # Exploration mode
                elif structured_response.solution.upper() == "EXHAUSTED":
                    result["suggested_solution"] = "EXHAUSTED"

                if self.verbose_llm:
                    _get_logger().debug(
                        f"✅ Structured output parsed: {structured_response.solution}"
                    )
            else:
                # ATTEMPT 2: Fallback to regex parsing (legacy, less reliable)
                if self.verbose_llm:
                    _get_logger().debug("⚠️ Using regex fallback for LLM parsing")

                response = await self._call_llm(reasoning_prompt)
                response_text = self._extract_response_text(response)

                if self.verbose_llm:
                    _get_logger().debug(f"🧠 LLM response: {response_text[:100]}...")

                # Pure: Parse response with regex (fragile fallback)
                parsed = parse_llm_response(response_text)

                # Pure: Build result
                result = build_llm_reasoning_result(parsed)

            # Pure: Update stats (returns new values instead of mutating)
            stats = update_llm_stats(
                calls=self._llm_reasoning_calls,
                successes=self._llm_reasoning_successes,
                failures=self._llm_reasoning_failures,
                result=result,
            )
            # Apply updated stats
            self._llm_reasoning_calls = stats["calls"]
            self._llm_reasoning_successes = stats["successes"]
            self._llm_reasoning_failures = stats["failures"]

            if result and self.verbose_llm:
                _get_logger().debug(f"✅ LLM suggested: {result['suggested_solution']}")
            elif self.verbose_llm:
                _get_logger().debug("LLM returned EXPLORE or unparseable response")

            return result

        except Exception as e:
            # Update failure stats
            self._llm_reasoning_calls += 1
            self._llm_reasoning_failures += 1
            if self.verbose_llm:
                _get_logger().error(f"LLM reasoning error: {e}")
            return None

    async def _call_llm(self, prompt: str) -> Any:
        """Make an LLM API call."""
        try:
            from autogen_core.models import SystemMessage, UserMessage

            return await self.model_client.create(
                messages=[
                    SystemMessage(content=self._current_system_prompt),
                    UserMessage(content=prompt, source="user"),
                ],
                extra_create_args={
                    "max_tokens": self.config.llm.max_tokens,
                    "temperature": self.config.llm.temperature,
                },
            )
        except ImportError:
            return await self.model_client.create(
                messages=[
                    {"role": "system", "content": self._current_system_prompt},
                    {"role": "user", "content": prompt},
                ],
                extra_create_args={
                    "max_tokens": self.config.llm.max_tokens,
                    "temperature": self.config.llm.temperature,
                },
            )

    async def _call_llm_structured(
        self,
        prompt: str,
        response_model: Type[ReasoningResponse] = ReasoningResponse,
    ) -> Optional[ReasoningResponse]:
        """
        Make an LLM API call with STRUCTURED OUTPUT (OpenAI feature).

        This is the ROBUST way to get structured responses from LLMs.
        Instead of asking the LLM to format text and then regex parsing,
        OpenAI guarantees the response matches the Pydantic schema.

        Args:
            prompt: The user prompt
            response_model: Pydantic model class for response structure

        Returns:
            Parsed Pydantic model instance, or None on error
        """
        try:
            # Try using OpenAI's native structured output feature
            from openai import AsyncOpenAI

            client = AsyncOpenAI()
            try:
                response = await client.beta.chat.completions.parse(
                    model=self.config.llm.model,
                    messages=[
                        {"role": "system", "content": self._current_system_prompt},
                        {"role": "user", "content": prompt},
                    ],
                    response_format=response_model,
                    max_tokens=self.config.llm.max_tokens,
                    temperature=self.config.llm.temperature,
                )

                # OpenAI returns parsed object directly
                parsed = response.choices[0].message.parsed

                if self.verbose_llm:
                    _get_logger().debug(
                        f"🧠 Structured LLM Response:\n"
                        f"   Solution: {parsed.solution}\n"
                        f"   Reasoning: {parsed.reasoning[:50]}...\n"
                        f"   Confidence: {parsed.confidence}"
                    )

                return parsed
            finally:
                try:
                    await client.close()
                except Exception:
                    pass

        except ImportError:
            # Fallback if openai package not available
            if self.verbose_llm:
                _get_logger().warning(
                    "OpenAI structured outputs not available, using regex fallback"
                )
            return None
        except Exception as e:
            if self.verbose_llm:
                _get_logger().warning(
                    f"Structured LLM call failed: {e}, using fallback"
                )
            return None

    def _extract_response_text(self, response: Any) -> str:
        """Extract text from LLM response."""
        if hasattr(response, "content"):
            return response.content
        elif isinstance(response, str):
            return response
        else:
            return str(response)

    # =========================================================================
    # HYBRID TASK PARSING (Rule-based + LLM Fallback)
    # =========================================================================

    async def _hybrid_parse_task(self, task: str) -> Any:
        """
        Hybrid task parsing: Rule-based first, LLM fallback for complex tasks.

        This combines the best of both approaches:
        1. Rule-based parsing (fast, deterministic) - for standard tasks
        2. LLM-assisted parsing (smart, flexible) - for complex/ambiguous tasks

        The LLM uses structured outputs (Pydantic) for guaranteed schema.

        Args:
            task: The raw task string

        Returns:
            ParsedTask object with extracted components
        """
        # STEP 1: Try rule-based parsing first (fast, no API call)
        parsed_task = self.strategy.parse_task(task)

        # Check if parsing was confident/complete
        parsing_confidence = self._assess_parsing_quality(task, parsed_task)

        if parsing_confidence >= 0.8:
            # Rule-based parsing succeeded
            if self.verbose_llm:
                _get_logger().debug(
                    f"📋 Rule-based parsing succeeded (confidence: {parsing_confidence:.2f})\n"
                    f"   Action: {parsed_task.action}\n"
                    f"   Entity: {parsed_task.entity}\n"
                    f"   Source: {parsed_task.source}\n"
                    f"   Target: {parsed_task.target}"
                )
            return parsed_task

        # STEP 2: LLM-assisted parsing for complex/ambiguous tasks
        if self.verbose_llm:
            _get_logger().info(
                f"🤖 Rule-based parsing uncertain (confidence: {parsing_confidence:.2f}), "
                f"using LLM fallback"
            )

        try:
            llm_parsed = await self._llm_parse_task(task)

            if llm_parsed:
                # Merge LLM parsing with rule-based (LLM overrides uncertain fields)
                if llm_parsed.entity and llm_parsed.entity != "unknown":
                    parsed_task.entity = llm_parsed.entity
                if llm_parsed.source:
                    parsed_task.source = llm_parsed.source
                if llm_parsed.target:
                    parsed_task.target = llm_parsed.target
                if llm_parsed.action and llm_parsed.action != "unknown":
                    parsed_task.action = llm_parsed.action
                if llm_parsed.parameters:
                    parsed_task.parameters.update(llm_parsed.parameters)

                if self.verbose_llm:
                    _get_logger().debug(
                        f"✅ LLM parsing enhanced result:\n"
                        f"   Action: {parsed_task.action}\n"
                        f"   Entity: {parsed_task.entity}\n"
                        f"   Source: {parsed_task.source}\n"
                        f"   Target: {parsed_task.target}\n"
                        f"   LLM Notes: {llm_parsed.parsing_notes or 'None'}"
                    )

        except Exception as e:
            if self.verbose_llm:
                _get_logger().warning(
                    f"LLM parsing failed: {e}, using rule-based result"
                )

        return parsed_task

    def _assess_parsing_quality(self, task: str, parsed_task: Any) -> float:
        """
        Assess the quality/confidence of rule-based parsing.

        Returns a score from 0.0 (low confidence) to 1.0 (high confidence).

        The scoring is designed to trigger LLM fallback for ambiguous tasks:
        - Generic defaults ("shipment", "book_shipment") don't boost confidence
        - Actual extracted entities/locations boost confidence
        - Domain-specific keywords in task increase confidence
        """
        score = 0.0
        task_lower = task.lower()

        # Generic defaults that don't indicate successful parsing
        generic_entities = {"shipment", "cargo", "package", "item", "customs"}
        generic_actions = {"book_shipment", "process", "handle"}

        # Check if action was meaningfully extracted (not just a generic default)
        if parsed_task.action and parsed_task.action != "unknown":
            # Give less credit for generic actions
            if parsed_task.action.lower() in generic_actions:
                score += 0.15  # Generic action
            else:
                score += 0.3  # Specific action extracted

        # Check if entity was meaningfully extracted
        if parsed_task.entity and parsed_task.entity != "unknown":
            # Give less credit for generic entities
            if parsed_task.entity.lower() in generic_entities:
                score += 0.15  # Generic entity (probably a default)
            else:
                score += 0.3  # Specific entity extracted

        # Source and target extraction - indicates more complete parsing
        if parsed_task.source or parsed_task.target:
            score += 0.2

        # Bonus: Check if parsed fields actually appear in the task text
        # This validates that extraction was accurate, not just defaulting
        if parsed_task.entity and parsed_task.entity.lower() in task_lower:
            if parsed_task.entity.lower() not in generic_entities:
                score += 0.15  # Extracted entity matches task text

        if parsed_task.source and parsed_task.source.lower() in task_lower:
            score += 0.1  # Source found in task

        if parsed_task.target and parsed_task.target.lower() in task_lower:
            score += 0.1  # Target found in task

        # Check for domain-specific keywords that indicate clear task intent
        domain_keywords = {
            "logistics": ["ship", "port", "cargo", "customs", "freight", "route"],
            "booking": ["book", "flight", "reservation", "passenger"],
            "coding": ["install", "package", "import", "dependency"],
            "devops": ["deploy", "pod", "stack", "container", "kubernetes"],
            "finance": ["trade", "buy", "sell", "market", "price"],
            "integration": ["oauth", "api", "webhook", "connect"],
        }

        domain = getattr(self.strategy, "domain_name", "logistics")
        keywords = domain_keywords.get(domain, [])

        keyword_matches = sum(1 for kw in keywords if kw in task_lower)
        if keyword_matches >= 2:
            score += 0.1  # Multiple domain keywords = clearer intent
        elif keyword_matches == 0:
            score -= 0.1  # No domain keywords = ambiguous task

        return max(0.0, min(score, 1.0))

    async def _llm_parse_task(self, task: str) -> Optional[TaskParseResponse]:
        """
        Use LLM with structured output to parse a complex task.

        This is called when rule-based parsing is uncertain.
        Uses OpenAI's JSON mode with explicit schema for compatibility.
        """
        parse_prompt = f"""Parse this task into structured JSON components.

TASK: {task}

DOMAIN: {self.strategy.domain_name}

AVAILABLE ACTIONS: {", ".join(self.strategy.get_available_actions())}

Return a JSON object with these fields:
{{
  "action": "primary action to perform (one of the available actions)",
  "entity": "main entity (port name, flight ID, package name, etc.)",
  "source": "origin/source if applicable, or null",
  "target": "destination/target if applicable, or null",
  "task_type": "category of task (shipment, booking, deployment, etc.)",
  "parameters": {{}},
  "confidence": "high/medium/low based on how clear the task is",
  "parsing_notes": "any ambiguity or assumptions made"
}}

Be precise. If something is unclear, say so in parsing_notes."""

        try:
            import json

            from openai import AsyncOpenAI

            client = AsyncOpenAI()

            try:
                # Use JSON mode for broader model compatibility
                response = await client.chat.completions.create(
                    model=self.config.llm.model,
                    messages=[
                        {
                            "role": "system",
                            "content": f"You are a task parser for the {self.strategy.domain_name} domain. "
                            "Always respond with valid JSON only, no other text.",
                        },
                        {"role": "user", "content": parse_prompt},
                    ],
                    response_format={"type": "json_object"},
                    max_tokens=300,
                    temperature=0.1,  # Low temperature for parsing
                )

                # Parse JSON response
                content = response.choices[0].message.content
                if content:
                    data = json.loads(content)
                    # Convert to TaskParseResponse
                    return TaskParseResponse(
                        action=data.get("action", "unknown"),
                        entity=data.get("entity", "unknown"),
                        source=data.get("source"),
                        target=data.get("target"),
                        task_type=data.get("task_type", "unknown"),
                        parameters=data.get("parameters", {}),
                        confidence=ConfidenceLevel(
                            data.get("confidence", "medium").lower()
                        ),
                        parsing_notes=data.get("parsing_notes"),
                    )

                return None
            finally:
                try:
                    await client.close()
                except Exception:
                    pass

        except Exception as e:
            if self.verbose_llm:
                _get_logger().debug(f"LLM task parsing failed: {e}")
            return None

    # =========================================================================
    # ENHANCED LOGGING UTILITIES
    # =========================================================================

    def _log_compass_decision(self, decision: Any, task: str) -> None:
        """Log detailed COMPASS decision for debugging and understanding."""
        if not self.verbose_llm:
            return

        action_emoji = {
            "BLOCK": "🚫",
            "PIVOT": "🔄",
            "FAST_PATH": "⚡",
            "PROCEED": "✅",
            "PROBE": "🔍",
        }

        emoji = action_emoji.get(str(decision.action.name), "❓")

        _get_logger().info(
            f"\n{'═' * 70}\n"
            f"{emoji} COMPASS DECISION: {decision.action.name}\n"
            f"{'═' * 70}\n"
            f"   Task: {task[:60]}...\n"
            f"   Reason: {decision.reason or 'N/A'}\n"
            f"   Blocking Constraint: {decision.blocking_constraint or 'None'}\n"
            f"   Negotiated Alternative: {decision.negotiated_alternative or 'None'}\n"
            f"   Constraint Context: {decision.constraint_context[:50] if decision.constraint_context else 'None'}...\n"
            f"{'═' * 70}"
        )

    def _log_llm_suggestion(
        self,
        suggestion: Optional[Dict[str, Any]],
        task: str,
        parsed_task: Any,
    ) -> None:
        """Log what LLM suggestion returns and its details."""
        if not self.verbose_llm:
            return

        if suggestion:
            _get_logger().info(
                f"\n{'─' * 70}\n"
                f"🧠 LLM SUGGESTION DETAILS\n"
                f"{'─' * 70}\n"
                f"   Task: {task[:50]}...\n"
                f"   Suggested Solution: {suggestion.get('suggested_solution', 'N/A')}\n"
                f"   Reasoning: {suggestion.get('reasoning', 'N/A')[:80]}...\n"
                f"   Confidence: {suggestion.get('confidence', 'N/A')}\n"
                f"   Current Entity: {parsed_task.entity}\n"
                f"{'─' * 70}"
            )
        else:
            _get_logger().debug(
                f"🧠 LLM returned None/EXPLORE for task: {task[:50]}..."
            )

    def _log_suggestion_application(
        self,
        parsed_task: Any,
        was_applied: bool,
        strategy_used: str,
    ) -> None:
        """Log how apply_llm_suggestion applies the suggestion."""
        if not self.verbose_llm:
            return

        preferred = parsed_task.parameters.get("preferred_solution")

        if was_applied and preferred:
            _get_logger().info(
                f"📌 SUGGESTION APPLIED:\n"
                f"   preferred_solution set to: {preferred}\n"
                f"   Strategy: {strategy_used}\n"
                f"   → execute_action will use this value FIRST"
            )
        else:
            _get_logger().debug(
                f"📌 No suggestion applied, execute_action will use entity: {parsed_task.entity}"
            )

    def _log_action_execution(
        self,
        parsed_task: Any,
        result: Any,
        used_preferred: bool,
    ) -> None:
        """Log how execute_action processes the task."""
        if not self.verbose_llm:
            return

        preferred = parsed_task.parameters.get("preferred_solution")

        _get_logger().info(
            f"\n{'─' * 70}\n"
            f"⚡ ACTION EXECUTION\n"
            f"{'─' * 70}\n"
            f"   Action: {parsed_task.action}\n"
            f"   Original Entity: {parsed_task.entity}\n"
            f"   Preferred Solution: {preferred or 'None'}\n"
            f"   Actually Used: {preferred if used_preferred else parsed_task.entity}\n"
            f"   Success: {result.success}\n"
            f"   Strategy: {result.strategy_used or 'N/A'}\n"
            f"{'─' * 70}"
        )

    def _log_gepa_evolution(
        self,
        trigger_reason: str,
        tasks_since: int,
        new_prompt_preview: str,
    ) -> None:
        """Log GEPA/COMPASS prompt evolution events."""
        if not self.verbose_llm:
            return

        _get_logger().info(
            f"\n{'═' * 70}\n"
            f"🧬 GEPA PROMPT EVOLUTION TRIGGERED\n"
            f"{'═' * 70}\n"
            f"   Trigger: {trigger_reason}\n"
            f"   Tasks Since Last Evolution: {tasks_since}\n"
            f"   New Prompt Preview: {new_prompt_preview[:100]}...\n"
            f"{'═' * 70}"
        )

    # =========================================================================
    # MAIN TASK EXECUTION
    # =========================================================================

    async def run_task(
        self,
        task: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Run a task using PRECEPT+COMPASS learning with GENERIC strategy-driven flow.

        This method is domain-agnostic and delegates all domain-specific logic
        to the injected DomainStrategy.

        Args:
            task: The task string to execute
            metadata: Optional dict with scenario metadata (e.g., condition_key for multi-condition)

        Returns:
            Dict with success, steps, duration, response, strategy, complexity, domain
        """
        self.total_tasks += 1
        start_time = time.time()

        # ═══════════════════════════════════════════════════════════════════════
        # TASK PARSING: Rule-based (fast) or Hybrid (rule-based + LLM fallback)
        # ═══════════════════════════════════════════════════════════════════════
        if self.config.agent.enable_hybrid_parsing:
            # Hybrid: tries rule-based first, falls back to LLM for complex tasks
            parsed_task = await self._hybrid_parse_task(task)
        else:
            # Standard: fast rule-based parsing only
            parsed_task = self.strategy.parse_task(task)

        # ═══════════════════════════════════════════════════════════════════════
        # INJECT MULTI-CONDITION METADATA (if provided)
        # This allows condition_key to be passed from scenario without being
        # visible in task description (prevents baselines from "reading" it)
        # ═══════════════════════════════════════════════════════════════════════
        if metadata:
            if parsed_task.parameters is None:
                parsed_task.parameters = {}
            # Inject condition_key from metadata (for multi-condition enforcement)
            if "condition_key" in metadata and metadata["condition_key"]:
                parsed_task.parameters["condition_key"] = metadata["condition_key"]
            elif "conditions" in metadata and metadata["conditions"]:
                # ═══════════════════════════════════════════════════════════════
                # BUGFIX: Derive condition_key from conditions list when not
                # explicitly provided. This fixes the code path where COMPASS
                # prompt evolution creates evaluation tasks with conditions but
                # no condition_key, causing execute_action to route through
                # book_shipment (simulation-based) instead of the hash-enforced
                # execute_logistics_multi_condition. Without this, rules learned
                # during COMPASS evaluation use the wrong enforcement path,
                # producing incorrect rules (e.g., R-482 → antwerp when the
                # hash answer is long_beach).
                # ═══════════════════════════════════════════════════════════════
                conditions = metadata["conditions"]
                derived_key = "+".join(sorted(str(c) for c in conditions))
                parsed_task.parameters["condition_key"] = derived_key
                if self.verbose_llm:
                    _get_logger().debug(
                        f"🔧 DERIVED condition_key from conditions list: {derived_key}"
                    )
            # Inject conditions list if available
            if "conditions" in metadata:
                parsed_task.parameters["conditions"] = metadata["conditions"]
            # Inject expected_solution for debugging/analysis (not used in execution)
            if "expected_solution" in metadata:
                parsed_task.parameters["expected_solution"] = metadata[
                    "expected_solution"
                ]

        # Complexity analysis
        complexity = self.mcp_client.analyze_complexity(
            task,
            f"{parsed_task.action} {parsed_task.entity}",
        )

        task_steps = 0
        overhead_steps = 0
        success = False
        final_response = ""
        strategy_used = ""
        first_try = False  # True if success without error recovery
        learned_rule_event: Optional[Dict[str, str]] = None

        try:
            # ═══════════════════════════════════════════════════════════════════
            # COMPASS DECISION POINT: Evaluate action before execution
            # Implements: Hierarchical constraints, fast-path routing
            # ═══════════════════════════════════════════════════════════════════
            self._compass_controller.set_user_instruction(task)
            compass_decision = self._compass_controller.evaluate_action(
                task=task,
                parsed_task=parsed_task,
                complexity=complexity,
            )

            # ═══════════════════════════════════════════════════════════════════
            # ENHANCED LOGGING: COMPASS Decision Details
            # ═══════════════════════════════════════════════════════════════════
            self._log_compass_decision(compass_decision, task)

            # Handle COMPASS decisions
            if compass_decision.action == COMPASSAction.BLOCK:
                # Physics constraint blocks this action entirely
                if self.verbose_llm:
                    _get_logger().warning(
                        f"🚫 COMPASS BLOCK: {compass_decision.reason}"
                    )
                return build_task_result(
                    success=False,
                    task_steps=0,
                    overhead_steps=1,
                    duration=time.time() - start_time,
                    response=f"Blocked by constraint: {compass_decision.blocking_constraint}",
                    strategy="compass:blocked",
                    complexity=complexity,
                    domain=self.strategy.domain_name,
                )

            if compass_decision.action == COMPASSAction.PIVOT:
                # Use negotiated alternative
                if self.verbose_llm:
                    _get_logger().info(
                        f"🔄 COMPASS PIVOT: {compass_decision.negotiated_alternative}"
                    )
                parsed_task.parameters["preferred_solution"] = (
                    compass_decision.negotiated_alternative
                )
                strategy_used = f"compass:pivot:{compass_decision.blocking_constraint}"
                overhead_steps += 1

            use_fast_path = compass_decision.action == COMPASSAction.FAST_PATH
            constraint_context = compass_decision.constraint_context

            # ═══════════════════════════════════════════════════════════════════
            # CONTEXT FETCH: Hybrid OR Compositional mode
            # If compositional generalization is enabled, use atomic constraint stacking
            # ═══════════════════════════════════════════════════════════════════
            condition_key = (parsed_task.parameters or {}).get("condition_key")
            compositional_context = None
            constraint_stack_prompt = ""

            if condition_key and self.config.agent.enable_compositional_generalization:
                # ═══════════════════════════════════════════════════════════════
                # COMPOSITIONAL MODE: Atomic Constraint Stacking
                # ═══════════════════════════════════════════════════════════════
                # 1. Decompose composite condition into atomic components
                # 2. Retrieve atomic precepts for each component
                # 3. Stack constraints for LLM synthesis
                # ═══════════════════════════════════════════════════════════════
                compositional_context = await fetch_context_compositional(
                    mcp_client=self.mcp_client,
                    query=f"{parsed_task.action} {parsed_task.target or parsed_task.entity}",
                    task_type=f"{self.strategy.domain_name}:{parsed_task.action}",
                    task_description=task,
                    condition_key=condition_key,
                    similarity_threshold=0.5,
                    min_precept_confidence=self.config.agent.atomic_precept_min_confidence,
                    semaphore=self._internal_semaphore,
                )

                # Build constraint stack prompt for LLM synthesis
                if compositional_context.constraint_stack:
                    constraint_stack_prompt = build_constraint_stack_prompt(
                        compositional_context.constraint_stack,
                        resolution=compositional_context.resolution,
                    )

                if self.verbose_llm:
                    _get_logger().info(
                        f"⚛️ COMPOSITIONAL FETCH: {condition_key[:40]}...\n"
                        f"   Mode: {compositional_context.synthesis_mode}\n"
                        f"   Coverage: {compositional_context.coverage:.0%}\n"
                        f"   Precepts found: {len(compositional_context.precepts_found)}\n"
                        f"   Missing: {compositional_context.precepts_missing[:3]}..."
                    )

                # Use compositional context as the main context
                # (it includes hybrid fallback fields)
                from .agent_functions import ContextFetchResult

                context = ContextFetchResult(
                    memories=compositional_context.memories,
                    procedure=compositional_context.procedure,
                    rules=compositional_context.rules,
                    exact_match_solution=compositional_context.exact_match_solution,
                    exact_match_key=compositional_context.exact_match_key,
                    match_tier=compositional_context.match_tier,
                    failed_options=compositional_context.failed_options,
                )

            elif condition_key:
                # Use enhanced 3-tier hybrid fetch for multi-condition scenarios
                context = await fetch_context_with_hybrid(
                    mcp_client=self.mcp_client,
                    query=f"{parsed_task.action} {parsed_task.target or parsed_task.entity}",
                    task_type=f"{self.strategy.domain_name}:{parsed_task.action}",
                    task_description=task,  # Full task for Tier 2 vector similarity
                    condition_key=condition_key,
                    similarity_threshold=0.5,  # 50% minimum overlap for partial matches
                    semaphore=self._internal_semaphore,
                )
                if self.verbose_llm:
                    _get_logger().debug(
                        f"🔀 3-TIER HYBRID FETCH: condition_key={condition_key[:40]}..."
                    )
            else:
                # Standard fetch for simple scenarios
                context = await fetch_context(
                    mcp_client=self.mcp_client,
                    query=f"{parsed_task.action} {parsed_task.target or parsed_task.entity}",
                    task_type=f"{self.strategy.domain_name}:{parsed_task.action}",
                    semaphore=self._internal_semaphore,
                )
            task_steps += 1

            # Apply procedural memory if available
            if apply_procedure_hint(parsed_task, context.procedure):
                overhead_steps += 1

            # Reset interceptor for new task
            self._refine_interceptor.reset()

            # ═══════════════════════════════════════════════════════════════════
            # COMPOSITIONAL DIRECT APPLICATION (Highest-Tier Wins)
            # ═══════════════════════════════════════════════════════════════════
            # When we have full compositional coverage with tier information,
            # we can DETERMINISTICALLY derive the correct solution:
            # - Sort precepts by tier (highest first)
            # - Apply the highest-tier precept's solution directly
            # This is O(1) compositional adaptation - PRECEPT's key advantage!
            # ═══════════════════════════════════════════════════════════════════
            was_rule_applied = False
            compositional_direct_solution = None

            if (
                not self.config.agent.enable_dynamic_tier_resolution
                and compositional_context is not None
                and compositional_context.synthesis_mode
                in ("full_compositional", "hierarchical_compositional")
                and compositional_context.precepts_found
                and len(compositional_context.precepts_found) > 1  # Multi-constraint
            ):
                # Sort precepts by tier (highest first)
                sorted_precepts = sorted(
                    compositional_context.precepts_found,
                    key=lambda p: p.get("tier", 1),
                    reverse=True,
                )

                # Get solution from highest-tier precept
                highest_precept = sorted_precepts[0]
                solution_hint = highest_precept.get("solution_hint", "")

                # Extract just the solution from various hint formats:
                # - "solution:hamburg" → "hamburg"
                # - "solution:LLM→hamburg→singapore" → "hamburg" (first valid port)
                # - "contributed_to:hamburg" → "hamburg"
                if ":" in solution_hint:
                    raw_solution = solution_hint.split(":", 1)[1]
                    # Handle exploration paths like "LLM→hamburg→singapore"
                    if "→" in raw_solution:
                        parts = raw_solution.split("→")
                        # Find first part that looks like a port (not "LLM")
                        for part in parts:
                            if part.lower() != "llm" and part.strip():
                                compositional_direct_solution = part.strip()
                                break
                    else:
                        compositional_direct_solution = raw_solution.strip()
                else:
                    compositional_direct_solution = (
                        solution_hint.strip() if solution_hint else None
                    )

                if compositional_direct_solution:
                    # DIRECT APPLICATION - highest tier wins!
                    parsed_task.parameters["preferred_solution"] = (
                        compositional_direct_solution
                    )
                    was_rule_applied = True
                    strategy_used = (
                        f"compositional:tier{highest_precept.get('tier', 1)}"
                    )

                    if self.verbose_llm:
                        _get_logger().info(
                            f"⚡ COMPOSITIONAL DIRECT: Applying highest-tier solution: "
                            f"{compositional_direct_solution} from [{highest_precept.get('condition')}] "
                            f"(tier={highest_precept.get('tier')}, no LLM reasoning needed)"
                        )
                    use_fast_path = True

            # ═══════════════════════════════════════════════════════════════════
            # HYBRID MATCH: DIRECT APPLICATION (No LLM needed!)
            # ═══════════════════════════════════════════════════════════════════
            # If hybrid fetch found a match (Tier 1, 2, or 3), apply it DIRECTLY
            # This bypasses LLM reasoning for maximum efficiency and accuracy.
            # This is PRECEPT's key advantage over baselines.
            # ═══════════════════════════════════════════════════════════════════
            if context.exact_match_solution and not compositional_direct_solution:
                # DIRECT APPLICATION - no LLM reasoning needed!
                parsed_task.parameters["preferred_solution"] = (
                    context.exact_match_solution
                )
                was_rule_applied = True
                tier = context.match_tier or 1
                tier_names = {1: "exact_match", 2: "vector_similarity", 3: "jaccard"}
                strategy_used = f"tier{tier}:{tier_names.get(tier, 'match')}"
                if self.verbose_llm:
                    tier_desc = {
                        1: "O(1) lookup",
                        2: "vector similarity",
                        3: "Jaccard similarity",
                    }
                    _get_logger().info(
                        f"⚡ TIER {tier} DIRECT: Applying match solution: "
                        f"{context.exact_match_solution} ({tier_desc.get(tier, 'match')}, no LLM)"
                    )
                # Skip LLM reasoning entirely - we have the answer!
                use_fast_path = True

            # ═══════════════════════════════════════════════════════════════════
            # LLM REASONING (Only if no exact match and not fast-path)
            # ═══════════════════════════════════════════════════════════════════
            elif not use_fast_path and (
                self.enable_llm_reasoning or self.force_llm_reasoning
            ):
                # Inject COMPASS constraint context into LLM reasoning
                combined_rules = context.rules
                if constraint_context:
                    combined_rules = f"{constraint_context}\n\n{context.rules}"

                # ═══════════════════════════════════════════════════════════════════
                # COMPOSITIONAL GENERALIZATION: Inject constraint stack for synthesis
                # This is the "Refine Layer" that stacks atomic precepts for the LLM
                # ═══════════════════════════════════════════════════════════════════
                if constraint_stack_prompt and compositional_context:
                    # Prepend constraint stack to rules for LLM synthesis
                    combined_rules = f"{constraint_stack_prompt}\n\n{combined_rules}"
                    if self.verbose_llm:
                        _get_logger().info(
                            f"⚛️ COMPOSITIONAL SYNTHESIS: Stacking {len(compositional_context.constraint_stack)} constraints"
                        )

                # ═══════════════════════════════════════════════════════════════════
                # CRITICAL FIX: Include failed_options in FORBIDDEN section
                # This prevents LLM from suggesting options that already failed!
                # Previously, failed_options were only used in FALLBACK (too late).
                # ═══════════════════════════════════════════════════════════════════
                forbidden_section = ""
                if context.failed_options:
                    forbidden_section = (
                        "\n⚠️ FORBIDDEN OPTIONS (already failed for this scenario):\n"
                        f"DO NOT suggest any of these: {', '.join(context.failed_options)}\n"
                        "These options have been tried and confirmed to fail.\n"
                    )
                    if self.verbose_llm:
                        _get_logger().debug(
                            f"📋 LLM AWARE: {len(context.failed_options)} forbidden options"
                        )

                llm_suggestion = await self._llm_reason_with_evolved_prompt(
                    task=task,
                    parsed_task=parsed_task,
                    memories=context.memories,
                    learned_rules=combined_rules,
                    forbidden_section=forbidden_section,  # NEW: Tell LLM what to avoid!
                    procedure=context.procedure,  # Include procedural memory
                )

                # ═══════════════════════════════════════════════════════════════
                # ENHANCED LOGGING: LLM Suggestion Details
                # ═══════════════════════════════════════════════════════════════
                self._log_llm_suggestion(llm_suggestion, task, parsed_task)

                was_rule_applied, strategy_used = apply_llm_suggestion(
                    parsed_task, llm_suggestion
                )

                # ═══════════════════════════════════════════════════════════════
                # ENHANCED LOGGING: How suggestion was applied
                # ═══════════════════════════════════════════════════════════════
                self._log_suggestion_application(
                    parsed_task, was_rule_applied, strategy_used
                )

                if llm_suggestion:
                    overhead_steps += 1

                # ═══════════════════════════════════════════════════════════════════
                # HARD FILTER: Validate LLM suggestion against valid options
                # LLMs may hallucinate invalid option names (e.g., "order_type_d").
                # We MUST validate against the actual valid options list.
                # ═══════════════════════════════════════════════════════════════════
                preferred = parsed_task.parameters.get("preferred_solution")
                all_options = self.strategy.get_options_for_task(parsed_task)
                all_options_lower = [opt.lower() for opt in all_options]

                # STEP 1: Check if preferred is a VALID option
                if preferred and preferred.lower() not in all_options_lower:
                    if self.verbose_llm:
                        _get_logger().debug(
                            f"⛔ VALIDATION FILTER: LLM suggested '{preferred}' but it's NOT a valid option!"
                        )
                        _get_logger().debug(f"   Valid options are: {all_options}")
                    # Fall back to a random valid option
                    import random

                    remaining = (
                        [
                            opt
                            for opt in all_options
                            if opt.lower()
                            not in [fo.lower() for fo in context.failed_options]
                        ]
                        if context.failed_options
                        else all_options
                    )

                    if remaining:
                        new_option = random.choice(remaining)
                        parsed_task.parameters["preferred_solution"] = new_option
                        if self.verbose_llm:
                            _get_logger().debug(
                                f"🔄 REPLACED with valid untried option: {new_option}"
                            )

                # STEP 2: Check if preferred is in failed_options (even if valid)
                elif preferred and context.failed_options:
                    if preferred.lower() in [
                        fo.lower() for fo in context.failed_options
                    ]:
                        if self.verbose_llm:
                            _get_logger().debug(
                                f"⛔ FAILED FILTER: LLM suggested '{preferred}' but it already failed!"
                            )
                        # Get a fresh option from remaining untried ones
                        import random

                        remaining = [
                            opt
                            for opt in all_options
                            if opt.lower()
                            not in [fo.lower() for fo in context.failed_options]
                        ]
                        if remaining:
                            new_option = random.choice(remaining)
                            parsed_task.parameters["preferred_solution"] = new_option
                            if self.verbose_llm:
                                _get_logger().debug(
                                    f"🔄 REPLACED with untried option: {new_option}"
                                )

            elif use_fast_path:
                if self.verbose_llm:
                    _get_logger().debug("⚡ COMPASS FAST PATH: Skipping LLM reasoning")
                if not strategy_used:
                    strategy_used = "compass:fast_path"

            # Execute action
            action_result = await self.strategy.execute_action(
                self.mcp_client, parsed_task
            )
            task_steps += 1
            final_response = action_result.response

            # ═══════════════════════════════════════════════════════════════════
            # ENHANCED LOGGING: Action Execution Details
            # ═══════════════════════════════════════════════════════════════════
            used_preferred = bool(parsed_task.parameters.get("preferred_solution"))
            self._log_action_execution(parsed_task, action_result, used_preferred)

            if action_result.success:
                success = True
                first_try = True  # Success without error recovery = first-try success
                strategy_used = action_result.strategy_used or strategy_used

                if self.verbose_llm:
                    preferred = parsed_task.parameters.get("preferred_solution")
                    if preferred:
                        _get_logger().debug(
                            f"✅ First-try success with LLM suggestion: {preferred}"
                        )

                # ═══════════════════════════════════════════════════════════════════
                # CRITICAL FIX: Record rule on FIRST-TRY success too!
                # Previously, rules were only recorded during error recovery.
                # This caused learned keys to be "forgotten" on subsequent encounters.
                # ═══════════════════════════════════════════════════════════════════
                condition_key = (parsed_task.parameters or {}).get("condition_key")
                if condition_key and strategy_used:
                    # ═══════════════════════════════════════════════════════════════
                    # FIX: Extract the actual solution from the strategy (GENERIC)
                    # 
                    # Domain strategy formats vary:
                    #   - Logistics: "origin:antwerp", "LLM:origin:antwerp", "customs:enhanced"
                    #   - Finance: "limit", "stop" (direct)
                    #   - Booking: "DL-123", "UA-200" (direct flight IDs)
                    #   - DevOps: "stack:remove_dependencies", "iam:option"
                    #   - Integration: "salesforce-backup" (direct)
                    #   - Coding: "install_package:conda:numpy"
                    #
                    # Generic strategy:
                    #   1. Skip internal strategies (tier, compass, compositional)
                    #   2. For solution-carrying strategies, extract last meaningful part
                    #   3. Fallback to preferred_solution if unclear
                    # ═══════════════════════════════════════════════════════════════
                    solution = None
                    
                    # Patterns that indicate internal routing, not actual solutions
                    skip_patterns = ["tier1", "tier2", "tier3", "compass", "compositional", "Failed"]
                    
                    if ":" in strategy_used:
                        parts = strategy_used.split(":")
                        first_part = parts[0].lower() if parts else ""
                        
                        # Skip internal routing strategies
                        if any(skip in first_part for skip in skip_patterns):
                            solution = parsed_task.parameters.get("preferred_solution")
                        elif "LLM-Reasoned" in strategy_used:
                            # LLM-Reasoned strategy - get from preferred_solution
                            solution = parsed_task.parameters.get("preferred_solution")
                        else:
                            # Solution-carrying strategy: extract last meaningful element
                            # e.g., "origin:antwerp" → "antwerp"
                            # e.g., "LLM:origin:antwerp" → "antwerp"
                            # e.g., "stack:remove_dependencies" → "remove_dependencies"
                            # e.g., "install_package:conda:numpy" → "conda" (manager is the solution)
                            solution = parts[-1]
                            # If last part looks like a retry marker, use second-to-last
                            if solution and ("retry" in solution.lower() or solution.startswith("(")):
                                solution = parts[-2] if len(parts) > 1 else solution
                    else:
                        # No colon - check if it's a valid solution or internal strategy
                        if not any(skip in strategy_used.lower() for skip in skip_patterns):
                            solution = strategy_used
                    
                    # Only record if we have a valid solution
                    if not solution:
                        solution = parsed_task.parameters.get("preferred_solution")
                    
                    if solution:
                        await record_successful_solution(
                            mcp_client=self.mcp_client,
                            error_code=condition_key,
                            solution=solution,
                            context="first-try-success",
                            verbose=self.verbose_llm,
                            domain=self.strategy.domain_name
                            if hasattr(self.strategy, "domain_name")
                            else "general",
                            # Skip atomic conditions in learned_rules when atomic storage enabled
                            skip_atomic_in_learned_rules=self.config.agent.enable_atomic_precept_storage,
                        )
                        if self.verbose_llm:
                            _get_logger().info(
                                f"📝 RULE RECORDED on first-try: {condition_key[:40]}... → {solution}"
                            )
                        learned_rule_event = {
                            "rule_key": condition_key,
                            "solution": solution,
                            "via": "first_try",
                        }

                        # ═══════════════════════════════════════════════════════════════
                        # COMPOSITIONAL GENERALIZATION: Extract atomic precepts
                        # ═══════════════════════════════════════════════════════════════
                        if self.config.agent.enable_atomic_precept_storage:
                            await extract_and_store_atomic_precepts(
                                mcp_client=self.mcp_client,
                                condition_key=condition_key,
                                solution=solution,
                                domain=self.strategy.domain_name
                                if hasattr(self.strategy, "domain_name")
                                else "general",
                            )

            elif not action_result.success:
                # ═══════════════════════════════════════════════════════════════
                # BUGFIX: Trigger error recovery on ANY failure, not just those
                # with structured error codes. Previously, failures without a
                # TE-XXX/R-XXX error_code (e.g., MCP validation errors) silently
                # bypassed the entire recovery pipeline -- no CONSTRAINT marking,
                # no UNLEARN, no COMPASS PROBE, no SMART PIVOT. This capped Pₜ
                # in scenarios where the tool rejected inputs at the validation
                # layer rather than returning domain-specific error codes.
                # ═══════════════════════════════════════════════════════════════
                # Error recovery needed - NOT a first-try success
                # Error recovery with deterministic pruning
                (
                    success,
                    final_response,
                    strategy_used,
                    extra_steps,
                    recovery_learning_event,
                ) = await self._handle_error_recovery(
                    task=task,
                    parsed_task=parsed_task,
                    action_result=action_result,
                    context=context,
                )
                if recovery_learning_event:
                    learned_rule_event = recovery_learning_event
                task_steps += extra_steps
                overhead_steps += 1

            # Store experience and trigger learning
            failure_context = format_failure_context(task, final_response, success)
            condition_key = (parsed_task.parameters or {}).get("condition_key", "")
            counters = await store_experience_and_trigger_learning(
                mcp_client=self.mcp_client,
                task=task,
                success=success,
                strategy=strategy_used,
                domain=self.strategy.domain_name,
                tasks_since_consolidation=self.tasks_since_consolidation,
                tasks_since_compass=self.tasks_since_compass,
                consecutive_failures=self.consecutive_failures,
                config=self.config.agent,
                failure_context=failure_context,
                verbose=self.verbose_llm,
                condition_key=condition_key,
            )

            # ═══════════════════════════════════════════════════════════════════
            # ONLINE VALIDATION: Register task result for real-time COMPASS/GEPA
            # ═══════════════════════════════════════════════════════════════════
            # Instead of using static past tasks for validation, we register the
            # CURRENT task's verified result. This enables:
            # - Dynamic: Always relevant to current training
            # - Generalizable: Works for any domain automatically
            # - Honest: Uses verified signals (success/failure from environment)
            # - Not cheating: COMPASS/GEPA never sees expected_solution
            # ═══════════════════════════════════════════════════════════════════
            try:
                await self.mcp_client.call_tool(
                    "register_task_for_online_validation",
                    {
                        "task": task,
                        "success": success,
                        "steps": task_steps + overhead_steps,
                        "error_code": action_result.error_code
                        if hasattr(action_result, "error_code") and action_result.error_code
                        else "",
                        "error_message": failure_context if not success else "",
                        "domain": self.strategy.domain_name,
                        "strategy": strategy_used,
                    },
                )
            except Exception as e:
                if self.verbose_llm:
                    _get_logger().debug(f"Online validation registration failed: {e}")

            self.tasks_since_consolidation = counters["tasks_since_consolidation"]
            self.tasks_since_compass = counters["tasks_since_compass"]
            self.consecutive_failures = update_failure_counter(
                self.consecutive_failures, success
            )

            # ═══════════════════════════════════════════════════════════════════
            # GEPA PROMPT EVOLUTION CHECK
            # ═══════════════════════════════════════════════════════════════════
            if self.tasks_since_compass == 0:
                prompt_updated = await self.refresh_evolved_prompt()
                if prompt_updated:
                    overhead_steps += 1
                    # Enhanced GEPA evolution logging
                    self._log_gepa_evolution(
                        trigger_reason=f"Interval reached (every {self.compass_evolution_interval} tasks)",
                        tasks_since=self.compass_evolution_interval,
                        new_prompt_preview=self._current_system_prompt[:150]
                        if self._current_system_prompt
                        else "N/A",
                    )

            # Track learning events
            if was_rule_applied:
                self.learning_events.append(f"{task[:50]} → Applied: {strategy_used}")

        except Exception as e:
            _get_logger().error(f"Error in run_task: {e}")
            final_response = f"Error: {e}"
            success = False

        duration = time.time() - start_time

        if success:
            self.successful_tasks += 1
        self.steps_per_task.append(task_steps + overhead_steps)

        # Store for COMPASS scoring
        self.task_results.append(
            build_task_record(
                task=task,
                success=success,
                steps=task_steps,
                overhead=overhead_steps,
                duration=duration,
                strategy=strategy_used,
            )
        )

        return build_task_result(
            success=success,
            task_steps=task_steps,
            overhead_steps=overhead_steps,
            duration=duration,
            response=final_response,
            strategy=strategy_used,
            complexity=complexity,
            domain=self.strategy.domain_name,
            first_try=first_try,
            rule_learned=bool(learned_rule_event),
            learned_rule_key=learned_rule_event["rule_key"] if learned_rule_event else "",
            learned_solution=learned_rule_event["solution"] if learned_rule_event else "",
            learned_via=learned_rule_event["via"] if learned_rule_event else "",
        )

    async def _handle_error_recovery(
        self,
        task: str,
        parsed_task: Any,
        action_result: Any,
        context: Any,
    ) -> tuple:
        """
        Handle error recovery with deterministic pruning and COMPASS evaluation.

        COMPASS integration:
        - Evaluates errors for constraint tier (Physics > Policy > Instruction)
        - Triggers epistemic detours for vague errors
        - Negotiates alternatives when physics blocks

        Returns:
            Tuple of (success, final_response, strategy_used, extra_steps, learning_event)
        """
        # ═══════════════════════════════════════════════════════════════════
        # BUGFIX: Synthesize a fallback error_code when the tool returns a
        # failure without a structured error code (e.g., MCP validation errors,
        # generic rejections). Without this, downstream functions that expect
        # a non-None error_code (record_error, add_constraint, COMPASS
        # evaluate_error, learn_pattern) would receive None, causing silent
        # failures or crashes (e.g., None.lower() in learn_pattern).
        #
        # The synthesized code uses the condition_key if available (preserving
        # the semantic link to the scenario) or a generic "EXEC-FAIL" marker.
        # ═══════════════════════════════════════════════════════════════════
        effective_error_code = action_result.error_code
        if not effective_error_code:
            condition_key = (parsed_task.parameters or {}).get("condition_key", "")
            effective_error_code = (
                f"EXEC-FAIL-{condition_key}" if condition_key else "EXEC-FAIL"
            )
            if self.verbose_llm:
                _get_logger().info(
                    f"🔧 SYNTHESIZED error_code: {effective_error_code} "
                    f"(tool returned failure without structured error code)"
                )

        original_error_code = effective_error_code
        extra_steps = 0

        # Determine failed solution
        failed_solution = parsed_task.parameters.get(
            "preferred_solution", parsed_task.source
        )
        error_context = (
            f"{parsed_task.action} {parsed_task.source}→{parsed_task.target}"
        )

        # Record error (for learning)
        await record_error(self.mcp_client, effective_error_code, error_context)

        # Add constraint (for pruning)
        constraint = add_error_constraint(
            self._refine_interceptor,
            effective_error_code,
            action_result.response,
            failed_solution,
        )

        if self.verbose_llm:
            _get_logger().debug(
                f"    🚫 CONSTRAINT: {failed_solution} → {constraint.constraint_type.value}"
            )

        # ═══════════════════════════════════════════════════════════════════
        # RULE UNLEARNING: Report failure if solution came from a learned rule
        # ═══════════════════════════════════════════════════════════════════
        # If the failed solution came from a learned rule (Tier 1, 2, or 3),
        # report the failure for potential rule invalidation. This enables
        # PRECEPT to adapt when rules become stale (e.g., environment drift).
        # ═══════════════════════════════════════════════════════════════════
        if context and hasattr(context, "exact_match_key") and context.exact_match_key:
            # Solution came from a rule - check if it matches what failed
            if context.exact_match_solution == failed_solution:
                try:
                    from .agent_functions import report_rule_failure
                    invalidation_msg = await report_rule_failure(
                        mcp_client=self.mcp_client,
                        condition_key=context.exact_match_key,
                        failed_solution=failed_solution,
                        error_message=action_result.response,
                        verbose=self.verbose_llm,
                    )
                    if invalidation_msg:
                        _get_logger().warning(
                            f"🗑️ RULE INVALIDATED after failure: {context.exact_match_key[:40]}..."
                        )
                except Exception as e:
                    _get_logger().debug(f"Rule failure reporting failed (non-critical): {e}")

        # ═══════════════════════════════════════════════════════════════════
        # COMPASS ERROR EVALUATION
        # Implements: Epistemic detour, hierarchical constraint discovery
        # ═══════════════════════════════════════════════════════════════════
        compass_error_decision = self._compass_controller.evaluate_error(
            error_code=effective_error_code,
            error_message=action_result.response,
            context={
                "task": task,
                "action": parsed_task.action,
                "parsed_task": parsed_task.parameters,
                "entity": parsed_task.entity,
            },
        )

        # Handle COMPASS decisions
        if compass_error_decision.action == COMPASSAction.PROBE:
            # Epistemic detour: EXECUTE probe through domain strategy
            if compass_error_decision.probe_spec:
                if self.verbose_llm:
                    _get_logger().info(
                        f"🔍 COMPASS PROBE: {compass_error_decision.probe_spec.probe_id}"
                    )
                try:
                    # Execute probe through COMPASS controller
                    probe_result = await self._compass_controller.execute_probe(
                        mcp_client=self.mcp_client,
                        probe_spec=compass_error_decision.probe_spec,
                        context={
                            "parsed_task": parsed_task.parameters,
                            "entity": parsed_task.entity,
                            "error_code": effective_error_code,
                            "error_message": action_result.response,
                        },
                    )

                    # Process probe result
                    probe_decision = self._compass_controller.process_probe_result(
                        probe_result
                    )
                    extra_steps += 1

                    # Learn from probe outcome
                    self._compass_controller.learn_pattern(
                        error_pattern=effective_error_code.lower(),
                        action="probe",
                        succeeded=probe_result.success
                        and probe_result.constraint_discovered is not None,
                        probe_id=compass_error_decision.probe_spec.probe_id,
                    )

                    # Apply probe decision
                    if probe_decision.action == COMPASSAction.BLOCK:
                        if self.verbose_llm:
                            _get_logger().warning(
                                f"🚫 PROBE REVEALED: {probe_decision.blocking_constraint}"
                            )
                        return (
                            False,
                            f"Probe revealed physics constraint: {probe_decision.blocking_constraint}",
                            f"compass:probe_block:{probe_decision.blocking_constraint}",
                            extra_steps,
                            None,
                        )

                    if (
                        probe_decision.action == COMPASSAction.PIVOT
                        and probe_decision.negotiated_alternative
                    ):
                        parsed_task.parameters["preferred_solution"] = (
                            probe_decision.negotiated_alternative
                        )
                        if self.verbose_llm:
                            _get_logger().info(
                                f"🔄 PROBE PIVOT: {probe_decision.negotiated_alternative}"
                            )

                except Exception as e:
                    if self.verbose_llm:
                        _get_logger().warning(f"Probe execution failed: {e}")
                    extra_steps += 1
            else:
                if self.verbose_llm:
                    _get_logger().debug("COMPASS suggested probe but no probe spec")
                extra_steps += 1

        if compass_error_decision.action == COMPASSAction.BLOCK:
            # Physics constraint discovered - stop trying
            if self.verbose_llm:
                _get_logger().warning(
                    f"🚫 COMPASS BLOCK (Physics): {compass_error_decision.blocking_constraint}"
                )
            return (
                False,
                f"Blocked by physics constraint: {compass_error_decision.blocking_constraint}",
                f"compass:physics_block:{compass_error_decision.blocking_constraint}",
                extra_steps,
                None,
            )

        if compass_error_decision.action == COMPASSAction.PIVOT:
            # Use negotiated alternative from COMPASS
            if compass_error_decision.negotiated_alternative:
                parsed_task.parameters["preferred_solution"] = (
                    compass_error_decision.negotiated_alternative
                )
                if self.verbose_llm:
                    _get_logger().info(
                        f"🔄 COMPASS PIVOT: {compass_error_decision.negotiated_alternative}"
                    )

        # Get constraint context for LLM (temporal abstraction)
        compass_constraint_context = compass_error_decision.constraint_context

        # Legacy diagnostic probe (for logging)
        if self.verbose_llm:
            probe = self._refine_interceptor.suggest_diagnostic_probe(
                effective_error_code,
                action_result.response,
            )
            if probe:
                _get_logger().debug(f"🔍 DIAGNOSTIC PROBE: {probe}")

        # Smart pivot loop - use config.agent.max_retries (single source of truth)
        # This aligns with baseline's config.max_attempts for equal exploration budget
        max_pivots = self.max_retries
        last_error = action_result.response
        learning_event: Optional[Dict[str, str]] = None

        for pivot_num in range(max_pivots):
            # Get forbidden injection (in-episode failures)
            forbidden_injection = self._refine_interceptor.get_forbidden_injection()

            # ═══════════════════════════════════════════════════════════════════
            # CRITICAL FIX: Also include cross-episode failed options!
            # The forbidden_injection only contains in-episode failures.
            # We need to ALSO include context.failed_options from partial_progress.
            # ═══════════════════════════════════════════════════════════════════
            if context.failed_options:
                partial_progress_warning = (
                    "\n⚠️ CROSS-EPISODE FORBIDDEN (failed in previous encounters):\n"
                    f"DO NOT suggest: {', '.join(context.failed_options)}\n"
                )
                forbidden_injection = (
                    f"{forbidden_injection}\n{partial_progress_warning}"
                    if forbidden_injection
                    else partial_progress_warning
                )

            # ═══════════════════════════════════════════════════════════════════
            # COMPASS TEMPORAL ABSTRACTION: Inject constraint context into LLM
            # This maintains strategic context across retries
            # ═══════════════════════════════════════════════════════════════════
            combined_rules = context.rules
            if compass_constraint_context:
                combined_rules = f"{compass_constraint_context}\n\n{context.rules}"

            # Ask LLM for new suggestion
            new_suggestion = None
            if self.enable_llm_reasoning:
                llm_pivot_suggestion = await self._llm_reason_with_evolved_prompt(
                    task=task,
                    parsed_task=parsed_task,
                    memories=context.memories,
                    learned_rules=combined_rules,
                    error_feedback=last_error,
                    forbidden_section=forbidden_injection,
                    procedure=context.procedure,  # Include procedural memory
                )
                if llm_pivot_suggestion:
                    suggested = llm_pivot_suggestion.get("suggested_solution")

                    # Fix A: Configurable EXHAUSTED handling
                    # When disable_exhausted_exit=True, ignore LLM "EXHAUSTED" signal
                    # and continue with random/remaining options exploration
                    if suggested and suggested.upper() == "EXHAUSTED":
                        if self.verbose_llm:
                            _get_logger().debug(
                                "⛔ LLM says EXHAUSTED - checking config..."
                            )
                        if not self.config.agent.disable_exhausted_exit:
                            # Original behavior: stop immediately
                            if self.verbose_llm:
                                _get_logger().debug(
                                    "   → Exiting (disable_exhausted_exit=False)"
                                )
                            break
                        else:
                            # Fix A: Continue with fallback exploration
                            if self.verbose_llm:
                                _get_logger().debug(
                                    "   → Continuing with fallback (disable_exhausted_exit=True)"
                                )
                            # Don't set new_suggestion - let fallback handle it

                    elif suggested and not self._refine_interceptor.is_forbidden(
                        suggested
                    ):
                        # ═══════════════════════════════════════════════════════════
                        # VALIDATION FIX: Check if suggestion is a valid option!
                        # LLMs may hallucinate invalid options like "order_type_d"
                        # ═══════════════════════════════════════════════════════════
                        pivot_all_options = self.strategy.get_options_for_task(
                            parsed_task
                        )
                        pivot_options_lower = [opt.lower() for opt in pivot_all_options]

                        if suggested.lower() not in pivot_options_lower:
                            if self.verbose_llm:
                                _get_logger().debug(
                                    f"⛔ PIVOT VALIDATION: '{suggested}' is NOT a valid option!"
                                )
                            # Don't set new_suggestion - let fallback handle it
                        else:
                            # Check cross-episode failed_options
                            is_cross_episode_forbidden = (
                                context.failed_options
                                and suggested.lower()
                                in [fo.lower() for fo in context.failed_options]
                            )
                            if is_cross_episode_forbidden:
                                if self.verbose_llm:
                                    _get_logger().debug(
                                        f"⛔ CROSS-EPISODE BLOCKED: '{suggested}' failed in previous encounter"
                                    )
                                # Don't set new_suggestion - let fallback handle it
                            else:
                                new_suggestion = suggested
                                parsed_task.parameters["preferred_solution"] = suggested
                                extra_steps += 1
                    elif suggested:
                        self._refine_interceptor.record_prevented_retry()
                        if self.verbose_llm:
                            _get_logger().debug(
                                f"⚡ DUMB RETRY PREVENTED: '{suggested}' is forbidden"
                            )

            # Fallback to remaining options
            if not new_suggestion:
                all_options = self.strategy.get_options_for_task(parsed_task)
                # Options are already shuffled by domain strategy for fair exploration
                remaining = self._refine_interceptor.get_remaining_options(all_options)

                # ═══════════════════════════════════════════════════════════════
                # PARTIAL PROGRESS: Also skip options that failed during training
                # This allows resuming from where training left off
                # ═══════════════════════════════════════════════════════════════
                if context.failed_options:
                    remaining = [
                        opt
                        for opt in remaining
                        if opt.lower()
                        not in [fo.lower() for fo in context.failed_options]
                    ]
                    if self.verbose_llm and context.failed_options:
                        _get_logger().debug(
                            f"📋 RESUME: Skipping {len(context.failed_options)} previously failed options"
                        )

                if remaining:
                    new_suggestion = remaining[0]
                    parsed_task.parameters["preferred_solution"] = new_suggestion
                    if self.verbose_llm:
                        _get_logger().debug(
                            f"🔀 FALLBACK: Trying option: {new_suggestion}"
                        )
                else:
                    # Fix C: Configurable random fallback
                    # When enable_random_fallback=True, choose randomly from ALL options
                    # like Full Reflexion does, instead of giving up
                    if self.config.agent.enable_random_fallback:
                        import random

                        new_suggestion = random.choice(all_options)
                        parsed_task.parameters["preferred_solution"] = new_suggestion
                        if self.verbose_llm:
                            _get_logger().debug(
                                f"🎲 RANDOM FALLBACK: Trying random option: {new_suggestion}"
                            )
                    else:
                        # Original behavior: stop immediately
                        if self.verbose_llm:
                            _get_logger().debug(
                                "⏹️ All options exhausted - exiting pivot loop"
                            )
                        break

            if self.verbose_llm:
                _get_logger().debug(
                    f"🔄 SMART PIVOT #{pivot_num + 1}: {new_suggestion}"
                )

            # Execute pivot
            pivot_result = await self.strategy.execute_action(
                self.mcp_client, parsed_task
            )
            extra_steps += 1

            if pivot_result.success:
                # Record successful solution
                working_solution = parsed_task.parameters.get(
                    "preferred_solution", new_suggestion
                )
                if original_error_code and working_solution:
                    # ═══════════════════════════════════════════════════════════
                    # MULTI-CONDITION: Use condition_key if available
                    # ═══════════════════════════════════════════════════════════
                    # For multi-condition scenarios, use the full condition key
                    # e.g., "P-220+R-482 → antwerp" instead of "R-482 → antwerp"
                    condition_key = (parsed_task.parameters or {}).get("condition_key")
                    rule_key = condition_key if condition_key else original_error_code

                    await record_successful_solution(
                        mcp_client=self.mcp_client,
                        error_code=rule_key,  # Multi-condition key or single error code
                        solution=working_solution,
                        context="",  # Simple format for better LLM parsing
                        verbose=self.verbose_llm,
                        domain=self.strategy.domain_name
                        if hasattr(self.strategy, "domain_name")
                        else "general",
                        # Skip atomic conditions in learned_rules when atomic storage enabled
                        skip_atomic_in_learned_rules=self.config.agent.enable_atomic_precept_storage,
                    )

                    # Track rule learned via pivot
                    self.learning_events.append(
                        f"{task[:30]}... → Learned: {rule_key} → {working_solution}"
                    )
                    learning_event = {
                        "rule_key": rule_key,
                        "solution": working_solution,
                        "via": "pivot",
                    }

                    # ═══════════════════════════════════════════════════════════
                    # PROCEDURAL MEMORY: Store successful recovery as a procedure
                    # This enables the agent to learn "how-to" strategies dynamically
                    # ═══════════════════════════════════════════════════════════
                    try:
                        task_type = f"{self.strategy.domain_name}:{parsed_task.action}"
                        procedure_name = (
                            f"recovery_{original_error_code}_{working_solution}"
                        )
                        procedure_steps = (
                            f"1. Attempt default action for {parsed_task.action}\n"
                            f"2. If error '{original_error_code}' occurs, use '{working_solution}'\n"
                            f"3. Learned from: {task[:50]}..."
                        )
                        await self.mcp_client.call_tool(
                            "store_procedure",
                            {
                                "name": procedure_name,
                                "task_type": task_type,
                                "steps": procedure_steps,
                            },
                        )
                        if self.verbose_llm:
                            _get_logger().debug(
                                f"📋 PROCEDURAL MEMORY: Stored recovery procedure for {original_error_code}"
                            )
                    except Exception:
                        pass  # Don't fail if procedure storage fails

                    # ═══════════════════════════════════════════════════════════
                    # COMPOSITIONAL GENERALIZATION: Extract atomic precepts
                    # ═══════════════════════════════════════════════════════════
                    if self.config.agent.enable_atomic_precept_storage and rule_key:
                        await extract_and_store_atomic_precepts(
                            mcp_client=self.mcp_client,
                            condition_key=rule_key,
                            solution=working_solution,
                            domain=self.strategy.domain_name
                            if hasattr(self.strategy, "domain_name")
                            else "general",
                        )

                # Log successful pivot for debugging
                if self.verbose_llm:
                    _get_logger().debug(
                        f"✅ PIVOT SUCCESS: {original_error_code} → {working_solution}"
                    )
                return (
                    True,
                    pivot_result.response,
                    pivot_result.strategy_used or f"SmartPivot:{pivot_num + 1}",
                    extra_steps,
                    learning_event,
                )
            else:
                last_error = pivot_result.response
                failed_option = parsed_task.parameters.get(
                    "preferred_solution", new_suggestion
                )
                # ═══════════════════════════════════════════════════════════
                # BUGFIX: Always add constraints on pivot failure, even
                # without a structured error_code. Synthesize one if needed.
                # Previously, pivots that failed without error_code would
                # NOT be added to the forbidden list, allowing the LLM to
                # retry the same failed option in the next pivot iteration.
                # ═══════════════════════════════════════════════════════════
                pivot_error_code = pivot_result.error_code or f"PIVOT-FAIL-{pivot_num + 1}"
                self._refine_interceptor.add_constraint(
                    solution=failed_option,
                    error_code=pivot_error_code,
                    error_message=pivot_result.response,
                )
                await self.mcp_client.record_error(
                    pivot_error_code,
                    f"Pivot {pivot_num + 1}: {parsed_task.action}",
                )

                # ═══════════════════════════════════════════════════════════
                # PARTIAL PROGRESS: Record failed option for future resume
                # ═══════════════════════════════════════════════════════════
                condition_key = (parsed_task.parameters or {}).get("condition_key")
                if condition_key and failed_option:
                    try:
                        await self.mcp_client.call_tool(
                            "record_failed_option",
                            {
                                "condition_key": condition_key,
                                "failed_option": failed_option,
                                "error_code": pivot_error_code,
                            },
                        )
                    except Exception:
                        pass  # Don't fail if partial progress recording fails

        # All pivots exhausted - agent must learn through experience, not hardcoded knowledge
        if self.verbose_llm:
            _get_logger().debug(
                f"❌ ALL PIVOTS EXHAUSTED: {original_error_code} (max_pivots={max_pivots})"
            )
        return (False, last_error, "", extra_steps, None)

    # =========================================================================
    # STATISTICS AND GETTERS
    # =========================================================================

    def get_success_rate(self) -> float:
        """Get the current success rate using pure function."""
        from .agent_functions import compute_success_rate

        return compute_success_rate(self.successful_tasks, self.total_tasks)

    def get_compass_stats(self) -> Dict[str, Any]:
        """Get COMPASS-specific statistics including controller stats."""
        stats = {}

        # Get MCP client COMPASS stats
        if self.mcp_client and hasattr(self.mcp_client, "compass_stats"):
            stats["mcp_compass"] = self.mcp_client.compass_stats

        # Get COMPASS controller stats (hierarchical constraints, probes, etc.)
        if self._compass_controller:
            stats["controller"] = self._compass_controller.get_stats()

        return stats

    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive statistics using pure functions."""
        from .agent_functions import compute_average, compute_success_rate

        compass_stats = self.get_compass_stats()
        scores = {}
        if self.task_results:
            scores = compute_scores_from_task_results(self.task_results)

        return {
            "total_tasks": self.total_tasks,
            "successful_tasks": self.successful_tasks,
            "success_rate": compute_success_rate(
                self.successful_tasks, self.total_tasks
            ),
            "avg_steps": compute_average([float(s) for s in self.steps_per_task]),
            "domain": self.strategy.domain_name,
            "tasks_since_consolidation": self.tasks_since_consolidation,
            "tasks_since_compass": self.tasks_since_compass,
            "consecutive_failures": self.consecutive_failures,
            "learning_events": len(self.learning_events),
            "compass_stats": compass_stats,
            "compass_scores": scores,
            "prompt_generation": self._prompt_generation,
            "prompt_updated_at_task": self._prompt_updated_at_task,
            "has_evolved_prompt": self._prompt_generation > 0,
            "llm_reasoning_calls": self._llm_reasoning_calls,
            "llm_reasoning_successes": self._llm_reasoning_successes,
            "llm_reasoning_failures": self._llm_reasoning_failures,
            "pruning_stats": self._pruning_stats,
            "dumb_retries_prevented": self._pruning_stats["dumb_retries_prevented"],
            "total_constraints": self._pruning_stats["total_constraints"],
        }

    def get_llm_reasoning_stats(self) -> Dict[str, Any]:
        """Get statistics about LLM reasoning usage using pure functions."""
        from .agent_functions import compute_success_rate

        return {
            "total_calls": self._llm_reasoning_calls,
            "successes": self._llm_reasoning_successes,
            "failures": self._llm_reasoning_failures,
            "success_rate": compute_success_rate(
                self._llm_reasoning_successes, self._llm_reasoning_calls
            ),
            "enabled": self.enable_llm_reasoning,
            "forced": self.force_llm_reasoning,
            "prompt_generation": self._prompt_generation,
        }

    def get_pruning_stats(self) -> Dict[str, Any]:
        """Get statistics about DETERMINISTIC PRUNING using pure functions."""
        stats = self._pruning_stats
        # Pure calculation for efficiency
        efficiency = stats["dumb_retries_prevented"] / max(
            1, stats["total_constraints"]
        )
        return {
            "total_constraints": stats["total_constraints"],
            "hard_constraints": stats["hard_constraints"],
            "soft_constraints": stats["soft_constraints"],
            "dumb_retries_prevented": stats["dumb_retries_prevented"],
            "diagnostic_probes": stats["diagnostic_probes"],
            "pruning_efficiency": efficiency,
        }

    # =========================================================================
    # SESSION MANAGEMENT
    # =========================================================================

    def start_session(self, session_id: Optional[str] = None) -> str:
        """Start a new conversation session."""
        import uuid

        self._session_id = session_id or str(uuid.uuid4())[:8]
        self._session_start_time = time.time()
        self._session_messages: List[Dict[str, Any]] = []
        self._session_task_count = 0
        self._session_successes = 0

        return self._session_id

    def end_session(self, store_experience: bool = True) -> Dict[str, Any]:
        """End the current session and optionally store the experience."""
        if not hasattr(self, "_session_id"):
            return {"error": "No active session"}

        session_duration = time.time() - getattr(self, "_session_start_time", 0)
        session_stats = {
            "session_id": self._session_id,
            "duration": session_duration,
            "messages": len(getattr(self, "_session_messages", [])),
            "tasks": getattr(self, "_session_task_count", 0),
            "successes": getattr(self, "_session_successes", 0),
            "success_rate": self._calculate_session_success_rate(),
        }

        self._session_id = None
        self._session_start_time = None
        self._session_messages = []
        self._session_task_count = 0
        self._session_successes = 0

        return session_stats

    def _calculate_session_success_rate(self) -> float:
        """Calculate session success rate using pure function."""
        from .agent_functions import compute_success_rate

        task_count = getattr(self, "_session_task_count", 0)
        successes = getattr(self, "_session_successes", 0)
        return compute_success_rate(successes, task_count)

    async def chat(self, message: str, apply_learning: bool = True) -> str:
        """Chat with the agent in a conversational way."""
        if hasattr(self, "_session_messages"):
            self._session_messages.append({"role": "user", "content": message})

        # Check if this looks like a task
        task_keywords = ["book", "ship", "install", "deploy", "run", "execute", "clear"]
        is_task = any(kw in message.lower() for kw in task_keywords)

        if is_task:
            result = await self.run_task(message)
            if hasattr(self, "_session_task_count"):
                self._session_task_count += 1
                if result.get("success"):
                    self._session_successes += 1
            response = result.get("response", "Task completed.")
        else:
            memories = await self.mcp_client.retrieve_memories(message, top_k=3)
            rules = await self.mcp_client.get_learned_rules()

            context = f"""Based on my knowledge and memories:

Relevant Memories:
{memories}

Learned Rules:
{rules}

Please help with: {message}"""

            try:
                from autogen_core.models import SystemMessage, UserMessage

                chat_response = await self.model_client.create(
                    messages=[
                        SystemMessage(content=self._current_system_prompt),
                        UserMessage(content=context, source="user"),
                    ],
                    extra_create_args={"max_tokens": 500},
                )
                response = (
                    chat_response.content
                    if hasattr(chat_response, "content")
                    else str(chat_response)
                )
            except ImportError:
                response = "Unable to process: AutoGen core not available"
            except Exception as e:
                response = f"I encountered an error: {e}"

        if hasattr(self, "_session_messages"):
            self._session_messages.append({"role": "assistant", "content": response})

        return response

    async def chat_stream(self, message: str, apply_learning: bool = True):
        """Stream a chat response."""
        response = await self.chat(message, apply_learning)
        yield response

    def get_conversation_history(self) -> List[Dict[str, Any]]:
        """Get the current conversation history."""
        return getattr(self, "_session_messages", [])

    def reset_conversation(self) -> None:
        """Reset the conversation history."""
        self._session_messages = []
        self._session_task_count = 0
        self._session_successes = 0
