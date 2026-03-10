"""
AutoGen PRECEPT Agent with COMPASS Advantages.

This module provides a generic AutoGen agent with PRECEPT learning capabilities
and COMPASS advantages (ML complexity analysis, smart rollouts, etc.).

Uses the Strategy Pattern for domain-specific behavior.
Works with ANY black swan category by injecting the appropriate strategy.

Features:
- ML-based complexity analysis (COMPASS)
- Smart rollout allocation (COMPASS)
- Dynamic rule learning (PRECEPT)
- Multi-strategy coordination
- Docker-based code execution for coding domain (optional)

Usage:
    from precept import PRECEPTAgent
    from precept.domain_strategies import LogisticsDomainStrategy

    # For logistics
    agent = PRECEPTAgent(domain_strategy=LogisticsDomainStrategy())
    await agent.connect()
    result = await agent.run_task("Book shipment from Rotterdam to Boston")

    # For coding with Docker execution
    from precept.domain_strategies import CodingDomainStrategy
    strategy = CodingDomainStrategy(enable_docker_execution=True)
    agent = PRECEPTAgent(domain_strategy=strategy)
"""

import asyncio
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

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

from .domain_strategies.base import DomainStrategy
from .precept_orchestrator import PRECEPTConfig
from .scoring import compute_scores_from_task_results


class PRECEPTAgent:
    """
    Generic AutoGen agent with PRECEPT learning + COMPASS advantages.

    Uses the Strategy Pattern for domain-specific behavior.
    Works with ANY black swan category by injecting the appropriate strategy.

    Features:
    - ML-based complexity analysis (COMPASS)
    - Smart rollout allocation (COMPASS)
    - Dynamic rule learning (PRECEPT)
    - Multi-strategy coordination
    - Docker-based code execution for coding domain (optional)
    - Dynamic configuration updates from real execution feedback

    Usage:
        # For logistics
        agent = PRECEPTAgent(domain_strategy=LogisticsDomainStrategy())

        # For coding (simulated execution)
        agent = PRECEPTAgent(domain_strategy=CodingDomainStrategy())

        # For coding with Docker execution (real code execution)
        strategy = CodingDomainStrategy(enable_docker_execution=True)
        agent = PRECEPTAgent(domain_strategy=strategy)

        # For any custom domain
        agent = PRECEPTAgent(domain_strategy=MyCustomDomainStrategy())
    """

    def __init__(
        self,
        domain_strategy: DomainStrategy,
        model: str = "gpt-4o-mini",
        precept_config: Optional[PRECEPTConfig] = None,
        server_script: Optional[Path] = None,
    ):
        """
        Initialize the AutoGen PRECEPT agent.

        Args:
            domain_strategy: The domain strategy to use (determines domain behavior)
            model: The OpenAI model to use
            precept_config: Optional PRECEPT configuration
            server_script: Path to the MCP server script (defaults to precept_mcp_server.py)
        """
        if not AUTOGEN_AVAILABLE:
            raise ImportError(
                "AutoGen dependencies not available. "
                "Install with: pip install autogen-agentchat autogen-ext mcp"
            )

        self.strategy = domain_strategy
        self.model = model
        self.mcp_client = None
        self.agent: Optional[AssistantAgent] = None
        self.model_client: Optional[OpenAIChatCompletionClient] = None

        # Server script path
        if server_script is None:
            self.server_script = Path(__file__).parent / "precept_mcp_server.py"
        else:
            self.server_script = server_script

        # ─── PRECEPT Configuration ───
        self.precept_config = precept_config or PRECEPTConfig(
            consolidation_interval=3,
            compass_evolution_interval=2,
            max_memories=1000,
            enable_compass_optimization=True,
        )

        # Interval thresholds from config
        self.CONSOLIDATION_INTERVAL = self.precept_config.consolidation_interval
        self.COMPASS_EVOLUTION_INTERVAL = self.precept_config.compass_evolution_interval
        self.FAILURE_THRESHOLD = 2

        # Statistics
        self.total_tasks = 0
        self.successful_tasks = 0
        self.steps_per_task: List[int] = []

        # COMPASS tracking
        self.tasks_since_consolidation = 0
        self.tasks_since_compass = 0
        self.consecutive_failures = 0

        # Task results for GEPA scoring
        self.task_results: List[Dict] = []
        self.learning_events: List[str] = []

        # ═══════════════════════════════════════════════════════════════════════
        # COMPASS PROMPT EVOLUTION - THE KEY PRECEPT ADVANTAGE
        # ═══════════════════════════════════════════════════════════════════════
        # This tracks the current system prompt which evolves during training
        # COMPASS advantages over basic approaches:
        # - ML-based complexity analysis
        # - Smart rollout allocation
        # - Dynamic prompt evolution with learned rules
        self._current_system_prompt: str = ""
        self._prompt_generation: int = 0  # Track COMPASS evolution generation
        self._prompt_updated_at_task: int = 0

    async def connect(self):
        """Connect to MCP server and initialize agent."""
        # Import here to avoid circular imports
        from .compass_mcp_client import PRECEPTMCPClientWithCOMPASS

        print(
            f"🚀 Initializing AutoGen PRECEPT Agent for [{self.strategy.domain_name}]..."
        )

        # Get project root for PYTHONPATH
        project_root = self.server_script.parent.parent.parent

        # Connect using the COMPASS-enhanced client
        self.mcp_client = PRECEPTMCPClientWithCOMPASS(
            server_params=StdioServerParameters(
                command="python3",
                args=[str(self.server_script)],
                env={**os.environ, "PYTHONPATH": str(project_root / "src")},
            )
        )
        await self.mcp_client.connect()

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

        print("  ✓ Connected to MCP server")
        print(f"  ✓ Domain: {self.strategy.domain_name}")
        print(f"  ✓ Actions: {self.strategy.get_available_actions()}")
        print("  ✓ COMPASS Advantages enabled:")
        print("    • PRECEPTComplexityAnalyzer (ML-based)")
        print("    • SmartRolloutStrategy (adaptive)")
        print("    • MultiStrategyCoordinator")
        print("    • Dynamic Prompt Evolution (COMPASS)")

        # Show Docker execution status for coding domain
        if self.strategy.domain_name == "coding":
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
                    print("    • Docker Code Execution (sandboxed)")
                else:
                    print("    • Subprocess Fallback (Docker unavailable)")
                print("    • Dynamic Learning from Execution")
            else:
                print("    • Simulated Execution (MCP-based)")

        print(f"✅ AutoGen PRECEPT Agent [{self.strategy.domain_name}] ready")

    async def refresh_evolved_prompt(self) -> bool:
        """
        Refresh the system prompt with evolved version from COMPASS.

        THIS IS THE KEY PRECEPT ADVANTAGE:
        - Fetches the best evolved prompt from COMPASS
        - Includes all learned rules and domain mappings
        - Recreates the agent with the new prompt

        COMPASS advantages over basic approaches:
        - ML-based complexity analysis guides evolution
        - Smart rollout allocation (fewer wasted attempts)
        - Dynamic prompt evolution with consolidated wisdom

        Returns:
            True if prompt was updated, False otherwise
        """
        if not self.mcp_client:
            return False

        try:
            # Get evolved prompt from GEPA
            evolved_prompt = await self.mcp_client.get_evolved_prompt(
                include_rules=True
            )

            if evolved_prompt and evolved_prompt != self._current_system_prompt:
                # Store old prompt for comparison
                old_prompt_preview = self._current_system_prompt[:100]

                # Update current prompt
                self._current_system_prompt = evolved_prompt
                self._prompt_generation += 1
                self._prompt_updated_at_task = self.total_tasks

                # Recreate agent with evolved prompt
                autogen_tools = self.strategy.create_autogen_tools(self.mcp_client)
                self.agent = AssistantAgent(
                    name=f"PRECEPT_{self.strategy.category.value}_Agent",
                    model_client=self.model_client,
                    tools=autogen_tools,
                    system_message=self._current_system_prompt,
                )

                self.learning_events.append(
                    f"PROMPT EVOLVED at task #{self.total_tasks} (gen {self._prompt_generation})"
                )
                print(f"    🧬 Prompt evolved! Generation {self._prompt_generation}")
                return True

        except Exception as e:
            print(f"    ⚠️ Prompt refresh failed: {e}")

        return False

    def get_current_prompt(self) -> str:
        """Get the current (potentially evolved) system prompt."""
        return self._current_system_prompt

    def get_prompt_stats(self) -> Dict[str, Any]:
        """Get statistics about prompt evolution."""
        return {
            "generation": self._prompt_generation,
            "last_updated_at_task": self._prompt_updated_at_task,
            "prompt_length": len(self._current_system_prompt),
            "has_learned_rules": "LEARNED RULES" in self._current_system_prompt,
            "has_domain_knowledge": "DOMAIN-SPECIFIC" in self._current_system_prompt,
        }

    async def _llm_reason_with_evolved_prompt(
        self,
        task: str,
        parsed_task: Any,
        memories: str,
    ) -> Optional[Dict[str, Any]]:
        """
        Use LLM to reason about task using evolved prompt with learned rules.

        THIS IS THE KEY PRECEPT ADVANTAGE:
        - The evolved prompt contains consolidated wisdom from training
        - LLM can generalize beyond exact pattern matches
        - Enables flexible application of learned rules

        Per Google's Context Engineering whitepaper:
        - "Memory-as-a-Tool" - agent decides when to retrieve
        - "Relevance" - what specific information belongs in the model's window now

        Args:
            task: The raw task string
            parsed_task: The parsed task object
            memories: Retrieved memories from episodic store

        Returns:
            Dict with suggested_solution, reasoning, and confidence, or None
        """
        if not self._current_system_prompt or not self.model_client:
            return None

        try:
            # Get valid options for this task type (domain-specific)
            valid_options = []
            if hasattr(self.strategy, "get_options_for_task"):
                valid_options = self.strategy.get_options_for_task(parsed_task)
            elif hasattr(self.strategy, "get_available_options"):
                valid_options = self.strategy.get_available_options()

            options_str = (
                ", ".join(valid_options)
                if valid_options
                else "No specific options available"
            )

            # Build the reasoning prompt with evolved context
            reasoning_prompt = f"""Based on your learned rules and domain knowledge, analyze this task:

TASK: {task}
PARSED: action={parsed_task.action}, entity={parsed_task.entity}

VALID OPTIONS FOR THIS TASK:
{options_str}

RELEVANT MEMORIES:
{memories[:500] if memories else "No relevant memories found."}

QUESTION: Based on your learned rules, which option from VALID OPTIONS should be used?
If you recognize a pattern from your learned rules, suggest the appropriate option.
If unsure, respond with "EXPLORE" to try default approach.

IMPORTANT: You MUST suggest an option from the VALID OPTIONS list above, or EXPLORE.

Respond in this format:
SOLUTION: <option_name from VALID OPTIONS or EXPLORE>
REASONING: <brief explanation>
CONFIDENCE: <high/medium/low>"""

            # Use the model client to reason

            response = await self.model_client.create(
                messages=[
                    {"role": "system", "content": self._current_system_prompt},
                    {"role": "user", "content": reasoning_prompt},
                ],
                extra_create_args={"max_tokens": 200},
            )

            # Parse LLM response
            response_text = (
                response.content if hasattr(response, "content") else str(response)
            )

            # Extract solution suggestion
            if "SOLUTION:" in response_text and "EXPLORE" not in response_text.upper():
                lines = response_text.split("\n")
                solution = None
                reasoning = ""
                confidence = "medium"

                for line in lines:
                    if line.startswith("SOLUTION:"):
                        solution = line.replace("SOLUTION:", "").strip()
                    elif line.startswith("REASONING:"):
                        reasoning = line.replace("REASONING:", "").strip()
                    elif line.startswith("CONFIDENCE:"):
                        confidence = line.replace("CONFIDENCE:", "").strip().lower()

                if solution and solution.lower() != "explore":
                    return {
                        "suggested_solution": solution,
                        "reasoning": reasoning,
                        "confidence": confidence,
                    }

            return None

        except Exception:
            # Silently handle errors - fall back to programmatic approach
            return None

    async def _try_fast_path(
        self,
        task: str,
        parsed_task: Any,  # ParsedTask from domain strategy
    ) -> Optional[Dict[str, Any]]:
        """
        Fast path for trivial tasks: Simple LLM code generation + direct execution.

        This bypasses the full PRECEPT pipeline (memories, rules, COMPASS) for speed.
        Used when:
        - Complexity analysis indicates "trivial" task
        - We want to reduce LLM calls for simple operations

        If fast path fails, caller should engage full PRECEPT pipeline.

        Args:
            task: Original task string
            parsed_task: Parsed task from strategy

        Returns:
            Dict with success/response if succeeded, None if failed
        """
        try:
            from openai import AsyncOpenAI

            client = AsyncOpenAI()
            try:
                # Simple prompt for code generation (no memories, no rules)
                prompt = f"""Generate Python code to accomplish this task. Return ONLY executable Python code.

Task: {task}

Requirements:
- Include proper error handling
- Print success/error messages
- Use sys.exit(1) on failure
- Import required modules at the top
"""

                response = await client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.2,
                    max_tokens=800,
                )

                code = response.choices[0].message.content.strip()

                # Extract code from markdown if present
                if "```python" in code:
                    code = code.split("```python")[1].split("```")[0].strip()
                elif "```" in code:
                    code = code.split("```")[1].split("```")[0].strip()

                # Execute via strategy's Docker execution (if available)
                if (
                    hasattr(self.strategy, "_execute_with_docker")
                    and hasattr(self.strategy, "enable_docker_execution")
                    and self.strategy.enable_docker_execution
                ):
                    # Create a temporary parsed task with the generated code
                    temp_task = parsed_task
                    temp_task.parameters["code"] = code

                    result = await self.strategy._execute_with_docker(
                        temp_task, self.mcp_client
                    )

                    if result.success:
                        return {
                            "success": True,
                            "response": result.response,
                            "strategy": "fast_path:docker",
                        }
                else:
                    # Fallback: subprocess execution
                    import subprocess
                    import tempfile

                    with tempfile.NamedTemporaryFile(
                        mode="w", suffix=".py", delete=False
                    ) as f:
                        f.write(code)
                        temp_file = f.name

                    try:
                        result = subprocess.run(
                            ["python3", temp_file],
                            capture_output=True,
                            text=True,
                            timeout=30,
                        )

                        if result.returncode == 0:
                            return {
                                "success": True,
                                "response": result.stdout,
                                "strategy": "fast_path:subprocess",
                            }
                    finally:
                        import os

                        os.unlink(temp_file)

                return None  # Fast path failed
            finally:
                try:
                    await client.close()
                except Exception:
                    pass

        except Exception:
            # Fast path failed - return None to trigger full PRECEPT
            return None

    async def disconnect(self):
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

    async def run_task(
        self,
        task: str,
        fast_path: bool = True,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Run a task using PRECEPT+COMPASS learning with GENERIC strategy-driven flow.

        HYBRID APPROACH (for speed):
        - TRIVIAL tasks: Try fast path first (simple LLM + execute), engage full PRECEPT only on failure
        - MEDIUM/COMPLEX tasks: Full PRECEPT pipeline from start

        This method is domain-agnostic and delegates all domain-specific logic
        to the injected DomainStrategy. Works with:
        - Logistics (ports, shipments)
        - Coding (packages, imports)
        - DevOps (stacks, pods)
        - Finance (orders, trades)
        - Booking (reservations, payments)
        - Integration (OAuth, APIs)

        Flow:
        1. Parse task using strategy
        2. Analyze complexity
        3. [FAST PATH] For trivial tasks: Try simple execution first
        4. [FULL PATH] Get and apply learned rules using strategy
        5. Execute action using strategy
        6. Handle errors using strategy
        7. COMPASS evolution (domain-agnostic)

        Args:
            task: The task string to execute
            fast_path: Enable fast path for trivial tasks (default: True)
            metadata: Optional dict with scenario metadata (e.g., condition_key for multi-condition)

        Returns:
            Dict with success, steps, duration, response, strategy, complexity, domain
        """
        self.total_tasks += 1
        start_time = time.time()

        # ═══════════════════════════════════════════════════════════════════
        # STEP 1: Parse Task Using Domain Strategy
        # ═══════════════════════════════════════════════════════════════════
        parsed_task = self.strategy.parse_task(task)

        # ═══════════════════════════════════════════════════════════════════
        # INJECT MULTI-CONDITION METADATA (if provided)
        # This allows condition_key to be passed from scenario without being
        # visible in task description (prevents baselines from "reading" it)
        # ═══════════════════════════════════════════════════════════════════
        if metadata:
            if parsed_task.parameters is None:
                parsed_task.parameters = {}
            # Inject condition_key from metadata (for multi-condition enforcement)
            if "condition_key" in metadata and metadata["condition_key"]:
                parsed_task.parameters["condition_key"] = metadata["condition_key"]
            # Inject conditions list if available
            if "conditions" in metadata:
                parsed_task.parameters["conditions"] = metadata["conditions"]
            # Inject expected_solution for debugging/analysis (not used in execution)
            if "expected_solution" in metadata:
                parsed_task.parameters["expected_solution"] = metadata["expected_solution"]

        # ═══════════════════════════════════════════════════════════════════
        # COMPASS ADVANTAGE 1: Complexity Analysis (ML-based, domain-aware)
        # ═══════════════════════════════════════════════════════════════════
        complexity = self.mcp_client.analyze_complexity(
            task,
            f"{parsed_task.action} {parsed_task.entity}",
        )

        task_steps = 0
        overhead_steps = 0
        attempts = 0  # Track attempts for consistent P₁ definition across methods
        success = False
        final_response = ""
        strategy_used = ""

        # ═══════════════════════════════════════════════════════════════════
        # HYBRID APPROACH: Fast Path for Trivial Tasks
        # Try simple execution first, engage full PRECEPT only on failure
        # This reduces LLM calls for easy tasks while maintaining learning
        # ═══════════════════════════════════════════════════════════════════
        if fast_path and complexity.get("level", "medium") == "trivial":
            fast_result = await self._try_fast_path(task, parsed_task)
            if fast_result and fast_result.get("success"):
                # Fast path succeeded! Record success and return
                task_steps = 1
                duration = time.time() - start_time

                # Still learn from success (but minimal overhead)
                await self.mcp_client.store_experience(
                    f"{self.strategy.domain_name}:{parsed_task.action}",
                    outcome="success",
                    learning=f"Fast path succeeded for {parsed_task.action}",
                )

                return {
                    "success": True,
                    "steps": task_steps,
                    "overhead_steps": 0,
                    "duration": duration,
                    "response": fast_result.get("response", ""),
                    "strategy": "fast_path:simple",
                    "complexity": complexity,
                    "domain": self.strategy.domain_name,
                    "fast_path_used": True,
                }

            # Fast path failed - continue to full PRECEPT pipeline
            task_steps = 1  # Count the fast path attempt
            print("    ⚡ Fast path failed, engaging full PRECEPT...")

        try:
            # ─── Memory Retrieval (Context Engineering: Reactive Retrieval) ───
            query = f"{parsed_task.action} {parsed_task.target or parsed_task.entity}"
            memories = await self.mcp_client.retrieve_memories(query, top_k=10)
            task_steps += 1

            # ─── Procedural Memory (Context Engineering: How-to Strategies) ───
            task_type = f"{self.strategy.domain_name}:{parsed_task.action}"
            procedure = await self.mcp_client.get_procedure(task_type)
            if procedure and "No procedure found" not in procedure:
                # We have a known procedure for this task type
                parsed_task.parameters["procedure_hint"] = procedure
                overhead_steps += 1

            # ═══════════════════════════════════════════════════════════════════
            # STEP 2: Get and Apply Learned Rules
            # TWO-TIER APPROACH:
            #   Tier 1: Fast programmatic lookup (exact pattern match)
            #   Tier 2: LLM reasoning with evolved prompt (generalization)
            # ═══════════════════════════════════════════════════════════════════
            rules_response = await self.mcp_client.get_learned_rules()
            rules = rules_response.split("\n") if rules_response else []

            # TIER 1: Try fast programmatic rule application first
            parsed_task, was_rule_applied, strategy_used = (
                self.strategy.apply_learned_rules(
                    parsed_task,
                    rules,
                )
            )

            # ═══════════════════════════════════════════════════════════════════
            # TIER 2: LLM Reasoning with Evolved Prompt (PRECEPT Key Advantage)
            # If no programmatic rule matched, use LLM to reason with learned knowledge
            # ═══════════════════════════════════════════════════════════════════
            if not was_rule_applied and self._prompt_generation > 0:
                # LLM has evolved prompt with consolidated wisdom - let it reason
                llm_suggestion = await self._llm_reason_with_evolved_prompt(
                    task=task,
                    parsed_task=parsed_task,
                    memories=memories,
                )
                if llm_suggestion:
                    # Apply LLM's suggested modifications to parsed_task
                    if llm_suggestion.get("suggested_solution"):
                        parsed_task.parameters["preferred_solution"] = llm_suggestion[
                            "suggested_solution"
                        ]
                        strategy_used = f"LLM-Reasoned:{llm_suggestion.get('reasoning', 'applied')[:30]}"
                        was_rule_applied = True
                    overhead_steps += 1  # LLM reasoning is overhead

            # ═══════════════════════════════════════════════════════════════════
            # STEP 3: Execute Action Using Domain Strategy
            # ═══════════════════════════════════════════════════════════════════
            action_result = await self.strategy.execute_action(
                self.mcp_client,
                parsed_task,
            )
            task_steps += 1
            attempts += 1  # First attempt
            final_response = action_result.response

            if action_result.success:
                success = True
                strategy_used = action_result.strategy_used or strategy_used

            # ═══════════════════════════════════════════════════════════════════
            # STEP 4: Handle Errors Using Domain Strategy
            # ═══════════════════════════════════════════════════════════════════
            elif action_result.error_code:
                overhead_steps += 1  # Error recording is overhead

                recovery_result = await self.strategy.handle_error(
                    self.mcp_client,
                    action_result.error_code,
                    parsed_task,
                    {"original_response": action_result.response},
                )
                task_steps += 1  # Recovery attempt is a task step
                attempts += 1  # Second attempt (first recovery)

                if recovery_result.success:
                    success = True
                    final_response = recovery_result.response
                    strategy_used = recovery_result.strategy_used
                else:
                    # Check for chained errors
                    if (
                        recovery_result.error_code
                        and recovery_result.error_code != action_result.error_code
                    ):
                        overhead_steps += 1

                        second_recovery = await self.strategy.handle_error(
                            self.mcp_client,
                            recovery_result.error_code,
                            parsed_task,
                            {"original_response": recovery_result.response},
                        )
                        task_steps += 1
                        attempts += 1  # Third attempt (second recovery)

                        if second_recovery.success:
                            success = True
                            final_response = second_recovery.response
                            strategy_used = second_recovery.strategy_used

            # ─── Store Experience (domain-agnostic) ───
            # Only store experiences with learning value (failures/recoveries)
            if (
                not success
                or "Fallback" in strategy_used
                or "Recovery" in strategy_used
            ):
                await self.mcp_client.store_experience(
                    task=task,
                    outcome="success" if success else "failure",
                    strategy=strategy_used,
                    lessons=f"{'Worked' if success else 'Failed'} with {strategy_used}",
                    domain=self.strategy.domain_name,
                )
                overhead_steps += 1

                # ─── Procedural Memory (Context Engineering: Store How-to) ───
                # Store successful recovery as a procedure for future tasks
                if success and "Recovery" in strategy_used:
                    task_type = f"{self.strategy.domain_name}:{parsed_task.action}"
                    await self.mcp_client.store_procedure(
                        name=f"{parsed_task.action}_{parsed_task.entity or 'general'}",
                        task_type=task_type,
                        steps=f"1. Try default approach\n2. On error, use: {strategy_used}",
                    )
                    overhead_steps += 1

            # ─── Record for GEPA Scoring (domain-agnostic) ───
            self.task_results.append(
                {
                    "task": task,
                    "success": success,
                    "steps": task_steps,
                    "strategy": strategy_used,
                    "complexity": complexity["total_steps"],
                    "domain": self.strategy.domain_name,
                }
            )

            # ─── Learn from execution (COMPASS advantage) ───
            self.mcp_client.learn_from_execution(task, task_steps, success)

            # ═══════════════════════════════════════════════════════════════════
            # COMPASS EVOLUTION (domain-agnostic)
            # ═══════════════════════════════════════════════════════════════════
            self.tasks_since_consolidation += 1
            self.tasks_since_compass += 1

            if success:
                self.consecutive_failures = 0
            else:
                self.consecutive_failures += 1

            # ─── Consolidation Trigger ───
            if self.tasks_since_consolidation >= self.CONSOLIDATION_INTERVAL:
                print(
                    f"    🔄 CONSOLIDATION triggered (every {self.CONSOLIDATION_INTERVAL} tasks)"
                )
                await self.mcp_client.trigger_consolidation()
                overhead_steps += 1
                self.tasks_since_consolidation = 0
                self.learning_events.append(
                    f"CONSOLIDATION at task #{self.total_tasks}"
                )

            # ─── COMPASS Evolution Trigger ───
            trigger_compass = False
            trigger_reason = ""

            if self.tasks_since_compass >= self.COMPASS_EVOLUTION_INTERVAL:
                trigger_compass = True
                trigger_reason = f"interval ({self.COMPASS_EVOLUTION_INTERVAL} tasks)"

            if self.consecutive_failures >= self.FAILURE_THRESHOLD:
                trigger_compass = True
                trigger_reason = f"repeated failures ({self.consecutive_failures})"

            if trigger_compass:
                print(f"    🧬 COMPASS EVOLUTION triggered ({trigger_reason})")

                # COMPASS ADVANTAGE 2: Smart Rollout Decision
                scores = compute_scores_from_task_results(self.task_results[-5:])
                current_score = scores.get("task_success_rate", 0.5)

                rollout_decision = self.mcp_client.decide_rollouts(task, current_score)
                print(
                    f"    ✓ Smart rollout: {rollout_decision.num_rollouts} rollouts ({rollout_decision.focus})"
                )

                # Only do full evolution if not early stopping (score < 0.98)
                if rollout_decision.focus != "skip":
                    context = (
                        f"Task #{self.total_tasks}: {task[:50]} | {final_response[:80]}"
                    )
                    await self.mcp_client.trigger_compass_evolution(
                        failure_context=context
                    )
                    overhead_steps += 1

                    # ═══════════════════════════════════════════════════════════════
                    # KEY PRECEPT ADVANTAGE: Refresh prompt with evolved version
                    # This is what makes test-time learning actually work!
                    # ═══════════════════════════════════════════════════════════════
                    prompt_updated = await self.refresh_evolved_prompt()
                    if prompt_updated:
                        overhead_steps += 1
                else:
                    print(f"    ✓ Early stop: Score {current_score:.2f} ≥ 0.98")
                    # Still refresh prompt to include latest learned rules
                    await self.refresh_evolved_prompt()

                self.tasks_since_compass = 0
                self.learning_events.append(f"COMPASS at task #{self.total_tasks}")

        except Exception as e:
            print(f"    Error: {e}")
            final_response = str(e)

        duration = time.time() - start_time

        if success:
            self.successful_tasks += 1

        self.steps_per_task.append(task_steps)

        return {
            "success": success,
            "steps": task_steps,
            "attempts": attempts,  # For consistent P₁ definition (attempts == 1 = first-try)
            "overhead": overhead_steps,
            "duration": duration,
            "response": final_response[:200] if final_response else "",
            "strategy": strategy_used,
            "complexity": complexity,
            "domain": self.strategy.domain_name,
        }

    def get_success_rate(self) -> float:
        """Get the current success rate."""
        if self.total_tasks == 0:
            return 0.0
        return self.successful_tasks / self.total_tasks

    def get_compass_stats(self) -> Dict[str, Any]:
        """Get COMPASS advantage statistics."""
        if self.mcp_client:
            return self.mcp_client.compass_stats
        return {}

    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive statistics."""
        stats = {
            "total_tasks": self.total_tasks,
            "successful_tasks": self.successful_tasks,
            "success_rate": self.get_success_rate(),
            "avg_steps": sum(self.steps_per_task) / len(self.steps_per_task)
            if self.steps_per_task
            else 0,
            "learning_events": len(self.learning_events),
            "compass_stats": self.get_compass_stats(),
            "prompt_stats": self.get_prompt_stats(),
            "domain": self.strategy.domain_name,
        }

        # Add execution stats for coding domain with Docker enabled
        if hasattr(self.strategy, "get_execution_stats"):
            stats["execution_stats"] = self.strategy.get_execution_stats()

        # Add conversation stats
        if hasattr(self, "_conversation_history"):
            stats["conversation"] = {
                "session_active": self._session_active,
                "turns": len(self._conversation_history),
                "session_id": self._session_id,
            }

        return stats

    # ═══════════════════════════════════════════════════════════════════════════════
    # MULTI-TURN CONVERSATION SUPPORT
    # Leverages AutoGen's built-in stateful AssistantAgent
    # ═══════════════════════════════════════════════════════════════════════════════

    def start_session(self, session_id: Optional[str] = None) -> str:
        """
        Start a new conversation session.

        The AssistantAgent maintains internal state automatically.
        This method initializes PRECEPT's tracking of the conversation
        for learning purposes.

        Args:
            session_id: Optional session identifier (auto-generated if not provided)

        Returns:
            The session ID
        """
        import uuid

        self._session_id = session_id or str(uuid.uuid4())[:8]
        self._session_active = True
        self._conversation_history: List[Dict[str, Any]] = []
        self._session_start_time = time.time()

        # Reset the agent to start fresh conversation
        # (AutoGen's AssistantAgent accumulates state across run() calls)
        if self.agent and self.model_client:
            autogen_tools = self.strategy.create_autogen_tools(self.mcp_client)
            self.agent = AssistantAgent(
                name=f"PRECEPT_{self.strategy.category.value}_Agent",
                model_client=self.model_client,
                tools=autogen_tools,
                system_message=self._current_system_prompt,
            )

        return self._session_id

    def end_session(self, store_experience: bool = True) -> Dict[str, Any]:
        """
        End the current conversation session.

        Optionally stores the conversation as an experience for PRECEPT learning.

        Args:
            store_experience: If True, stores conversation summary for future learning

        Returns:
            Session summary with statistics
        """
        if not hasattr(self, "_session_active") or not self._session_active:
            return {"error": "No active session"}

        session_duration = time.time() - self._session_start_time

        summary = {
            "session_id": self._session_id,
            "turns": len(self._conversation_history),
            "duration": session_duration,
            "success_rate": self._calculate_session_success_rate(),
        }

        # Store conversation as experience for PRECEPT learning
        if store_experience and self._conversation_history and self.mcp_client:
            try:
                # Create a summary of the conversation for memory
                conversation_summary = self._summarize_conversation()

                # Store in episodic memory
                import asyncio

                asyncio.create_task(
                    self.mcp_client.store_experience(
                        task=f"conversation_session:{self._session_id}",
                        result=conversation_summary,
                        success=summary["success_rate"] > 0.5,
                        confidence=summary["success_rate"],
                    )
                )
                summary["experience_stored"] = True
            except Exception as e:
                summary["experience_stored"] = False
                summary["store_error"] = str(e)

        self._session_active = False

        return summary

    def _calculate_session_success_rate(self) -> float:
        """Calculate success rate for current session."""
        if not self._conversation_history:
            return 0.0
        successful = sum(
            1 for turn in self._conversation_history if turn.get("success", False)
        )
        return successful / len(self._conversation_history)

    def _summarize_conversation(self) -> str:
        """Create a summary of the conversation for memory storage."""
        if not self._conversation_history:
            return "Empty conversation"

        lines = [
            f"Session {self._session_id} ({len(self._conversation_history)} turns):"
        ]
        for i, turn in enumerate(self._conversation_history[:5]):  # First 5 turns
            user_msg = turn.get("user", "")[:100]
            assistant_msg = turn.get("assistant", "")[:100]
            lines.append(f"  [{i + 1}] User: {user_msg}...")
            lines.append(f"      Assistant: {assistant_msg}...")

        if len(self._conversation_history) > 5:
            lines.append(f"  ... and {len(self._conversation_history) - 5} more turns")

        return "\n".join(lines)

    async def chat(
        self,
        message: str,
        apply_learning: bool = True,
    ) -> str:
        """
        Send a message and get a response in a multi-turn conversation.

        This method leverages AutoGen's built-in stateful conversation support.
        The AssistantAgent automatically maintains conversation history across
        multiple calls to run().

        PRECEPT Integration:
        - Retrieves relevant memories before responding
        - Applies learned rules when applicable
        - Stores successful patterns for future learning
        - Triggers COMPASS evolution periodically

        Args:
            message: The user's message
            apply_learning: If True, applies PRECEPT learning (memory retrieval, rules)

        Returns:
            The assistant's response
        """
        if not self.agent:
            raise RuntimeError("Agent not connected. Call connect() first.")

        # Ensure session is active
        if not hasattr(self, "_session_active") or not self._session_active:
            self.start_session()

        start_time = time.time()

        # ═══════════════════════════════════════════════════════════════════════════
        # PRECEPT INTEGRATION: Enhance context with learning
        # ═══════════════════════════════════════════════════════════════════════════
        enhanced_message = message
        memories_used = ""

        if apply_learning and self.mcp_client:
            try:
                # Retrieve relevant memories
                memories = await self.mcp_client.retrieve_memories(message, top_k=3)
                if memories and "No memories found" not in memories:
                    memories_used = memories

                    # Include memory context in the message for the first turn
                    # (subsequent turns rely on AutoGen's built-in context)
                    if len(self._conversation_history) == 0:
                        enhanced_message = f"""CONTEXT FROM MEMORY:
{memories[:500]}

USER MESSAGE: {message}"""
            except Exception:
                pass  # Silently continue without memory enhancement

        # ═══════════════════════════════════════════════════════════════════════════
        # AUTOGEN MULTI-TURN: Use agent.run() which maintains conversation state
        # ═══════════════════════════════════════════════════════════════════════════
        try:
            result = await self.agent.run(task=enhanced_message)

            # Extract the assistant's response
            if result.messages:
                response = result.messages[-1].content
                if hasattr(response, "text"):
                    response = response.text
                elif not isinstance(response, str):
                    response = str(response)
            else:
                response = "No response generated."

            success = True

        except Exception as e:
            response = f"Error: {str(e)}"
            success = False

        duration = time.time() - start_time

        # ═══════════════════════════════════════════════════════════════════════════
        # PRECEPT LEARNING: Store turn and trigger learning
        # ═══════════════════════════════════════════════════════════════════════════

        # Record turn in conversation history
        turn = {
            "turn": len(self._conversation_history) + 1,
            "user": message,
            "assistant": response,
            "success": success,
            "duration": duration,
            "memories_used": bool(memories_used),
            "timestamp": time.time(),
        }
        self._conversation_history.append(turn)

        # Periodic PRECEPT learning
        if apply_learning and len(self._conversation_history) % 5 == 0:
            # Every 5 turns, check if COMPASS evolution needed
            self.tasks_since_compass += 1
            if self.tasks_since_compass >= self.COMPASS_EVOLUTION_INTERVAL:
                await self.refresh_evolved_prompt()
                self.tasks_since_compass = 0

        return response

    async def chat_stream(
        self,
        message: str,
        apply_learning: bool = True,
    ):
        """
        Send a message and stream the response in a multi-turn conversation.

        Uses AutoGen's run_stream() for real-time response generation.

        Args:
            message: The user's message
            apply_learning: If True, applies PRECEPT learning

        Yields:
            Response chunks as they are generated
        """
        if not self.agent:
            raise RuntimeError("Agent not connected. Call connect() first.")

        # Ensure session is active
        if not hasattr(self, "_session_active") or not self._session_active:
            self.start_session()

        start_time = time.time()

        # Memory enhancement for first turn
        enhanced_message = message
        if apply_learning and self.mcp_client and len(self._conversation_history) == 0:
            try:
                memories = await self.mcp_client.retrieve_memories(message, top_k=3)
                if memories and "No memories found" not in memories:
                    enhanced_message = f"CONTEXT FROM MEMORY:\n{memories[:500]}\n\nUSER MESSAGE: {message}"
            except Exception:
                pass

        # Stream response using AutoGen's run_stream
        full_response = ""
        try:
            async for chunk in self.agent.run_stream(task=enhanced_message):
                # Handle different chunk types
                if hasattr(chunk, "content"):
                    content = chunk.content
                    if content:
                        full_response += str(content)
                        yield str(content)
                elif hasattr(chunk, "messages"):
                    # TaskResult at the end
                    if chunk.messages:
                        final_content = chunk.messages[-1].content
                        if final_content and str(final_content) not in full_response:
                            full_response += str(final_content)
                            yield str(final_content)

            success = True
        except Exception as e:
            full_response = f"Error: {str(e)}"
            yield full_response
            success = False

        # Record turn
        turn = {
            "turn": len(self._conversation_history) + 1,
            "user": message,
            "assistant": full_response,
            "success": success,
            "duration": time.time() - start_time,
            "streamed": True,
            "timestamp": time.time(),
        }
        self._conversation_history.append(turn)

    def get_conversation_history(self) -> List[Dict[str, Any]]:
        """
        Get the current conversation history.

        Returns:
            List of turn dictionaries with user/assistant messages
        """
        if not hasattr(self, "_conversation_history"):
            return []
        return self._conversation_history.copy()

    def reset_conversation(self) -> None:
        """
        Reset the conversation while keeping the session active.

        Clears conversation history and resets the agent's internal state.
        """
        if hasattr(self, "_conversation_history"):
            self._conversation_history = []

        # Reset the agent to clear AutoGen's internal state
        if self.agent and self.model_client:
            autogen_tools = self.strategy.create_autogen_tools(self.mcp_client)
            self.agent = AssistantAgent(
                name=f"PRECEPT_{self.strategy.category.value}_Agent",
                model_client=self.model_client,
                tools=autogen_tools,
                system_message=self._current_system_prompt,
            )
