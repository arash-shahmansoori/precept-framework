"""
COMPASS-Enhanced MCP Client.

This module provides an MCP client with COMPASS advantages:
- PRECEPTComplexityAnalyzer: ML-based complexity detection
- SmartRolloutStrategy: Adaptive rollout allocation
- Multi-strategy coordination

Usage:
    from precept import PRECEPTMCPClientWithCOMPASS
    from mcp import StdioServerParameters

    client = PRECEPTMCPClientWithCOMPASS(
        server_params=StdioServerParameters(...)
    )
    await client.connect()

    # Use COMPASS advantages
    complexity = client.analyze_complexity("Book shipment to Boston")
    decision = client.decide_rollouts("task", current_score=0.8)
"""

from typing import Any, Dict, Optional

# MCP imports
try:
    from mcp import StdioServerParameters

    MCP_AVAILABLE = True
except ImportError:
    MCP_AVAILABLE = False
    StdioServerParameters = None

from .complexity_analyzer import (
    PRECEPTComplexityAnalyzer,
    MultiStrategyCoordinator,
    RolloutDecision,
    SmartRolloutStrategy,
)

# Import the base LogisticsMCPClient
try:
    from .precept_mcp_client import LogisticsMCPClient

    PRECEPT_MCP_CLIENT_AVAILABLE = True
except ImportError:
    PRECEPT_MCP_CLIENT_AVAILABLE = False
    LogisticsMCPClient = None


class PRECEPTMCPClientWithCOMPASS:
    """
    Extended MCP Client with COMPASS advantages.

    Integrates:
    - PRECEPTComplexityAnalyzer: ML-based complexity detection
    - SmartRolloutStrategy: Adaptive rollout allocation
    - COMPASSCompilationEngine: Full COMPASS evolution

    COMPASS Advantages:
    - ML-based hop detection (3 hops vs fixed 2-hop)
    - Smart rollout allocation (2 vs 15 rollouts)
    - Multi-strategy retrieval
    - 6.7x faster, 100% accuracy vs 50%

    Usage:
        client = PRECEPTMCPClientWithCOMPASS(server_params=...)
        await client.connect()

        # Analyze complexity
        complexity = client.analyze_complexity("Book shipment from Rotterdam to Boston")

        # Decide rollouts
        decision = client.decide_rollouts("task", current_score=0.8)
    """

    def __init__(self, server_params: Optional[Any] = None):
        """
        Initialize the COMPASS-enhanced MCP client.

        Args:
            server_params: MCP StdioServerParameters for connecting to the server
        """
        if not PRECEPT_MCP_CLIENT_AVAILABLE:
            raise ImportError(
                "LogisticsMCPClient not available. "
                "Make sure precept_mcp_client.py is properly installed."
            )

        self._base_client = LogisticsMCPClient(server_params)

        # ─── COMPASS Advantages Integration ───
        # From complexity_analyzer.py
        self.complexity_analyzer = PRECEPTComplexityAnalyzer(
            use_ml=True,  # ML-based complexity detection (COMPASS advantage)
            cache_enabled=True,  # Caching (COMPASS advantage)
            learning_enabled=True,  # Continuous learning (COMPASS advantage)
        )

        # Smart rollout strategy (COMPASS advantage: 2 vs 15 rollouts)
        self.rollout_strategy = SmartRolloutStrategy(
            diversity_threshold=0.7,
            confidence_threshold=0.9,
            min_rollouts=1,
            max_rollouts=15,
            diversity_rollouts=5,
            consistency_rollouts=3,
        )

        # Multi-strategy coordinator
        self.strategy_coordinator = MultiStrategyCoordinator()

        # COMPASS statistics
        self.compass_stats = {
            "complexity_analyses": 0,
            "rollouts_saved": 0,
            "early_stops": 0,
            "cache_hits": 0,
            "total_rollouts": 0,
            "ml_initialized": self.complexity_analyzer.ml_initialized,
            "ml_status": self.complexity_analyzer.ml_status,
        }

    # ─── Delegate base client methods ───

    async def connect(self):
        """Connect to the MCP server."""
        await self._base_client.connect()

    async def disconnect(self):
        """Disconnect from the MCP server."""
        await self._base_client.disconnect()

    async def book_shipment(self, origin: str, destination: str) -> str:
        """Book a shipment (delegates to base client)."""
        return await self._base_client.book_shipment(origin, destination)

    async def check_port(self, port: str) -> str:
        """Check port status (delegates to base client)."""
        return await self._base_client.check_port(port)

    async def clear_customs(
        self, destination: str, documentation: str = "standard"
    ) -> str:
        """Clear customs for a shipment (delegates to base client)."""
        return await self._base_client.clear_customs(destination, documentation)

    async def get_learned_rules(self) -> str:
        """Get learned rules (delegates to base client)."""
        return await self._base_client.get_learned_rules()

    async def get_rule_hybrid(
        self,
        condition_key: str,
        task_description: str = "",
        similarity_threshold: float = 0.5,
        top_k: int = 3,
    ) -> str:
        """
        HYBRID RULE RETRIEVAL: 3-Tier strategy (O(1) + Vector + Jaccard).
        Delegates to base client's hybrid retrieval method.
        """
        return await self._base_client.get_rule_hybrid(
            condition_key=condition_key,
            task_description=task_description,
            similarity_threshold=similarity_threshold,
            top_k=top_k,
        )

    async def record_error(self, error_code: str, context: str, solution: str = "") -> str:
        """Record an error (delegates to base client)."""
        return await self._base_client.record_error(error_code, context, solution)

    async def clear_learned_data(
        self,
        clear_rules: bool = True,
        clear_experiences: bool = False,
        clear_domain_mappings: bool = True,
    ) -> str:
        """
        Clear learned data for fair experiment resets.

        Delegates to base client's clear_learned_data method.
        """
        return await self._base_client.clear_learned_data(
            clear_rules=clear_rules,
            clear_experiences=clear_experiences,
            clear_domain_mappings=clear_domain_mappings,
        )

    async def reload_learned_rules(self) -> str:
        """
        Reload learned rules from disk to ensure in-memory state is fresh.

        CRITICAL: Call this AFTER training and BEFORE testing to ensure
        the hybrid lookup uses the LATEST rules from training.
        """
        return await self._base_client.reload_learned_rules()

    async def record_solution(
        self, error_code: str, solution: str, context: str = "", task_succeeded: bool = False
    ) -> str:
        """Record a successful solution (delegates to base client)."""
        return await self._base_client.record_solution(error_code, solution, context, task_succeeded)

    async def retrieve_memories(self, query: str, top_k: int = 5) -> str:
        """Retrieve memories (delegates to base client)."""
        return await self._base_client.retrieve_memories(query, top_k)

    async def store_experience(
        self,
        task: str,
        outcome: str,
        strategy: str = "",
        lessons: str = "",
        domain: str = "logistics",
        error_code: str = "",
        solution: str = "",
        failed_options: str = "",
        task_type: str = "",
        condition_key: str = "",
    ) -> str:
        """Store experience (delegates to base client) with enriched metadata."""
        return await self._base_client.store_experience(
            task=task,
            outcome=outcome,
            strategy=strategy,
            lessons=lessons,
            domain=domain,
            error_code=error_code,
            solution=solution,
            failed_options=failed_options,
            task_type=task_type,
            condition_key=condition_key,
        )

    async def trigger_consolidation(self) -> str:
        """Trigger memory consolidation (delegates to base client)."""
        return await self._base_client.trigger_consolidation()

    async def trigger_compass_evolution(self, failure_context: str = "") -> str:
        """Trigger COMPASS evolution (delegates to base client)."""
        return await self._base_client.trigger_compass_evolution(failure_context)

    async def get_procedure(self, task_type: str) -> str:
        """Get procedural memory for a task type (delegates to base client)."""
        return await self._base_client.get_procedure(task_type)

    async def store_procedure(self, name: str, task_type: str, steps: str) -> str:
        """Store a procedure in procedural memory (delegates to base client)."""
        return await self._base_client.store_procedure(name, task_type, steps)

    async def get_evolved_prompt(self, include_rules: bool = True) -> str:
        """
        Get the BEST EVOLVED PROMPT from COMPASS optimization.

        THIS IS THE KEY PRECEPT ADVANTAGE:
        - Returns the prompt evolved through COMPASS
        - Includes consolidated learned rules and domain mappings
        - Should be used to update the agent's system prompt

        COMPASS advantages:
        - ML-based complexity analysis
        - Smart rollout allocation
        - Dynamic prompt evolution with learned rules
        """
        return await self._base_client.get_evolved_prompt(include_rules)

    async def get_prompt_evolution_status(self) -> str:
        """Get detailed status of COMPASS prompt evolution."""
        return await self._base_client.get_prompt_evolution_status()

    async def get_server_stats(self) -> str:
        """Get server statistics (delegates to base client)."""
        return await self._base_client.get_server_stats()

    async def update_memory_usefulness(
        self,
        feedback: float,
        task_succeeded: bool = False,
        memory_ids: str = "",
    ) -> str:
        """
        Update usefulness of retrieved memories based on task outcome.

        FEEDBACK LOOP: Call this after task completion to improve future retrievals.
        Memories that help solve tasks get higher usefulness scores.

        Args:
            feedback: -1.0 to 1.0 (negative = hindered, positive = helped)
            task_succeeded: Whether the overall task completed successfully
            memory_ids: Comma-separated memory IDs (optional, uses last retrieved if empty)

        Returns:
            Confirmation of updated memories
        """
        return await self._base_client.update_memory_usefulness(
            feedback=feedback,
            task_succeeded=task_succeeded,
            memory_ids=memory_ids,
        )

    async def get_last_retrieved_ids(self) -> str:
        """
        Get the IDs of memories from the last retrieval.

        Use this to track which memories were retrieved for feedback purposes.

        Returns:
            Comma-separated list of memory IDs from last retrieval
        """
        return await self._base_client.get_last_retrieved_ids()

    async def call_tool(self, tool_name: str, arguments: dict) -> str:
        """
        Call any tool on the MCP server (delegates to base client).

        This is the universal method for calling tools dynamically,
        especially useful for multi-domain support where tools vary.

        Args:
            tool_name: Name of the tool to call (e.g., "install_package")
            arguments: Arguments to pass to the tool

        Returns:
            Tool result as string
        """
        return await self._base_client.call_tool(tool_name, arguments)

    # ─── COMPASS-specific methods ───

    def analyze_complexity(self, task: str, goal: str = "") -> Dict[str, Any]:
        """
        Analyze task complexity using COMPASS ML analyzer.

        Returns complexity estimate with:
        - level: "trivial" | "medium" | "complex" (for fast path decision)
        - retrieval_hops
        - tool_steps
        - reasoning_steps
        - dominant_dimension
        - confidence

        Args:
            task: The task description
            goal: Optional goal description

        Returns:
            Dict with complexity metrics including "level" for hybrid approach
        """
        estimate = self.complexity_analyzer.analyze(
            task=task,
            goal=goal,
            domain="logistics",
        )
        self.compass_stats["complexity_analyses"] += 1

        # ═══════════════════════════════════════════════════════════════════════
        # HYBRID APPROACH: Classify complexity level for fast path decision
        # - TRIVIAL: Single step, high confidence → use fast path
        # - MEDIUM: 2-3 steps → full PRECEPT
        # - COMPLEX: 4+ steps or multiple dimensions → full PRECEPT + rollouts
        # ═══════════════════════════════════════════════════════════════════════
        total_steps = estimate.total_estimated_steps
        confidence = estimate.confidence

        if total_steps == 1 and confidence >= 0.8:
            level = "trivial"
        elif total_steps <= 3 or confidence >= 0.6:
            level = "medium"
        else:
            level = "complex"

        return {
            "level": level,  # NEW: For fast path decision
            "total_steps": total_steps,
            "retrieval_hops": estimate.retrieval_hops,
            "tool_steps": estimate.tool_steps,
            "reasoning_steps": estimate.reasoning_steps,
            "dominant_dimension": estimate.dominant_dimension.value,
            "confidence": confidence,
            "detected_tools": estimate.detected_tools,
            "detected_entities": estimate.detected_entities,
        }

    def decide_rollouts(
        self,
        task: str,
        current_score: float,
        diversity_score: Optional[float] = None,
    ) -> RolloutDecision:
        """
        Decide rollout allocation using COMPASS smart rollout strategy.

        COMPASS advantage: Uses only 2 rollouts when confident
        vs fixed 15 rollouts in basic approaches.

        Args:
            task: The task description
            current_score: Current performance score
            diversity_score: Optional diversity score

        Returns:
            RolloutDecision with num_rollouts and focus
        """
        complexity = self.complexity_analyzer.analyze(task, domain="logistics")
        decision = self.rollout_strategy.decide(
            task_complexity=complexity,
            current_score=current_score,
            diversity_score=diversity_score,
        )

        if decision.focus == "skip":
            self.compass_stats["early_stops"] += 1
        if decision.num_rollouts < 15:
            self.compass_stats["rollouts_saved"] += 15 - decision.num_rollouts

        return decision

    def learn_from_execution(
        self,
        task: str,
        actual_steps: int,
        success: bool,
    ):
        """
        Learn from execution to improve future complexity estimates.

        COMPASS advantage: Continuous learning from successful executions.

        Args:
            task: The task that was executed
            actual_steps: The actual number of steps taken
            success: Whether the task succeeded
        """
        self.complexity_analyzer.learn_from_execution(
            task=task,
            actual_steps=actual_steps,
            success=success,
            domain="logistics",
        )
