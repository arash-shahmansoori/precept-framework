"""
PRECEPT Orchestrator: Planning Resilience via Experience, Context Engineering & Probing Trajectories

Combines COMPASS (Genetic-Pareto Optimization) with Evo-Memory (ReMem) into a
continuous improvement cycle:

1. OPTIMIZATION PHASE (Compiler / COMPASS)
   - Analyze agent history
   - Evolve system prompts using genetic-pareto selection
   - Optimize memory usage instructions

2. DEPLOYMENT PHASE (Runtime / Evo-Memory)
   - Execute tasks using ReMem loop
   - Store high-quality experiences
   - Route tasks to optimal prompt versions

3. CONSOLIDATION PHASE (The Merge)
   - Analyze frequent memory patterns
   - "Bake" lessons into system prompts
   - Prune consolidated memories
   - Feed performance data back to COMPASS

This creates a self-optimizing agent that learns both:
- Fast (runtime memory for immediate learning)
- Deep (prompt evolution for permanent improvements)
"""

import asyncio
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from .memory_consolidation import (
    ConsolidationResult,
    MemoryConsolidator,
)
from .memory_store import MemoryStore
from .pareto_memory import (
    ParetoMemoryManager,
    PromptVersion,
    TaskTypeRouter,
)
from .remem_pipeline import ReMem, ReMemState
from .ingestion import (
    SoftIngestionManager,
    FeedbackIngestionManager,
    ExecutionTrace,
    PRECEPTIngestionCoordinator,
)
from .llm_clients import precept_llm_client, precept_embedding_fn


class PRECEPTPhase(Enum):
    """Current phase of the PRECEPT cycle."""
    
    IDLE = "idle"
    OPTIMIZATION = "optimization"  # COMPASS evolution running
    DEPLOYMENT = "deployment"  # ReMem runtime active
    CONSOLIDATION = "consolidation"  # Merging memories into prompts


@dataclass
class PRECEPTConfig:
    """Configuration for PRECEPT framework."""
    
    # Memory configuration
    memory_storage_path: Optional[Path] = None
    max_memories: int = 1000
    
    # ReMem configuration
    max_steps_per_task: int = 10
    retrieve_top_k: int = 5
    
    # Consolidation configuration
    consolidation_interval: int = 50  # Tasks between consolidation runs
    min_strategy_count: int = 5  # Minimum occurrences before consolidation
    min_lesson_count: int = 3
    min_success_rate: float = 0.7
    
    # COMPASS integration
    enable_compass_optimization: bool = True
    compass_evolution_interval: int = 100  # Tasks between COMPASS runs
    
    # Pareto memory
    enable_prompt_routing: bool = True
    
    # Performance tracking
    enable_detailed_logging: bool = True
    track_all_executions: bool = True


@dataclass
class PRECEPTStats:
    """Statistics for PRECEPT operation."""
    
    total_tasks: int = 0
    successful_tasks: int = 0
    failed_tasks: int = 0
    
    total_memories_stored: int = 0
    total_memories_consolidated: int = 0
    
    consolidation_runs: int = 0
    compass_runs: int = 0
    
    prompt_switches: int = 0
    
    start_time: float = field(default_factory=time.time)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        runtime = time.time() - self.start_time
        return {
            "total_tasks": self.total_tasks,
            "successful_tasks": self.successful_tasks,
            "failed_tasks": self.failed_tasks,
            "success_rate": self.successful_tasks / self.total_tasks if self.total_tasks > 0 else 0,
            "total_memories_stored": self.total_memories_stored,
            "total_memories_consolidated": self.total_memories_consolidated,
            "consolidation_runs": self.consolidation_runs,
            "compass_runs": self.compass_runs,
            "prompt_switches": self.prompt_switches,
            "runtime_seconds": runtime,
            "tasks_per_hour": self.total_tasks / (runtime / 3600) if runtime > 0 else 0,
        }


class PRECEPTOrchestrator:
    """
    Main orchestrator for the PRECEPT unified framework.
    
    PRECEPT = Planning Resilience via Experience, Context Engineering & Probing Trajectories
    
    This is the central coordinator that manages:
    - Memory store (episodic memory)
    - ReMem pipeline (runtime learning)
    - Memory consolidation (prompt mutation)
    - Pareto memory (prompt routing)
    - COMPASS integration (prompt evolution)
    """
    
    def __init__(
        self,
        llm_client: Optional[Callable] = None,
        config: Optional[PRECEPTConfig] = None,
        action_executor: Optional[Callable] = None,
        embedding_fn: Optional[Callable] = None,
        compass_evolve_fn: Optional[Callable] = None,  # Function to trigger COMPASS evolution
    ):
        # Use actual OpenAI LLM client by default - NO MOCKS
        self.llm_client = llm_client or precept_llm_client
        self.config = config or PRECEPTConfig()
        self.action_executor = action_executor
        self.compass_evolve_fn = compass_evolve_fn
        
        # Current phase
        self.phase = PRECEPTPhase.IDLE
        
        # Initialize components
        self._init_memory_store(embedding_fn)
        self._init_ingestion()  # Three-stream ingestion
        self._init_remem()
        self._init_consolidation()
        self._init_pareto_memory()
        
        # Statistics
        self.stats = PRECEPTStats()
        
        # Current system prompts (will be evolved)
        self.system_prompts: Dict[str, str] = {}
        
        # Task counter for triggering consolidation/evolution
        self.tasks_since_consolidation = 0
        self.tasks_since_compass = 0
        
        # Execution history
        self.execution_history: List[Dict[str, Any]] = []
    
    def _init_memory_store(self, embedding_fn: Optional[Callable]) -> None:
        """Initialize the memory store."""
        # Use actual OpenAI embeddings by default - NO MOCKS
        self.memory_store = MemoryStore(
            storage_path=self.config.memory_storage_path,
            embedding_fn=embedding_fn or precept_embedding_fn,
            max_memories=self.config.max_memories,
        )
    
    def _init_ingestion(self) -> None:
        """
        Initialize the three-stream ingestion architecture.
        
        PRECEPT uses three distinct ingestion streams:
        1. Hard Ingestion: Document → Vector DB (external, pre-deployment)
        2. Soft Ingestion: Experience patches (runtime, real-time)
        3. Feedback Ingestion: Execution traces (COMPASS, batch)
        """
        # Soft Ingestion Manager (Runtime patches)
        # Allows agent to "patch" the static knowledge base instantly
        self.soft_ingestion = SoftIngestionManager(
            max_patches=500,
            auto_expire_days=30.0,
        )
        
        # Feedback Ingestion Manager (Training traces)
        # Captures execution traces for COMPASS analysis
        self.feedback_ingestion = FeedbackIngestionManager(
            max_traces=1000,
            retention_days=30.0,
        )
        
        # Unified coordinator
        self.ingestion = PRECEPTIngestionCoordinator(
            soft_manager=self.soft_ingestion,
            feedback_manager=self.feedback_ingestion,
        )
    
    def _init_remem(self) -> None:
        """Initialize the ReMem pipeline."""
        self.remem = ReMem(
            memory_store=self.memory_store,
            llm_client=self.llm_client,
            action_executor=self.action_executor,
            config={
                "max_steps": self.config.max_steps_per_task,
                "retrieve_top_k": self.config.retrieve_top_k,
            },
        )
    
    def _init_consolidation(self) -> None:
        """Initialize the memory consolidator."""
        self.consolidator = MemoryConsolidator(
            memory_store=self.memory_store,
            llm_client=self.llm_client,
        )
        # Configure frequency analyzer
        self.consolidator.frequency_analyzer.min_strategy_count = self.config.min_strategy_count
        self.consolidator.frequency_analyzer.min_lesson_count = self.config.min_lesson_count
        self.consolidator.frequency_analyzer.min_success_rate = self.config.min_success_rate
    
    def _init_pareto_memory(self) -> None:
        """Initialize the Pareto memory manager."""
        task_router = TaskTypeRouter(
            llm_client=self.llm_client if self.config.enable_prompt_routing else None,
            use_llm_classification=self.config.enable_prompt_routing,
        )
        self.pareto_memory = ParetoMemoryManager(
            memory_store=self.memory_store,
            llm_client=self.llm_client,
            task_router=task_router,
        )
    
    def set_system_prompts(self, prompts: Dict[str, str]) -> None:
        """
        Set the current system prompts.
        
        These will be mutated during consolidation phases.
        """
        self.system_prompts = prompts.copy()
    
    def register_pareto_prompt(
        self,
        prompt_text: str,
        optimized_for: List[str],
        scores: Dict[str, float],
        **kwargs,
    ) -> PromptVersion:
        """
        Register a prompt from COMPASS Pareto frontier.
        
        This enables task-specific prompt routing.
        """
        from .pareto_memory import TaskType
        
        task_types = []
        for type_str in optimized_for:
            try:
                task_types.append(TaskType(type_str))
            except ValueError:
                task_types.append(TaskType.GENERAL)
        
        if not task_types:
            task_types = [TaskType.GENERAL]
        
        return self.pareto_memory.register_pareto_prompt(
            prompt_text=prompt_text,
            optimized_for=task_types,
            pareto_scores=scores,
            **kwargs,
        )
    
    async def run_task(
        self,
        task: str,
        goal: str,
        domain: Optional[str] = None,
        context: Optional[str] = None,
    ) -> ReMemState:
        """
        Run a task through the PRECEPT pipeline.
        
        This is the main entry point for task execution.
        
        1. Select optimal prompt (if Pareto prompts available)
        2. Execute task using ReMem
        3. Update statistics and performance tracking
        4. Trigger consolidation/evolution if needed
        """
        self.phase = PRECEPTPhase.DEPLOYMENT
        
        # Step 1: Select prompt if routing enabled
        selected_prompt = None
        if self.config.enable_prompt_routing and self.pareto_memory.prompt_versions:
            try:
                selected_prompt = await self.pareto_memory.select_prompt_for_task(
                    task=task,
                    goal=goal,
                    domain=domain,
                )
                if self.pareto_memory.active_prompt_id != getattr(self, '_last_prompt_id', None):
                    self.stats.prompt_switches += 1
                    self._last_prompt_id = self.pareto_memory.active_prompt_id
            except Exception as e:
                print(f"Prompt selection failed: {e}")
        
        # Step 2: Execute task using ReMem
        state = await self.remem.run(
            task=task,
            goal=goal,
            domain=domain,
            context=context,
        )
        
        # Step 3: Update statistics
        self.stats.total_tasks += 1
        if state.success:
            self.stats.successful_tasks += 1
        else:
            self.stats.failed_tasks += 1
        
        # Record in memory store stats
        self.stats.total_memories_stored = self.memory_store.stats["total_added"]
        
        # Track prompt performance
        if selected_prompt:
            classification = await self.pareto_memory.task_router.classify_task(task, goal)
            self.pareto_memory.record_task_outcome(
                prompt_id=selected_prompt.id,
                task_type=classification.primary_type,
                score=state.confidence if state.success else 0.0,
            )
        
        # Store execution history
        if self.config.track_all_executions:
            self.execution_history.append({
                "task": task,
                "goal": goal,
                "domain": domain,
                "success": state.success,
                "confidence": state.confidence,
                "steps": state.step_count,
                "prompt_id": selected_prompt.id if selected_prompt else None,
                "timestamp": time.time(),
            })
        
        # FEEDBACK INGESTION: Record execution trace for COMPASS analysis
        self._record_execution_trace(state, task, goal, domain, selected_prompt)
        
        # Step 4: Check if consolidation/evolution needed
        self.tasks_since_consolidation += 1
        self.tasks_since_compass += 1
        
        if self.tasks_since_consolidation >= self.config.consolidation_interval:
            # Trigger consolidation in background
            asyncio.create_task(self._run_consolidation())
            self.tasks_since_consolidation = 0
        
        if (
            self.config.enable_compass_optimization
            and self.compass_evolve_fn
            and self.tasks_since_compass >= self.config.compass_evolution_interval
        ):
            # Trigger COMPASS evolution in background
            asyncio.create_task(self._run_compass_evolution())
            self.tasks_since_compass = 0
        
        self.phase = PRECEPTPhase.IDLE
        
        return state
    
    async def run_batch(
        self,
        tasks: List[Dict[str, str]],
    ) -> List[ReMemState]:
        """
        Run multiple tasks in sequence.
        
        Each task dict should have 'task', 'goal', and optionally 'domain', 'context'.
        """
        results = []
        for task_dict in tasks:
            state = await self.run_task(
                task=task_dict["task"],
                goal=task_dict["goal"],
                domain=task_dict.get("domain"),
                context=task_dict.get("context"),
            )
            results.append(state)
        return results
    
    async def _run_consolidation(self, force: bool = False) -> ConsolidationResult:
        """Run the consolidation phase."""
        self.phase = PRECEPTPhase.CONSOLIDATION
        
        result = await self.consolidator.consolidate(
            current_prompts=self.system_prompts,
            force_consolidation=force,
        )
        
        # Update system prompts with consolidated rules
        if result.prompt_additions:
            consolidated_section = self.consolidator.get_all_rules_as_prompt_section()
            if "system" in self.system_prompts:
                # Append consolidated rules to system prompt
                self.system_prompts["system"] = (
                    self.system_prompts["system"] + "\n\n" + consolidated_section
                )
        
        # Update statistics
        self.stats.consolidation_runs += 1
        self.stats.total_memories_consolidated += result.stats.get("memories_pruned", 0)
        
        self.phase = PRECEPTPhase.IDLE
        
        return result
    
    async def _run_compass_evolution(self) -> Optional[Dict[str, Any]]:
        """
        Trigger COMPASS evolution with performance data.
        
        This is the "Compiler" phase.
        """
        if not self.compass_evolve_fn:
            return None
        
        self.phase = PRECEPTPhase.OPTIMIZATION
        
        # Prepare data for COMPASS
        compass_input = {
            "current_prompts": self.system_prompts,
            "pareto_performance": self.pareto_memory.export_for_compass(),
            "consolidated_rules": [
                rule.to_prompt_instruction()
                for rule in self.consolidator.consolidated_rules.values()
            ],
            "memory_stats": self.memory_store.get_stats(),
            "frequent_strategies": self.memory_store.get_frequent_strategies(),
            "frequent_lessons": self.memory_store.get_frequent_lessons(),
        }
        
        # Run COMPASS evolution
        try:
            result = await self.compass_evolve_fn(compass_input)
            
            # Import new prompts from COMPASS
            if result and "new_candidates" in result:
                imported = self.pareto_memory.import_from_compass(result["new_candidates"])
                print(f"Imported {imported} new prompts from COMPASS")
            
            self.stats.compass_runs += 1
            
            self.phase = PRECEPTPhase.IDLE
            return result
        
        except Exception as e:
            print(f"COMPASS evolution failed: {e}")
            self.phase = PRECEPTPhase.IDLE
            return None
    
    def _record_execution_trace(
        self,
        state: ReMemState,
        task: str,
        goal: str,
        domain: Optional[str],
        selected_prompt: Optional[PromptVersion],
    ) -> None:
        """
        Record execution trace for FEEDBACK INGESTION.
        
        This is part of the Training Stream - traces are analyzed
        by COMPASS during the optimization phase.
        """
        import hashlib
        
        trace = ExecutionTrace(
            id=hashlib.md5(f"{task}:{time.time()}".encode()).hexdigest()[:12],
            task=task,
            goal=goal,
            domain=domain or "general",
            steps=state.trajectory,
            total_steps=state.step_count,
            success=state.success,
            final_answer=state.final_answer,
            confidence=state.confidence,
            documents_retrieved=[],  # Could track retrieved doc IDs
            patches_applied=[],  # Could track applied patch IDs
            execution_time_ms=(time.time() - state.start_time) * 1000,
            llm_calls=state.llm_calls,
            tokens_used=0,  # Could track if available
            success_factors=[] if not state.success else ["Task completed successfully"],
            failure_factors=[] if state.success else ["Task did not complete"],
            started_at=state.start_time,
            completed_at=time.time(),
        )
        
        self.feedback_ingestion.ingest_trace(trace)
    
    # =========================================================================
    # SOFT INGESTION Methods (Experience/Wisdom Stream)
    # =========================================================================
    
    def create_knowledge_patch(
        self,
        document_id: str,
        correction: str,
        task: str,
        observation: str,
        domain: str = "general",
    ) -> None:
        """
        Create a soft patch to correct/augment static knowledge.
        
        This is SOFT INGESTION - instantly patches the knowledge base
        without re-indexing the Vector DB.
        
        Example:
            agent.create_knowledge_patch(
                document_id="hamburg_port_manual",
                correction="Hamburg shows 'operational' but has strike delays",
                task="Route pharma shipment",
                observation="Booking failed due to undocumented strike",
                domain="shipping",
            )
        """
        self.soft_ingestion.ingest_correction(
            target_document_id=document_id,
            correction=correction,
            source_task=task,
            source_observation=observation,
            domain=domain,
        )
    
    def create_warning_patch(
        self,
        query_pattern: str,
        warning: str,
        task: str,
        domain: str = "general",
    ) -> None:
        """
        Create a warning patch for certain query patterns.
        
        Example:
            agent.create_warning_patch(
                query_pattern="Speed priority shipment",
                warning="Avoid Hamburg port for all Speed priority shipments",
                task="Route pharma shipment",
                domain="shipping",
            )
        """
        from .ingestion import IngestionPriority
        self.soft_ingestion.ingest_warning(
            query_pattern=query_pattern,
            warning=warning,
            source_task=task,
            priority=IngestionPriority.HIGH,
            domain=domain,
        )
    
    def get_patches_for_query(
        self,
        query: str,
        document_ids: Optional[List[str]] = None,
        domain: Optional[str] = None,
    ) -> List:
        """
        Get soft patches that apply to a query.
        
        Called during retrieval to augment Vector DB results.
        """
        return self.soft_ingestion.get_patches_for_retrieval(
            query=query,
            document_ids=document_ids,
            domain=domain,
        )
    
    # =========================================================================
    # FEEDBACK INGESTION Methods (Training Stream)
    # =========================================================================
    
    def analyze_execution_traces(self) -> Dict[str, Any]:
        """
        Analyze execution traces for patterns.
        
        Called during COMPASS optimization phase.
        """
        return self.feedback_ingestion.analyze_patterns()
    
    def get_optimization_recommendations(self) -> List[Dict[str, Any]]:
        """
        Get recommendations for prompt optimization.
        
        Returns patterns that should be consolidated into prompts.
        """
        return self.feedback_ingestion.get_consolidation_recommendations()
    
    async def force_consolidation(self) -> ConsolidationResult:
        """Force an immediate consolidation run."""
        return await self._run_consolidation(force=True)
    
    async def force_compass_evolution(self) -> Optional[Dict[str, Any]]:
        """Force an immediate COMPASS evolution run."""
        return await self._run_compass_evolution()
    
    def get_current_prompt(self, prompt_key: str = "system") -> str:
        """
        Get the current system prompt with consolidated rules.
        
        This returns the "evolved" prompt including baked-in lessons.
        """
        base_prompt = self.system_prompts.get(prompt_key, "")
        consolidated = self.consolidator.get_all_rules_as_prompt_section()
        
        if consolidated:
            return f"{base_prompt}\n\n{consolidated}"
        return base_prompt
    
    def get_effective_prompt(self, prompt_key: str = "system") -> str:
        """Alias for get_current_prompt for compatibility."""
        return self.get_current_prompt(prompt_key)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive statistics."""
        return {
            "precept": self.stats.to_dict(),
            "memory": self.memory_store.get_stats(),
            "remem": self.remem.get_stats(),
            "consolidation": self.consolidator.get_stats(),
            "pareto_memory": self.pareto_memory.get_stats(),
            "ingestion": self.ingestion.get_all_stats(),
        }
    
    def save_state(self, path: Optional[Path] = None) -> None:
        """Save the current state to disk."""
        # Save memory store
        self.memory_store.save()
        
        # Could also save:
        # - Consolidated rules
        # - Pareto prompt versions
        # - Execution history
        # - Statistics
    
    def get_improvement_report(self) -> Dict[str, Any]:
        """
        Generate a report on agent improvement over time.
        
        Shows how the agent has evolved through the PRECEPT cycle.
        """
        if not self.execution_history:
            return {"status": "no_history"}
        
        # Calculate success rate over time (windowed)
        window_size = 20
        success_rates = []
        
        for i in range(0, len(self.execution_history), window_size):
            window = self.execution_history[i:i + window_size]
            successes = sum(1 for e in window if e["success"])
            success_rates.append(successes / len(window))
        
        # Memory growth
        memory_stats = self.memory_store.get_stats()
        
        # Consolidation impact
        consolidation_stats = self.consolidator.get_stats()
        
        return {
            "total_tasks": len(self.execution_history),
            "success_rate_trend": success_rates,
            "current_success_rate": success_rates[-1] if success_rates else 0,
            "improvement": success_rates[-1] - success_rates[0] if len(success_rates) > 1 else 0,
            "memories_stored": memory_stats.get("current_size", 0),
            "rules_consolidated": consolidation_stats.get("total_rules", 0),
            "compass_evolutions": self.stats.compass_runs,
            "prompt_switches": self.stats.prompt_switches,
        }


# Convenience function for creating a PRECEPT agent
async def create_precept_agent(
    initial_prompts: Dict[str, str],
    llm_client: Optional[Callable] = None,
    config: Optional[PRECEPTConfig] = None,
    action_executor: Optional[Callable] = None,
    embedding_fn: Optional[Callable] = None,
    compass_evolve_fn: Optional[Callable] = None,
) -> PRECEPTOrchestrator:
    """
    Create and initialize a PRECEPT agent.
    
    PRECEPT = Planning Resilience via Experience, Context Engineering & Probing Trajectories
    
    This is the recommended way to create a PRECEPT agent.
    Uses ACTUAL OpenAI API by default - NO MOCKS.
    
    Args:
        initial_prompts: Initial system prompts
        llm_client: Function for LLM calls (default: actual OpenAI)
        config: PRECEPT configuration
        action_executor: Function for executing actions
        embedding_fn: Function for generating embeddings (default: actual OpenAI)
        compass_evolve_fn: Function for triggering COMPASS evolution
    
    Returns:
        Initialized PRECEPTOrchestrator
    """
    orchestrator = PRECEPTOrchestrator(
        llm_client=llm_client,  # Will default to precept_llm_client
        config=config,
        action_executor=action_executor,
        embedding_fn=embedding_fn,  # Will default to precept_embedding_fn
        compass_evolve_fn=compass_evolve_fn,
    )
    
    orchestrator.set_system_prompts(initial_prompts)
    
    return orchestrator
