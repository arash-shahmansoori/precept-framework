"""
ReMem Pipeline: Think-Act-Refine Loop for PRECEPT Framework.

Implements the ReMem (Reasoning, Acting, and Memory) architecture from Evo-Memory paper:
1. Think: Reason about the task using retrieved memories
2. Act: Execute actions to solve the task
3. Refine: Summarize experience and update memory

Enhanced with Context Engineering patterns from Google whitepaper:
- Reactive Retrieval: Agent decides when to retrieve (Memory-as-a-Tool)
- Session Compaction: Compress long trajectories (Recursive Summarization)
- Background Memory: Async "Refine" step (Non-blocking Learning)
- Procedural Memory: Store strategies, not just facts

This is the "Runtime" layer that operates during task execution.
"""

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, TYPE_CHECKING

from pydantic import BaseModel, Field

from .memory_store import (
    Experience,
    ExperienceType,
    MemoryPriority,
    MemoryStore,
)
from .llm_clients import precept_llm_client

# Optional Context Engineering integration
if TYPE_CHECKING:
    from .context_engineering import ContextEngineeringManager


class ReMemPhase(Enum):
    """Current phase in the ReMem loop."""
    
    RETRIEVE = "retrieve"  # Retrieving relevant memories
    THINK = "think"  # Reasoning about the task
    ACT = "act"  # Executing action
    OBSERVE = "observe"  # Observing result
    REFINE = "refine"  # Refining memory
    PRUNE = "prune"  # Pruning unhelpful memories
    COMPLETE = "complete"  # Task completed


@dataclass
class ReMemState:
    """
    State tracking for ReMem execution.
    
    Maintains the current state of the Think-Act-Refine loop.
    """
    
    task: str
    goal: str
    
    # Execution state
    phase: ReMemPhase = ReMemPhase.RETRIEVE
    step_count: int = 0
    max_steps: int = 10
    
    # Retrieved context
    retrieved_memories: List[Experience] = field(default_factory=list)
    memory_context: str = ""
    
    # Trajectory
    trajectory: List[Dict[str, str]] = field(default_factory=list)
    current_thought: str = ""
    current_action: str = ""
    current_observation: str = ""
    
    # Outcome
    success: bool = False
    final_answer: str = ""
    confidence: float = 0.0
    
    # Metadata
    start_time: float = field(default_factory=time.time)
    llm_calls: int = 0
    
    def add_step(
        self,
        thought: str,
        action: str,
        observation: str,
    ) -> None:
        """Add a step to the trajectory."""
        self.trajectory.append({
            "step": self.step_count,
            "thought": thought,
            "action": action,
            "observation": observation,
            "timestamp": time.time(),
        })
        self.step_count += 1
        self.current_thought = thought
        self.current_action = action
        self.current_observation = observation
    
    def get_trajectory_text(self) -> str:
        """Get formatted trajectory text."""
        lines = []
        for step in self.trajectory:
            lines.append(f"Step {step['step']}:")
            lines.append(f"  Think: {step['thought']}")
            lines.append(f"  Act: {step['action']}")
            lines.append(f"  Observe: {step['observation']}")
        return "\n".join(lines)


class ReMemAction(BaseModel):
    """Output from the Act phase."""
    
    action_type: str = Field(description="Type of action to take")
    action_content: str = Field(description="Content/details of the action")
    rationale: str = Field(description="Why this action was chosen")
    uses_memory: bool = Field(description="Whether this action was informed by memory")
    memory_ids_used: List[str] = Field(default=[], description="IDs of memories that informed this action")


class ReMemThought(BaseModel):
    """Output from the Think phase."""
    
    analysis: str = Field(description="Analysis of the current situation")
    relevant_memories: List[str] = Field(description="Which memories are most relevant")
    proposed_approach: str = Field(description="Proposed approach based on reasoning")
    confidence: float = Field(description="Confidence in the approach (0-1)")
    should_prune_memories: List[str] = Field(default=[], description="Memory IDs that are unhelpful")


class ReMemRefinement(BaseModel):
    """Output from the Refine phase."""
    
    summary: str = Field(description="Concise summary of what was learned")
    strategy_used: str = Field(description="The strategy that was used")
    lessons_learned: List[str] = Field(description="Specific lessons learned")
    skills_demonstrated: List[str] = Field(description="Skills that were demonstrated")
    should_store: bool = Field(description="Whether this experience is worth storing")
    priority: str = Field(description="Priority level: critical, high, medium, low")
    domain: str = Field(description="Domain/category of this experience")


class ThinkActRefineLoop:
    """
    Core Think-Act-Refine execution loop.
    
    Orchestrates the ReMem pipeline during task execution.
    
    Enhanced with Context Engineering patterns:
    - Reactive Retrieval: Only retrieve when needed (latency savings)
    - Session Compaction: Compress long trajectories (context savings)
    - Background Memory: Async memory writes (response latency savings)
    """
    
    def __init__(
        self,
        memory_store: MemoryStore,
        llm_client: Optional[Callable] = None,
        action_executor: Optional[Callable] = None,
        max_steps: int = 10,
        retrieve_top_k: int = 5,
        context_engineering_manager: Optional["ContextEngineeringManager"] = None,
        enable_reactive_retrieval: bool = False,
        enable_session_compaction: bool = False,
        enable_background_memory: bool = False,
    ):
        self.memory_store = memory_store
        # Use actual OpenAI LLM client by default - NO MOCKS
        self.llm_client = llm_client or precept_llm_client
        self.action_executor = action_executor
        self.max_steps = max_steps
        self.retrieve_top_k = retrieve_top_k
        
        # Context Engineering integration
        self.ce_manager = context_engineering_manager
        self.enable_reactive_retrieval = enable_reactive_retrieval
        self.enable_session_compaction = enable_session_compaction
        self.enable_background_memory = enable_background_memory
        
        # Context Engineering statistics
        self.ce_stats = {
            "retrievals_skipped": 0,
            "compactions_performed": 0,
            "background_writes_queued": 0,
            "latency_saved_ms": 0.0,
        }
        
        # Prompt templates
        self.think_system_prompt = self._get_think_system_prompt()
        self.act_system_prompt = self._get_act_system_prompt()
        self.refine_system_prompt = self._get_refine_system_prompt()
    
    async def execute(
        self,
        task: str,
        goal: str,
        domain: Optional[str] = None,
        initial_context: Optional[str] = None,
        user_id: Optional[str] = None,
    ) -> ReMemState:
        """
        Execute the full Think-Act-Refine loop for a task.
        
        Returns the final state with trajectory and outcome.
        
        Enhanced with Context Engineering:
        - Reactive retrieval on step 0
        - Conditional retrieval on subsequent steps
        - Session compaction when trajectory grows
        """
        # Initialize state
        state = ReMemState(
            task=task,
            goal=goal,
            max_steps=self.max_steps,
        )
        
        # Phase 1: Smart Retrieval (Context Engineering: Reactive Retrieval)
        state.phase = ReMemPhase.RETRIEVE
        
        if self.ce_manager and self.enable_reactive_retrieval:
            # Use Context Engineering's reactive retrieval
            ce_context = await self.ce_manager.before_step(
                task=task,
                trajectory=[],
                step_count=0,
                domain=domain,
                user_id=user_id,
            )
            state.retrieved_memories = ce_context.get("retrieved_memories", [])
            if not ce_context.get("should_retrieve", True):
                self.ce_stats["retrievals_skipped"] += 1
        else:
            # Standard retrieval
            state.retrieved_memories = self.memory_store.retrieve_relevant(
                query=f"{task} {goal}",
                top_k=self.retrieve_top_k,
                domain=domain,
            )
        
        state.memory_context = self._format_memory_context(state.retrieved_memories)
        
        # Add procedural context if available
        if self.ce_manager:
            proc_context = self.ce_manager.get_prompt_context(
                user_id=user_id,
                domain=domain,
            )
            if proc_context:
                state.memory_context = f"{proc_context}\n\n{state.memory_context}"
        
        # Main loop: Think -> Act -> Observe
        while state.step_count < state.max_steps and state.phase != ReMemPhase.COMPLETE:
            # Think phase
            state.phase = ReMemPhase.THINK
            thought = await self._think(state)
            
            # Handle memory pruning if suggested
            if thought.should_prune_memories:
                state.phase = ReMemPhase.PRUNE
                for mem_id in thought.should_prune_memories:
                    self.memory_store.update_usefulness(mem_id, feedback=-0.5)
            
            # Act phase
            state.phase = ReMemPhase.ACT
            action = await self._act(state, thought)
            
            # Execute action and observe
            state.phase = ReMemPhase.OBSERVE
            observation = await self._execute_action(state, action)
            
            # Record step
            state.add_step(
                thought=thought.analysis,
                action=action.action_content,
                observation=observation,
            )
            
            # Context Engineering: Record pattern for consolidation
            if self.ce_manager:
                await self.ce_manager.after_step(
                    task=task,
                    step=state.trajectory[-1],
                observation=observation,
            )
            
            # Update memory usefulness for used memories
            for mem_id in action.memory_ids_used:
                self.memory_store.update_usefulness(mem_id, feedback=0.3)
            
            # Context Engineering: Session Compaction
            if (
                self.ce_manager and 
                self.enable_session_compaction and 
                len(state.trajectory) > 5
            ):
                compacted, summary = await self.ce_manager.session_compactor.compact_trajectory(
                    trajectory=state.trajectory,
                    task=task,
                    goal=goal,
                )
                if summary:
                    state.trajectory = compacted
                    self.ce_stats["compactions_performed"] += 1
            
            # Check if task is complete
            if self._is_task_complete(state, observation):
                state.phase = ReMemPhase.COMPLETE
                state.success = True
                state.final_answer = observation
                state.confidence = thought.confidence
        
        # Refine phase: Summarize and store experience
        state.phase = ReMemPhase.REFINE
        
        # Context Engineering: Background Memory Generation
        if self.ce_manager and self.enable_background_memory:
            # Queue memory write in background (non-blocking)
            job_id = self.ce_manager.background_writer.queue_memory_write(
                task=task,
                trajectory=state.trajectory,
                outcome="success" if state.success else "failure",
            )
            if job_id:
                self.ce_stats["background_writes_queued"] += 1
        else:
            # Synchronous memory write (blocking)
            await self._refine(state)
        
        # Context Engineering: After task processing
        if self.ce_manager:
            await self.ce_manager.after_task(
                task=task,
                trajectory=state.trajectory,
                outcome="success" if state.success else "failure",
                user_id=user_id,
                domain=domain,
            )
        
        return state
    
    async def _think(self, state: ReMemState) -> ReMemThought:
        """
        Think phase: Analyze situation and plan approach.
        
        Uses retrieved memories to inform reasoning.
        """
        think_prompt = self._create_think_prompt(state)
        
        thought = await self.llm_client(
            system_prompt=self.think_system_prompt,
            user_prompt=think_prompt,
            response_model=ReMemThought,
        )
        state.llm_calls += 1
        
        return thought
    
    async def _act(
        self, state: ReMemState, thought: ReMemThought
    ) -> ReMemAction:
        """
        Act phase: Decide and execute action.
        
        Chooses action based on reasoning and memory context.
        """
        act_prompt = self._create_act_prompt(state, thought)
        
        action = await self.llm_client(
            system_prompt=self.act_system_prompt,
            user_prompt=act_prompt,
            response_model=ReMemAction,
        )
        state.llm_calls += 1
        
        return action
    
    async def _execute_action(
        self, state: ReMemState, action: ReMemAction
    ) -> str:
        """Execute an action and return observation."""
        if self.action_executor:
            try:
                observation = await self.action_executor(
                    action.action_type,
                    action.action_content,
                    state,
                )
                return observation
            except Exception as e:
                return f"Action failed: {str(e)}"
        else:
            # Default: Return action as observation (for testing/simulation)
            return f"Executed: {action.action_content}"
    
    async def _refine(self, state: ReMemState) -> None:
        """
        Refine phase: Summarize experience and update memory.
        
        This is the key learning step that creates new memories.
        """
        refine_prompt = self._create_refine_prompt(state)
        
        refinement = await self.llm_client(
            system_prompt=self.refine_system_prompt,
            user_prompt=refine_prompt,
            response_model=ReMemRefinement,
        )
        state.llm_calls += 1
        
        # Store experience if deemed valuable
        if refinement.should_store:
            # Map priority string to enum
            priority_map = {
                "critical": MemoryPriority.CRITICAL,
                "high": MemoryPriority.HIGH,
                "medium": MemoryPriority.MEDIUM,
                "low": MemoryPriority.LOW,
            }
            priority = priority_map.get(
                refinement.priority.lower(), MemoryPriority.MEDIUM
            )
            
            # Determine experience type
            exp_type = (
                ExperienceType.SUCCESS if state.success
                else ExperienceType.FAILURE
            )
            
            # Store the experience
            self.memory_store.store_experience(
                task_description=state.task,
                goal=state.goal,
                trajectory=state.trajectory,
                outcome="success" if state.success else "failure",
                correctness=state.confidence if state.success else 0.0,
                strategy_used=refinement.strategy_used,
                lessons_learned=refinement.lessons_learned,
                skills_demonstrated=refinement.skills_demonstrated,
                domain=refinement.domain,
                experience_type=exp_type,
                priority=priority,
                summary=refinement.summary,
            )
    
    def _is_task_complete(self, state: ReMemState, observation: str) -> bool:
        """Check if the task is complete."""
        # Simple heuristics - can be customized
        completion_signals = [
            "task complete",
            "goal achieved",
            "success",
            "done",
            "finished",
        ]
        obs_lower = observation.lower()
        return any(signal in obs_lower for signal in completion_signals)
    
    def _format_memory_context(self, memories: List[Experience]) -> str:
        """Format retrieved memories as context."""
        if not memories:
            return "No relevant past experiences found."
        
        lines = ["RELEVANT EXPERIENCE FROM SIMILAR TASKS:", ""]
        for i, mem in enumerate(memories, 1):
            lines.append(f"[Experience #{i}] (ID: {mem.id})")
            lines.append(f"Goal: {mem.goal}")
            lines.append(f"Strategy: {mem.strategy_used}")
            lines.append(f"Outcome: {mem.outcome}")
            if mem.lessons_learned:
                lines.append(f"Lessons: {'; '.join(mem.lessons_learned)}")
            if mem.summary:
                lines.append(f"Summary: {mem.summary}")
            lines.append("")
        
        return "\n".join(lines)
    
    def _create_think_prompt(self, state: ReMemState) -> str:
        """Create prompt for Think phase."""
        return f"""
CURRENT TASK
============
Task: {state.task}
Goal: {state.goal}

{state.memory_context}

TRAJECTORY SO FAR
=================
{state.get_trajectory_text() if state.trajectory else "No actions taken yet."}

CURRENT STATE
=============
Step: {state.step_count}/{state.max_steps}

INSTRUCTIONS
============
1. Analyze the current situation and progress toward the goal
2. Consider which retrieved experiences are most relevant
3. Identify any memories that are NOT helpful (for pruning)
4. Propose your approach for the next action
5. Rate your confidence in this approach (0-1)

Focus on leveraging relevant past experiences while identifying unhelpful ones.
"""
    
    def _create_act_prompt(
        self, state: ReMemState, thought: ReMemThought
    ) -> str:
        """Create prompt for Act phase."""
        return f"""
CURRENT TASK
============
Task: {state.task}
Goal: {state.goal}

YOUR ANALYSIS
=============
{thought.analysis}

PROPOSED APPROACH
=================
{thought.proposed_approach}

RELEVANT MEMORIES
=================
{thought.relevant_memories}

TRAJECTORY SO FAR
=================
{state.get_trajectory_text() if state.trajectory else "No actions taken yet."}

INSTRUCTIONS
============
Choose the next action to take. Consider:
1. What action best advances toward the goal?
2. Are you using insights from past experiences?
3. What is the rationale for this specific action?

Specify the action clearly and note which memories (if any) informed your choice.
"""
    
    def _create_refine_prompt(self, state: ReMemState) -> str:
        """Create prompt for Refine phase."""
        return f"""
COMPLETED TASK
==============
Task: {state.task}
Goal: {state.goal}
Outcome: {"SUCCESS" if state.success else "FAILURE/INCOMPLETE"}

FULL TRAJECTORY
===============
{state.get_trajectory_text()}

FINAL ANSWER
============
{state.final_answer}

MEMORIES USED
=============
{self._format_memory_context(state.retrieved_memories)}

INSTRUCTIONS
============
Reflect on this experience and create a memory summary:

1. What was the overall strategy used?
2. What specific lessons were learned?
3. What skills were demonstrated?
4. Is this experience worth storing for future reference?
5. What priority level should it have?
6. What domain/category does this belong to?

Be concise but capture the key insights that would help in similar future tasks.
Focus on GENERALIZABLE lessons, not task-specific details.
"""
    
    def _get_think_system_prompt(self) -> str:
        """System prompt for Think phase."""
        return """You are an expert reasoning agent with access to past experiences.

Your role in the THINK phase:
- Analyze the current task and goal
- Consider relevant past experiences from memory
- Identify which memories are helpful vs. unhelpful
- Formulate a strategic approach

Be analytical and strategic. Use past experiences to inform your reasoning.
Flag any memories that seem irrelevant or potentially misleading."""
    
    def _get_act_system_prompt(self) -> str:
        """System prompt for Act phase."""
        return """You are an expert action agent that executes tasks efficiently.

Your role in the ACT phase:
- Choose the most effective action based on analysis
- Leverage insights from relevant past experiences
- Explain your rationale clearly
- Note which memories informed your decision

Be decisive and action-oriented. Build on successful strategies from the past."""
    
    def _get_refine_system_prompt(self) -> str:
        """System prompt for Refine phase."""
        return """You are an expert at learning and summarizing experiences.

Your role in the REFINE phase:
- Summarize the experience concisely
- Extract generalizable lessons and strategies
- Identify demonstrated skills
- Assess whether this experience is worth storing
- Categorize by domain and priority

Focus on lessons that will help in FUTURE tasks, not just this specific task.
Be selective - not every experience needs to be stored."""


class ReMem:
    """
    High-level ReMem interface for the PRECEPT framework.
    
    Provides simplified API for running the Think-Act-Refine loop.
    
    Enhanced with Context Engineering integration for production efficiency:
    - Reactive Retrieval: ~100ms latency savings per skipped retrieval
    - Session Compaction: ~30% context window savings
    - Background Memory: ~200ms response latency savings
    """
    
    def __init__(
        self,
        memory_store: MemoryStore,
        llm_client: Optional[Callable] = None,
        action_executor: Optional[Callable] = None,
        config: Optional[Dict[str, Any]] = None,
        context_engineering_manager: Optional["ContextEngineeringManager"] = None,
    ):
        self.memory_store = memory_store
        # Use actual OpenAI LLM client by default - NO MOCKS
        self.llm_client = llm_client or precept_llm_client
        self.action_executor = action_executor
        
        # Configuration
        config = config or {}
        self.max_steps = config.get("max_steps", 10)
        self.retrieve_top_k = config.get("retrieve_top_k", 5)
        
        # Context Engineering configuration
        self.ce_manager = context_engineering_manager
        enable_ce = config.get("enable_context_engineering", False)
        
        # Create the execution loop
        self.loop = ThinkActRefineLoop(
            memory_store=memory_store,
            llm_client=self.llm_client,
            action_executor=action_executor,
            max_steps=self.max_steps,
            retrieve_top_k=self.retrieve_top_k,
            context_engineering_manager=context_engineering_manager,
            enable_reactive_retrieval=config.get("enable_reactive_retrieval", enable_ce),
            enable_session_compaction=config.get("enable_session_compaction", enable_ce),
            enable_background_memory=config.get("enable_background_memory", enable_ce),
        )
        
        # Execution history
        self.execution_history: List[ReMemState] = []
    
    async def run(
        self,
        task: str,
        goal: str,
        domain: Optional[str] = None,
        context: Optional[str] = None,
    ) -> ReMemState:
        """
        Run the ReMem loop for a task.
        
        This is the main entry point for task execution with memory.
        """
        state = await self.loop.execute(
            task=task,
            goal=goal,
            domain=domain,
            initial_context=context,
        )
        
        self.execution_history.append(state)
        
        return state
    
    def get_stats(self) -> Dict[str, Any]:
        """Get execution statistics including Context Engineering metrics."""
        if not self.execution_history:
            return {"executions": 0}
        
        success_count = sum(1 for s in self.execution_history if s.success)
        total_steps = sum(s.step_count for s in self.execution_history)
        total_llm_calls = sum(s.llm_calls for s in self.execution_history)
        
        stats = {
            "executions": len(self.execution_history),
            "success_rate": success_count / len(self.execution_history),
            "avg_steps": total_steps / len(self.execution_history),
            "avg_llm_calls": total_llm_calls / len(self.execution_history),
            "memory_stats": self.memory_store.get_stats(),
        }
        
        # Add Context Engineering stats if available
        if self.loop.ce_stats:
            stats["context_engineering"] = {
                **self.loop.ce_stats,
                "efficiency_gains": {
                    "retrieval_latency_saved_ms": self.loop.ce_stats.get("latency_saved_ms", 0),
                    "compactions_performed": self.loop.ce_stats.get("compactions_performed", 0),
                    "background_writes": self.loop.ce_stats.get("background_writes_queued", 0),
                },
            }
        
        # Add Context Engineering Manager stats if available
        if self.ce_manager:
            stats["context_engineering_manager"] = self.ce_manager.get_stats()
        
        return stats

