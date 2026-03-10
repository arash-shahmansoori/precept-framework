"""
Context Engineering Module for PRECEPT Framework.

Implements efficiency patterns from Google's Context Engineering whitepaper
to make PRECEPT production-ready at scale.

Key Patterns Implemented:
1. Memory-as-a-Tool (Reactive Retrieval) - Agent decides when to retrieve
2. Session Compaction (Recursive Summarization) - Compress reasoning traces
3. Background Memory Generation - Async "Refine" step for low latency
4. Procedural Memory - Store "how-to" strategies, not just facts
5. Memory Scoping - Application-level vs User-level isolation
6. Smart Consolidation Triggers - Detect duplicates/conflicts
7. Irrelevance-based Pruning - Remove consolidated memories

Reference: Google Context Engineering Whitepaper 2024
"""

import asyncio
import hashlib
import logging
import threading
import time
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from queue import Queue
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

from pydantic import BaseModel, Field

from .llm_clients import precept_llm_client
from .memory_store import Experience, ExperienceType, MemoryPriority, MemoryStore

logger = logging.getLogger(__name__)


# =============================================================================
# SECTION 1: MEMORY SCOPING
# =============================================================================


class MemoryScope(Enum):
    """
    Memory scope levels from the whitepaper.

    Determines who can access and benefit from a memory.
    """

    SYSTEM = "system"  # Global rules, applies to ALL agents/users
    APPLICATION = "application"  # App-wide, all users of this application
    USER = "user"  # User-specific personalization
    SESSION = "session"  # Temporary, current session only


@dataclass
class ScopedMemory:
    """
    Memory with explicit scope information.

    Enables isolation and sharing at appropriate levels.
    """

    id: str
    content: str
    scope: MemoryScope
    scope_id: str  # user_id, app_id, or "global"
    memory_type: str  # "fact", "procedure", "preference", "warning"
    created_at: float = field(default_factory=time.time)
    access_count: int = 0
    last_accessed: float = field(default_factory=time.time)
    ttl: Optional[float] = None  # Time-to-live in seconds

    @property
    def is_expired(self) -> bool:
        if self.ttl is None:
            return False
        return (time.time() - self.created_at) > self.ttl


class MemoryScopeManager:
    """
    Manages memory isolation and sharing across scopes.

    From whitepaper: "Application-Level Scope stores Black Swan protocols
    so all agents benefit immediately. User-Level Scope stores personalization."
    """

    def __init__(self):
        self.memories: Dict[MemoryScope, Dict[str, ScopedMemory]] = {
            scope: {} for scope in MemoryScope
        }
        self.scope_indices: Dict[str, Set[str]] = defaultdict(set)

    def store(
        self,
        content: str,
        scope: MemoryScope,
        scope_id: str = "global",
        memory_type: str = "fact",
        ttl: Optional[float] = None,
    ) -> ScopedMemory:
        """Store a memory with explicit scope."""
        memory_id = hashlib.md5(
            f"{content}:{scope.value}:{scope_id}".encode()
        ).hexdigest()[:12]

        memory = ScopedMemory(
            id=memory_id,
            content=content,
            scope=scope,
            scope_id=scope_id,
            memory_type=memory_type,
            ttl=ttl if ttl else self._default_ttl(scope),
        )

        self.memories[scope][memory_id] = memory
        self.scope_indices[scope_id].add(memory_id)

        return memory

    def retrieve(
        self,
        query: str,
        user_id: Optional[str] = None,
        app_id: Optional[str] = None,
        include_system: bool = True,
    ) -> List[ScopedMemory]:
        """
        Retrieve memories respecting scope hierarchy.

        Order: SESSION -> USER -> APPLICATION -> SYSTEM
        Higher specificity = higher priority
        """
        results = []

        # System scope (if requested)
        if include_system:
            results.extend(self._search_scope(MemoryScope.SYSTEM, query))

        # Application scope
        if app_id:
            results.extend(self._search_scope(MemoryScope.APPLICATION, query, app_id))

        # User scope
        if user_id:
            results.extend(self._search_scope(MemoryScope.USER, query, user_id))

        # Session scope always included
        results.extend(self._search_scope(MemoryScope.SESSION, query))

        # Filter expired
        results = [m for m in results if not m.is_expired]

        return results

    def _search_scope(
        self,
        scope: MemoryScope,
        query: str,
        scope_id: str = "global",
    ) -> List[ScopedMemory]:
        """Simple text search within a scope."""
        query_words = set(query.lower().split())
        matches = []

        for memory in self.memories[scope].values():
            if scope_id != "global" and memory.scope_id != scope_id:
                continue

            memory_words = set(memory.content.lower().split())
            overlap = len(query_words & memory_words)
            if overlap > 0:
                memory.access_count += 1
                memory.last_accessed = time.time()
                matches.append(memory)

        return matches

    def _default_ttl(self, scope: MemoryScope) -> Optional[float]:
        """Default TTL based on scope."""
        ttls = {
            MemoryScope.SESSION: 3600,  # 1 hour
            MemoryScope.USER: None,  # Permanent
            MemoryScope.APPLICATION: None,  # Permanent
            MemoryScope.SYSTEM: None,  # Permanent
        }
        return ttls.get(scope)

    def promote_to_higher_scope(
        self,
        memory_id: str,
        current_scope: MemoryScope,
        target_scope: MemoryScope,
    ) -> bool:
        """
        Promote a memory to a higher scope.

        Used when a pattern proves useful across users/sessions.
        """
        if current_scope not in self.memories:
            return False

        memory = self.memories[current_scope].get(memory_id)
        if not memory:
            return False

        # Create new memory at higher scope
        promoted = ScopedMemory(
            id=f"{memory_id}_promoted",
            content=memory.content,
            scope=target_scope,
            scope_id="global"
            if target_scope in [MemoryScope.SYSTEM, MemoryScope.APPLICATION]
            else memory.scope_id,
            memory_type=memory.memory_type,
        )

        self.memories[target_scope][promoted.id] = promoted
        return True

    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about memory distribution."""
        return {scope.value: len(memories) for scope, memories in self.memories.items()}


# =============================================================================
# SECTION 2: PROCEDURAL MEMORY
# =============================================================================


@dataclass
class Procedure:
    """
    A stored procedure/strategy - "knowing how" vs "knowing what".

    From whitepaper: "Procedural Memory stores workflows and playbooks,
    not just facts. Use extraction prompts to distill reusable strategies."
    """

    id: str
    name: str
    description: str
    steps: List[str]
    preconditions: List[str]
    postconditions: List[str]
    domain: str
    success_rate: float = 0.0
    execution_count: int = 0
    avg_duration_ms: float = 0.0
    created_at: float = field(default_factory=time.time)
    last_used: float = field(default_factory=time.time)

    def to_prompt_format(self) -> str:
        """Convert procedure to prompt-injectable format."""
        steps_text = "\n".join(
            f"  {i + 1}. {step}" for i, step in enumerate(self.steps)
        )

        return f"""
### PROCEDURE: {self.name}
**When to use:** {self.description}
**Preconditions:** {", ".join(self.preconditions) or "None"}
**Steps:**
{steps_text}
**Expected outcome:** {", ".join(self.postconditions) or "Task completion"}
**Success rate:** {self.success_rate * 100:.0f}%
"""


class ProcedureExtraction(BaseModel):
    """LLM output for procedure extraction."""

    procedure_name: str = Field(description="Short name for the procedure")
    description: str = Field(description="When to use this procedure")
    steps: List[str] = Field(description="Ordered list of steps")
    preconditions: List[str] = Field(
        description="Conditions that must be true before starting"
    )
    postconditions: List[str] = Field(description="Expected state after completion")
    domain: str = Field(description="Domain this procedure applies to")


class ProceduralMemoryStore:
    """
    Stores and retrieves procedural knowledge (strategies, playbooks).

    Distinguishes between:
    - Declarative memory: "Rotterdam is closed on Tuesdays" (fact)
    - Procedural memory: "How to route shipments when a port is closed" (strategy)
    """

    def __init__(self, llm_client: Optional[Callable] = None):
        self.procedures: Dict[str, Procedure] = {}
        self.domain_index: Dict[str, List[str]] = defaultdict(list)
        self.llm_client = llm_client or precept_llm_client

    async def extract_procedure(
        self,
        trajectory: List[Dict[str, str]],
        task: str,
        outcome: str,
        domain: str = "general",
    ) -> Optional[Procedure]:
        """
        Extract a reusable procedure from a successful execution.

        This is the "Soft Ingestion" that creates procedural knowledge.
        """
        if outcome != "success":
            return None

        extraction_prompt = f"""
Analyze this successful task execution and extract a reusable PROCEDURE:

TASK: {task}
DOMAIN: {domain}

EXECUTION TRAJECTORY:
{self._format_trajectory(trajectory)}

Extract a generalized procedure that could be reused for similar tasks.
Focus on the strategy/approach, not task-specific details.
"""

        try:
            extraction = await self.llm_client(
                system_prompt="You extract reusable procedures from successful task executions.",
                user_prompt=extraction_prompt,
                response_model=ProcedureExtraction,
            )

            procedure = Procedure(
                id=hashlib.md5(extraction.procedure_name.encode()).hexdigest()[:10],
                name=extraction.procedure_name,
                description=extraction.description,
                steps=extraction.steps,
                preconditions=extraction.preconditions,
                postconditions=extraction.postconditions,
                domain=extraction.domain or domain,
                success_rate=1.0,  # First success
                execution_count=1,
            )

            self._store_procedure(procedure)
            return procedure

        except Exception as e:
            logger.warning(f"Procedure extraction failed: {e}")
            return None

    def _store_procedure(self, procedure: Procedure) -> None:
        """Store or update a procedure."""
        # Check for similar existing procedure
        existing_id = self._find_similar(procedure)

        if existing_id:
            # Update existing procedure
            existing = self.procedures[existing_id]
            existing.execution_count += 1
            existing.success_rate = (
                existing.success_rate * (existing.execution_count - 1) + 1.0
            ) / existing.execution_count
            existing.last_used = time.time()
        else:
            # Add new procedure
            self.procedures[procedure.id] = procedure
            self.domain_index[procedure.domain].append(procedure.id)

    def _find_similar(self, procedure: Procedure) -> Optional[str]:
        """Find a similar existing procedure."""
        for proc_id, existing in self.procedures.items():
            if existing.domain != procedure.domain:
                continue

            # Simple name similarity check
            name_words = set(procedure.name.lower().split())
            existing_words = set(existing.name.lower().split())
            overlap = len(name_words & existing_words)

            if overlap >= len(name_words) * 0.5:
                return proc_id

        return None

    def get_applicable_procedures(
        self,
        task: str,
        domain: Optional[str] = None,
    ) -> List[Procedure]:
        """Get procedures that might apply to a task."""
        task_words = set(task.lower().split())
        matches = []

        candidate_ids = (
            self.domain_index.get(domain, [])
            if domain
            else list(self.procedures.keys())
        )

        for proc_id in candidate_ids:
            proc = self.procedures.get(proc_id)
            if not proc:
                continue

            # Check if preconditions might match
            proc_words = set(proc.description.lower().split())
            if task_words & proc_words:
                matches.append(proc)

        # Sort by success rate
        matches.sort(key=lambda p: p.success_rate, reverse=True)
        return matches[:5]

    def record_execution(
        self,
        procedure_id: str,
        success: bool,
        duration_ms: float,
    ) -> None:
        """Record a procedure execution outcome."""
        if procedure_id not in self.procedures:
            return

        proc = self.procedures[procedure_id]
        proc.execution_count += 1
        proc.success_rate = (
            proc.success_rate * (proc.execution_count - 1) + (1.0 if success else 0.0)
        ) / proc.execution_count
        proc.avg_duration_ms = (
            proc.avg_duration_ms * (proc.execution_count - 1) + duration_ms
        ) / proc.execution_count
        proc.last_used = time.time()

    def _format_trajectory(self, trajectory: List[Dict[str, str]]) -> str:
        """Format trajectory for extraction prompt."""
        lines = []
        for i, step in enumerate(trajectory):
            lines.append(f"Step {i + 1}:")
            lines.append(f"  Think: {step.get('thought', 'N/A')}")
            lines.append(f"  Action: {step.get('action', 'N/A')}")
            lines.append(f"  Result: {step.get('observation', 'N/A')}")
        return "\n".join(lines)

    def get_all_as_prompt_section(self) -> str:
        """Get all procedures formatted for prompt injection."""
        if not self.procedures:
            return ""

        lines = [
            "=== AVAILABLE PROCEDURES (Learned Strategies) ===",
            "",
        ]

        for proc in sorted(
            self.procedures.values(), key=lambda p: p.success_rate, reverse=True
        )[:10]:  # Top 10 by success rate
            lines.append(proc.to_prompt_format())

        return "\n".join(lines)


# =============================================================================
# SECTION 3: SESSION COMPACTION
# =============================================================================


class CompactionSummary(BaseModel):
    """LLM output for trajectory compaction."""

    summary: str = Field(description="Concise summary of the trajectory")
    key_actions: List[str] = Field(description="Most important actions taken")
    key_learnings: List[str] = Field(description="Key things learned")
    current_status: str = Field(description="Current status/progress")


class SessionCompactor:
    """
    Compacts long reasoning traces to prevent context overflow.

    From whitepaper: "Use Recursive Summarization or Token-Based Truncation.
    Compact older Thoughts into concise summary while keeping immediate
    Action and Observation fresh."
    """

    def __init__(
        self,
        max_trajectory_length: int = 10,
        compaction_threshold: int = 5,
        preserve_recent: int = 3,
        llm_client: Optional[Callable] = None,
    ):
        self.max_trajectory_length = max_trajectory_length
        self.compaction_threshold = compaction_threshold
        self.preserve_recent = preserve_recent
        self.llm_client = llm_client or precept_llm_client
        self.compaction_history: List[Dict[str, Any]] = []

    async def compact_trajectory(
        self,
        trajectory: List[Dict[str, str]],
        task: str,
        goal: str,
    ) -> Tuple[List[Dict[str, str]], str]:
        """
        Compact a trajectory if it exceeds threshold.

        Returns: (compacted_trajectory, summary_of_compacted_part)
        """
        if len(trajectory) <= self.compaction_threshold:
            return trajectory, ""

        # Split into older (to compact) and recent (to preserve)
        older = trajectory[: -self.preserve_recent]
        recent = trajectory[-self.preserve_recent :]

        # Summarize older steps
        summary = await self._summarize_steps(older, task, goal)

        # Create compacted trajectory
        compacted = [
            {
                "step": "summary",
                "thought": f"[COMPACTED] Previous {len(older)} steps summarized",
                "action": "summary",
                "observation": summary.summary,
            }
        ]
        compacted.extend(recent)

        # Track compaction
        self.compaction_history.append(
            {
                "timestamp": time.time(),
                "steps_compacted": len(older),
                "steps_preserved": len(recent),
                "summary_length": len(summary.summary),
            }
        )

        return compacted, summary.summary

    async def _summarize_steps(
        self,
        steps: List[Dict[str, str]],
        task: str,
        goal: str,
    ) -> CompactionSummary:
        """Summarize a list of steps into a concise form."""
        steps_text = self._format_steps(steps)

        prompt = f"""
Task: {task}
Goal: {goal}

Summarize these {len(steps)} execution steps into a concise working memory:

{steps_text}

Create a summary that:
1. Captures the essential progress made
2. Notes key decisions and their outcomes
3. Preserves information needed for future steps
4. Is concise (max 200 words)
"""

        try:
            summary = await self.llm_client(
                system_prompt="You summarize agent execution traces for working memory.",
                user_prompt=prompt,
                response_model=CompactionSummary,
            )
            return summary
        except Exception as e:
            logger.warning(f"Compaction failed: {e}")
            return CompactionSummary(
                summary=f"Executed {len(steps)} steps toward goal.",
                key_actions=[s.get("action", "unknown") for s in steps[:3]],
                key_learnings=[],
                current_status="In progress",
            )

    def _format_steps(self, steps: List[Dict[str, str]]) -> str:
        """Format steps for summarization."""
        lines = []
        for i, step in enumerate(steps):
            lines.append(
                f"Step {i + 1}: {step.get('action', 'N/A')} -> {step.get('observation', 'N/A')[:100]}"
            )
        return "\n".join(lines)

    def get_stats(self) -> Dict[str, Any]:
        """Get compaction statistics."""
        if not self.compaction_history:
            return {"compactions": 0}

        total_compacted = sum(h["steps_compacted"] for h in self.compaction_history)
        return {
            "compactions": len(self.compaction_history),
            "total_steps_compacted": total_compacted,
            "avg_steps_per_compaction": total_compacted / len(self.compaction_history),
        }


# =============================================================================
# SECTION 4: MEMORY-AS-A-TOOL (REACTIVE RETRIEVAL)
# =============================================================================


class MemoryLoadDecision(BaseModel):
    """LLM decision on whether to load memory."""

    should_load: bool = Field(description="Whether memory retrieval is needed")
    query: str = Field(default="", description="Query to use for retrieval if loading")
    rationale: str = Field(description="Why memory is/isn't needed")
    expected_help: str = Field(
        default="", description="What kind of help is expected from memory"
    )


class ReactiveRetriever:
    """
    Memory-as-a-Tool: Agent decides when to retrieve.

    From whitepaper: "Instead of automatic retrieval, expose memory as a
    custom tool (e.g., load_memory). This allows the agent to use its Think
    step to decide if it needs to query its past, reducing latency."

    The retrieval decision uses an ADAPTIVE threshold instead of hardcoded values:
    - Base threshold is configurable (default: 5)
    - Adjusted based on task complexity (complex tasks get more retrieval opportunities)
    - Adjusted based on error encounters (errors trigger more retrieval)
    - Can be overridden to "always use LLM" mode for maximum accuracy
    """

    def __init__(
        self,
        memory_store: MemoryStore,
        llm_client: Optional[Callable] = None,
        retrieval_cost_ms: float = 100.0,
        # Configurable thresholds (no more hardcoded magic numbers!)
        base_skip_threshold: int = 5,  # Base step count to skip retrieval
        complexity_multiplier: float = 1.5,  # Multiply threshold for complex tasks
        error_extension: int = 2,  # Extend threshold after errors
        always_use_llm: bool = False,  # Never use heuristics, always ask LLM
        min_context_length: int = 200,  # Min context chars before skipping
    ):
        self.memory_store = memory_store
        self.llm_client = llm_client or precept_llm_client
        self.retrieval_cost_ms = retrieval_cost_ms

        # Configurable thresholds
        self.base_skip_threshold = base_skip_threshold
        self.complexity_multiplier = complexity_multiplier
        self.error_extension = error_extension
        self.always_use_llm = always_use_llm
        self.min_context_length = min_context_length

        # Dynamic state tracking
        self._current_task_complexity: float = 1.0
        self._errors_encountered: int = 0
        self._effective_threshold: int = base_skip_threshold

        # Statistics
        self.stats = {
            "decisions_made": 0,
            "retrievals_triggered": 0,
            "retrievals_skipped": 0,
            "latency_saved_ms": 0.0,
            "threshold_adjustments": 0,
        }

    def set_task_complexity(self, complexity: float) -> None:
        """
        Set task complexity to adjust retrieval threshold.

        Args:
            complexity: 1.0 = normal, 2.0 = complex, 0.5 = simple
        """
        self._current_task_complexity = max(0.5, min(3.0, complexity))
        self._recalculate_threshold()

    def record_error(self) -> None:
        """Record an error encounter to extend retrieval opportunities."""
        self._errors_encountered += 1
        self._recalculate_threshold()
        self.stats["threshold_adjustments"] += 1

    def reset_for_new_task(self) -> None:
        """Reset dynamic state for a new task."""
        self._current_task_complexity = 1.0
        self._errors_encountered = 0
        self._effective_threshold = self.base_skip_threshold

    def _recalculate_threshold(self) -> None:
        """Recalculate effective skip threshold based on current state."""
        # Start with base
        threshold = self.base_skip_threshold

        # Adjust for complexity (complex tasks get more retrieval opportunities)
        threshold = int(
            threshold * self._current_task_complexity * self.complexity_multiplier
        )

        # Extend for errors (errors mean we might need more help)
        threshold += self._errors_encountered * self.error_extension

        # Cap at reasonable maximum
        self._effective_threshold = min(threshold, 20)

    async def should_retrieve(
        self,
        task: str,
        current_context: str,
        step_count: int,
        has_recent_error: bool = False,
    ) -> MemoryLoadDecision:
        """
        Decide whether to retrieve from memory.

        Uses ADAPTIVE thresholds based on:
        - Task complexity
        - Error encounters
        - Context quality
        - Configurable base threshold

        Args:
            task: Current task description
            current_context: Accumulated context so far
            step_count: Current step number
            has_recent_error: Whether last action resulted in error
        """
        self.stats["decisions_made"] += 1

        # Track errors for adaptive threshold
        if has_recent_error:
            self.record_error()

        # === PROACTIVE: Always retrieve on first step ===
        if step_count == 0:
            return MemoryLoadDecision(
                should_load=True,
                query=task,
                rationale="First step - need context for planning",
                expected_help="Past experiences with similar tasks",
            )

        # === ALWAYS LLM MODE: Skip heuristics entirely ===
        if self.always_use_llm:
            decision = await self._llm_decision(task, current_context, step_count)
            self._track_decision(decision)
            return decision

        # === ADAPTIVE THRESHOLD CHECK ===
        # Check if we have sufficient context (not just step count!)
        context_is_sufficient = (
            step_count > self._effective_threshold
            and len(current_context) >= self.min_context_length
        )

        if context_is_sufficient:
            self.stats["retrievals_skipped"] += 1
            self.stats["latency_saved_ms"] += self.retrieval_cost_ms
            return MemoryLoadDecision(
                should_load=False,
                rationale=f"Sufficient context accumulated (threshold: {self._effective_threshold}, context: {len(current_context)} chars)",
            )

        # === REACTIVE: LLM decides for middle steps ===
        decision = await self._llm_decision(task, current_context, step_count)
        self._track_decision(decision)

        return decision

    def _track_decision(self, decision: MemoryLoadDecision) -> None:
        """Track retrieval decision statistics."""
        if decision.should_load:
            self.stats["retrievals_triggered"] += 1
        else:
            self.stats["retrievals_skipped"] += 1
            self.stats["latency_saved_ms"] += self.retrieval_cost_ms

    async def _llm_decision(
        self,
        task: str,
        current_context: str,
        step_count: int,
    ) -> MemoryLoadDecision:
        """Use LLM to decide on memory retrieval."""
        prompt = f"""
Task: {task}
Current step: {step_count}
Current context available:
{current_context[:500]}

Should I retrieve from long-term memory?

Consider:
- Do I have enough information to proceed?
- Would past experiences help with this specific situation?
- Is this a novel situation or a familiar one?

Retrieval has a latency cost of ~{self.retrieval_cost_ms}ms.
Only retrieve if it would meaningfully help.
"""

        try:
            return await self.llm_client(
                system_prompt="You decide when an agent should access its long-term memory.",
                user_prompt=prompt,
                response_model=MemoryLoadDecision,
            )
        except Exception:
            # Default to retrieval on uncertainty
            return MemoryLoadDecision(
                should_load=True,
                query=task,
                rationale="Defaulting to retrieval",
            )

    async def retrieve_if_needed(
        self,
        task: str,
        current_context: str,
        step_count: int,
        top_k: int = 5,
        domain: Optional[str] = None,
        has_recent_error: bool = False,
    ) -> Tuple[List[Experience], bool]:
        """
        Conditionally retrieve from memory.

        Args:
            task: Current task description
            current_context: Accumulated context so far
            step_count: Current step number
            top_k: Number of memories to retrieve
            domain: Optional domain filter
            has_recent_error: Whether last action resulted in error

        Returns: (retrieved_experiences, was_retrieval_triggered)
        """
        decision = await self.should_retrieve(
            task=task,
            current_context=current_context,
            step_count=step_count,
            has_recent_error=has_recent_error,
        )

        if not decision.should_load:
            return [], False

        query = decision.query or task
        experiences = self.memory_store.retrieve_relevant(
            query=query,
            top_k=top_k,
            domain=domain,
        )

        return experiences, True

    def get_stats(self) -> Dict[str, Any]:
        """Get retrieval statistics."""
        total_decisions = self.stats["decisions_made"]
        if total_decisions == 0:
            return self.stats

        return {
            **self.stats,
            "skip_rate": self.stats["retrievals_skipped"] / total_decisions,
            "trigger_rate": self.stats["retrievals_triggered"] / total_decisions,
        }


# =============================================================================
# SECTION 5: BACKGROUND MEMORY GENERATION (ASYNC REFINE)
# =============================================================================


@dataclass
class MemoryWriteJob:
    """A job for the background memory writer."""

    id: str
    task: str
    trajectory: List[Dict[str, str]]
    outcome: str
    timestamp: float = field(default_factory=time.time)
    priority: int = 1  # Higher = more urgent


class BackgroundMemoryWriter:
    """
    Asynchronous memory writing for low latency responses.

    From whitepaper: "Decouple the Soft Ingestion (writing the lesson) from
    the user response. The agent responds immediately, while a background
    service processes the Refine step and updates the vector database."
    """

    def __init__(
        self,
        memory_store: MemoryStore,
        procedural_store: ProceduralMemoryStore,
        llm_client: Optional[Callable] = None,
        max_queue_size: int = 100,
    ):
        self.memory_store = memory_store
        self.procedural_store = procedural_store
        self.llm_client = llm_client or precept_llm_client

        self.job_queue: Queue = Queue(maxsize=max_queue_size)
        self.is_running = False
        self.worker_thread: Optional[threading.Thread] = None

        # Statistics
        self.stats = {
            "jobs_queued": 0,
            "jobs_processed": 0,
            "jobs_failed": 0,
            "avg_processing_time_ms": 0.0,
        }

    def queue_memory_write(
        self,
        task: str,
        trajectory: List[Dict[str, str]],
        outcome: str,
        priority: int = 1,
    ) -> str:
        """
        Queue a memory write job for background processing.

        Returns immediately with job ID.
        """
        job = MemoryWriteJob(
            id=hashlib.md5(f"{task}:{time.time()}".encode()).hexdigest()[:10],
            task=task,
            trajectory=trajectory,
            outcome=outcome,
            priority=priority,
        )

        try:
            self.job_queue.put_nowait(job)
            self.stats["jobs_queued"] += 1
            return job.id
        except Exception:
            logger.warning("Memory write queue full, dropping job")
            return ""

    def start_background_worker(self):
        """Start the background worker thread."""
        if self.is_running:
            return

        self.is_running = True
        self.worker_thread = threading.Thread(target=self._worker_loop, daemon=True)
        self.worker_thread.start()
        logger.info("Background memory writer started")

    def stop_background_worker(self):
        """Stop the background worker."""
        self.is_running = False
        if self.worker_thread:
            self.worker_thread.join(timeout=5.0)
        logger.info("Background memory writer stopped")

    def _worker_loop(self):
        """Main worker loop for processing memory writes."""
        while self.is_running:
            try:
                # Get job with timeout
                job = self.job_queue.get(timeout=1.0)
                start_time = time.time()

                # Process the job (synchronously in this thread)
                asyncio.run(self._process_job(job))

                processing_time_ms = (time.time() - start_time) * 1000
                self._update_stats(processing_time_ms)

            except Exception:
                # Queue empty or other error, continue
                continue

    async def _process_job(self, job: MemoryWriteJob):
        """Process a single memory write job."""
        try:
            # Store declarative memory
            self.memory_store.store_experience(
                task_description=job.task,
                goal="",  # Extracted later
                trajectory=job.trajectory,
                outcome=job.outcome,
                correctness=1.0 if job.outcome == "success" else 0.0,
                strategy_used=self._extract_strategy(job.trajectory),
                lessons_learned=self._extract_lessons(job.trajectory),
                skills_demonstrated=[],
                domain="general",
                experience_type=(
                    ExperienceType.SUCCESS
                    if job.outcome == "success"
                    else ExperienceType.FAILURE
                ),
                priority=MemoryPriority.MEDIUM,
            )

            # CRITICAL: Persist to disk after storing!
            # This ensures experiences are saved to precept_experiences.json
            self.memory_store.save()

            # Also extract procedural memory if successful
            if job.outcome == "success":
                await self.procedural_store.extract_procedure(
                    trajectory=job.trajectory,
                    task=job.task,
                    outcome=job.outcome,
                )

            self.stats["jobs_processed"] += 1

        except Exception as e:
            logger.error(f"Memory write job failed: {e}")
            self.stats["jobs_failed"] += 1

    def _extract_strategy(self, trajectory: List[Dict[str, str]]) -> str:
        """Quick extraction of strategy from trajectory."""
        if not trajectory:
            return "unknown"

        actions = [step.get("action", "") for step in trajectory]
        return f"Strategy using {len(actions)} steps"

    def _extract_lessons(self, trajectory: List[Dict[str, str]]) -> List[str]:
        """Quick extraction of lessons from trajectory."""
        lessons = []
        for step in trajectory:
            obs = step.get("observation", "")
            if "failed" in obs.lower() or "error" in obs.lower():
                lessons.append(f"Avoid: {obs[:100]}")
            elif "success" in obs.lower():
                lessons.append(f"Works: {step.get('action', '')[:50]}")
        return lessons[:3]

    def _update_stats(self, processing_time_ms: float):
        """Update processing statistics."""
        n = self.stats["jobs_processed"]
        avg = self.stats["avg_processing_time_ms"]
        self.stats["avg_processing_time_ms"] = (avg * (n - 1) + processing_time_ms) / n

    def flush_and_wait(self, timeout: float = 10.0):
        """Wait for all queued jobs to complete."""
        start = time.time()
        while not self.job_queue.empty() and (time.time() - start) < timeout:
            time.sleep(0.1)

    def get_stats(self) -> Dict[str, Any]:
        """Get background writer statistics."""
        return {
            **self.stats,
            "queue_size": self.job_queue.qsize(),
        }


# =============================================================================
# SECTION 6: SMART CONSOLIDATION TRIGGERS
# =============================================================================


class ConflictType(Enum):
    """Types of memory conflicts that trigger consolidation."""

    DUPLICATE = "duplicate"  # Same information stored multiple times
    CONTRADICTION = "contradiction"  # Conflicting information
    SUPERSEDED = "superseded"  # Newer info replaces older
    REFINEMENT = "refinement"  # More specific version of general rule


@dataclass
class ConflictDetection:
    """Detected conflict between memories."""

    conflict_type: ConflictType
    memory_ids: List[str]
    description: str
    resolution_suggestion: str
    confidence: float


class SmartConsolidationTrigger:
    """
    Intelligent triggers for memory consolidation.

    From whitepaper: "When the memory manager detects Conflicting Information
    or Information Duplication (e.g., 50 separate memories saying Hamburg is
    closed), it flags this cluster. This signal triggers the GEPA optimizer."
    """

    def __init__(
        self,
        memory_store: MemoryStore,
        duplicate_threshold: int = 5,
        similarity_threshold: float = 0.8,
    ):
        self.memory_store = memory_store
        self.duplicate_threshold = duplicate_threshold
        self.similarity_threshold = similarity_threshold

        # Track patterns
        self.pattern_counts: Dict[str, int] = defaultdict(int)
        self.detected_conflicts: List[ConflictDetection] = []

    def should_consolidate(self) -> Tuple[bool, List[ConflictDetection]]:
        """
        Check if consolidation should be triggered.

        Returns: (should_consolidate, list_of_triggers)
        """
        conflicts = []

        # Check for duplicates
        duplicates = self._detect_duplicates()
        conflicts.extend(duplicates)

        # Check for contradictions
        contradictions = self._detect_contradictions()
        conflicts.extend(contradictions)

        self.detected_conflicts = conflicts

        # Trigger if we have enough significant conflicts
        should_trigger = len(conflicts) >= 3 or any(
            c.conflict_type == ConflictType.CONTRADICTION for c in conflicts
        )

        return should_trigger, conflicts

    def _detect_duplicates(self) -> List[ConflictDetection]:
        """Detect duplicate information in memories."""
        # Group memories by similar lessons
        lesson_groups: Dict[str, List[str]] = defaultdict(list)

        for exp in self.memory_store.episodic_memory.experiences:
            for lesson in exp.lessons_learned:
                normalized = lesson.lower().strip()[:50]
                lesson_groups[normalized].append(exp.id)

        # Find duplicates above threshold
        duplicates = []
        for lesson, memory_ids in lesson_groups.items():
            if len(memory_ids) >= self.duplicate_threshold:
                duplicates.append(
                    ConflictDetection(
                        conflict_type=ConflictType.DUPLICATE,
                        memory_ids=memory_ids,
                        description=f"'{lesson}' appears in {len(memory_ids)} memories",
                        resolution_suggestion="Consolidate into single rule",
                        confidence=0.9,
                    )
                )

        return duplicates

    def _detect_contradictions(self) -> List[ConflictDetection]:
        """Detect contradictory information in memories."""
        contradictions = []

        # Simple heuristic: look for opposite outcomes on similar tasks
        experiences = self.memory_store.episodic_memory.experiences

        for i, exp1 in enumerate(experiences):
            for exp2 in experiences[i + 1 :]:
                # Check if similar task but different outcome
                if self._are_similar_tasks(exp1, exp2) and exp1.outcome != exp2.outcome:
                    contradictions.append(
                        ConflictDetection(
                            conflict_type=ConflictType.CONTRADICTION,
                            memory_ids=[exp1.id, exp2.id],
                            description=f"Similar tasks with different outcomes: {exp1.outcome} vs {exp2.outcome}",
                            resolution_suggestion="Review and update rules",
                            confidence=0.7,
                        )
                    )

        return contradictions[:5]  # Limit to avoid explosion

    def _are_similar_tasks(self, exp1: Experience, exp2: Experience) -> bool:
        """Check if two experiences are for similar tasks."""
        words1 = set(exp1.task_description.lower().split())
        words2 = set(exp2.task_description.lower().split())

        if not words1 or not words2:
            return False

        overlap = len(words1 & words2)
        union = len(words1 | words2)

        return (overlap / union) >= self.similarity_threshold

    def record_pattern(self, pattern: str):
        """Record a pattern occurrence for duplicate detection."""
        normalized = pattern.lower().strip()[:50]
        self.pattern_counts[normalized] += 1

    def get_consolidation_candidates(self) -> List[str]:
        """Get patterns that are ready for consolidation."""
        return [
            pattern
            for pattern, count in self.pattern_counts.items()
            if count >= self.duplicate_threshold
        ]


# =============================================================================
# SECTION 7: IRRELEVANCE-BASED PRUNING
# =============================================================================


class IrrelevancePruner:
    """
    Prunes memories based on irrelevance after consolidation.

    From whitepaper: "Once the Instinct (Prompt) is updated, the system can
    identify old episodic memories as 'no longer useful' and delete them."
    """

    def __init__(
        self,
        memory_store: MemoryStore,
        max_age_days: float = 30.0,
        min_usefulness: float = 0.3,
        max_memories: int = 500,
    ):
        self.memory_store = memory_store
        self.max_age_days = max_age_days
        self.min_usefulness = min_usefulness
        self.max_memories = max_memories

        self.pruning_history: List[Dict[str, Any]] = []

    def prune(
        self,
        consolidated_rules: Set[str],
        force: bool = False,
    ) -> int:
        """
        Prune irrelevant memories.

        Removes memories that:
        1. Have been consolidated into rules
        2. Are too old without being accessed
        3. Have low usefulness scores
        4. Exceed capacity limits

        Returns: Number of memories pruned
        """
        pruned_count = 0
        current_time = time.time()

        experiences_to_remove = []

        for exp in self.memory_store.episodic_memory.experiences:
            should_remove = False
            reason = ""

            # Don't prune critical memories
            if exp.priority == MemoryPriority.CRITICAL:
                continue

            # Check if consolidated
            if self._is_consolidated(exp, consolidated_rules):
                should_remove = True
                reason = "consolidated"

            # Check age (only for low-priority)
            elif exp.priority in [MemoryPriority.LOW, MemoryPriority.MEDIUM]:
                age_days = (current_time - exp.timestamp) / 86400
                if age_days > self.max_age_days and exp.retrieval_count < 3:
                    should_remove = True
                    reason = "stale"

            # Check usefulness
            if exp.usefulness_score < self.min_usefulness and exp.retrieval_count > 5:
                should_remove = True
                reason = "unhelpful"

            if should_remove:
                experiences_to_remove.append((exp.id, reason))

        # Also check capacity
        if len(self.memory_store.episodic_memory.experiences) > self.max_memories:
            # Sort by value and mark excess for removal
            sorted_exps = sorted(
                self.memory_store.episodic_memory.experiences,
                key=lambda e: self._calculate_retention_value(e),
            )
            excess_count = len(sorted_exps) - self.max_memories
            for exp in sorted_exps[:excess_count]:
                if exp.id not in [e[0] for e in experiences_to_remove]:
                    experiences_to_remove.append((exp.id, "capacity"))

        # Perform removal
        ids_to_remove = {e[0] for e in experiences_to_remove}
        self.memory_store.episodic_memory.experiences = [
            e
            for e in self.memory_store.episodic_memory.experiences
            if e.id not in ids_to_remove
        ]

        pruned_count = len(experiences_to_remove)

        if pruned_count > 0:
            self.memory_store.episodic_memory._rebuild_indices()

            self.pruning_history.append(
                {
                    "timestamp": current_time,
                    "pruned_count": pruned_count,
                    "reasons": dict(
                        (
                            reason,
                            sum(1 for _, r in experiences_to_remove if r == reason),
                        )
                        for reason in ["consolidated", "stale", "unhelpful", "capacity"]
                    ),
                }
            )

        return pruned_count

    def _is_consolidated(self, exp: Experience, consolidated_rules: Set[str]) -> bool:
        """Check if experience's knowledge is consolidated."""
        # Check lessons
        for lesson in exp.lessons_learned:
            if any(lesson.lower() in rule.lower() for rule in consolidated_rules):
                return True

        # Check strategy
        if any(
            exp.strategy_used.lower() in rule.lower() for rule in consolidated_rules
        ):
            return True

        return False

    def _calculate_retention_value(self, exp: Experience) -> float:
        """Calculate value of retaining a memory."""
        current_time = time.time()

        # Factors
        recency = 1.0 / (1 + (current_time - exp.timestamp) / (86400 * 7))  # Week decay
        priority = exp.priority.value / 4.0
        usefulness = exp.usefulness_score
        access = min(1.0, exp.retrieval_count / 10.0)

        return 0.3 * recency + 0.3 * priority + 0.2 * usefulness + 0.2 * access

    def get_stats(self) -> Dict[str, Any]:
        """Get pruning statistics."""
        if not self.pruning_history:
            return {"total_pruned": 0}

        total_pruned = sum(h["pruned_count"] for h in self.pruning_history)
        return {
            "total_pruned": total_pruned,
            "pruning_events": len(self.pruning_history),
            "last_prune": self.pruning_history[-1] if self.pruning_history else None,
        }


# =============================================================================
# SECTION 8: INTEGRATED CONTEXT ENGINEERING MANAGER
# =============================================================================


class ContextEngineeringManager:
    """
    Master orchestrator for all Context Engineering optimizations.

    Integrates:
    1. Memory Scoping - Isolation and sharing
    2. Procedural Memory - Strategy storage
    3. Session Compaction - Trajectory compression
    4. Reactive Retrieval - Smart memory access
    5. Background Writing - Async memory updates
    6. Smart Consolidation - Conflict detection
    7. Irrelevance Pruning - Memory cleanup
    """

    def __init__(
        self,
        memory_store: MemoryStore,
        llm_client: Optional[Callable] = None,
        config: Optional[Dict[str, Any]] = None,
    ):
        config = config or {}
        self.memory_store = memory_store
        self.llm_client = llm_client or precept_llm_client

        # Initialize all components
        self.scope_manager = MemoryScopeManager()

        self.procedural_store = ProceduralMemoryStore(llm_client=self.llm_client)

        self.session_compactor = SessionCompactor(
            max_trajectory_length=config.get("max_trajectory_length", 10),
            compaction_threshold=config.get("compaction_threshold", 5),
            preserve_recent=config.get("preserve_recent", 3),
            llm_client=self.llm_client,
        )

        self.reactive_retriever = ReactiveRetriever(
            memory_store=memory_store,
            llm_client=self.llm_client,
            # Configurable retrieval thresholds (no more hardcoded values!)
            base_skip_threshold=config.get("retrieval_skip_threshold", 5),
            complexity_multiplier=config.get("complexity_multiplier", 1.5),
            error_extension=config.get("error_extension", 2),
            always_use_llm=config.get("always_use_llm_for_retrieval", False),
            min_context_length=config.get("min_context_length", 200),
        )

        self.background_writer = BackgroundMemoryWriter(
            memory_store=memory_store,
            procedural_store=self.procedural_store,
            llm_client=self.llm_client,
        )

        self.consolidation_trigger = SmartConsolidationTrigger(
            memory_store=memory_store,
            duplicate_threshold=config.get("duplicate_threshold", 5),
        )

        self.pruner = IrrelevancePruner(
            memory_store=memory_store,
            max_age_days=config.get("max_age_days", 30.0),
            min_usefulness=config.get("min_usefulness", 0.3),
            max_memories=config.get("max_memories", 500),
        )

        # Start background worker
        if config.get("enable_background_writer", True):
            self.background_writer.start_background_worker()

    def set_task_complexity(self, complexity: float) -> None:
        """
        Set current task complexity to adjust retrieval behavior.

        Called at task start based on complexity analysis.

        Args:
            complexity: 1.0 = normal, 2.0 = complex (more retrieval), 0.5 = simple (less retrieval)
        """
        self.reactive_retriever.set_task_complexity(complexity)

    def reset_for_new_task(self) -> None:
        """Reset all dynamic state for a new task."""
        self.reactive_retriever.reset_for_new_task()

    async def before_step(
        self,
        task: str,
        trajectory: List[Dict[str, str]],
        step_count: int,
        domain: Optional[str] = None,
        user_id: Optional[str] = None,
        has_recent_error: bool = False,
        task_complexity: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Called before each execution step.

        Returns context and recommendations.

        Args:
            task: Task description
            trajectory: Execution history so far
            step_count: Current step number
            domain: Optional domain filter
            user_id: Optional user for scoped memories
            has_recent_error: Whether last action resulted in error (extends retrieval)
            task_complexity: Optional complexity score (1.0 = normal)
        """
        result = {
            "should_retrieve": False,
            "retrieved_memories": [],
            "applicable_procedures": [],
            "scoped_memories": [],
            "compacted_trajectory": trajectory,
            "trajectory_summary": "",
            "effective_threshold": self.reactive_retriever._effective_threshold,
        }

        # Set task complexity if provided (on first step)
        if step_count == 0 and task_complexity is not None:
            self.set_task_complexity(task_complexity)

        # 1. Check if we should retrieve (with error awareness)
        current_context = self._format_context(trajectory)
        memories, did_retrieve = await self.reactive_retriever.retrieve_if_needed(
            task=task,
            current_context=current_context,
            step_count=step_count,
            domain=domain,
            has_recent_error=has_recent_error,
        )
        result["should_retrieve"] = did_retrieve
        result["retrieved_memories"] = memories

        # 2. Get applicable procedures
        result["applicable_procedures"] = (
            self.procedural_store.get_applicable_procedures(
                task=task,
                domain=domain,
            )
        )

        # 3. Get scoped memories
        result["scoped_memories"] = self.scope_manager.retrieve(
            query=task,
            user_id=user_id,
        )

        # 4. Compact trajectory if needed
        if len(trajectory) > self.session_compactor.compaction_threshold:
            compacted, summary = await self.session_compactor.compact_trajectory(
                trajectory=trajectory,
                task=task,
                goal="",
            )
            result["compacted_trajectory"] = compacted
            result["trajectory_summary"] = summary

        return result

    async def after_step(
        self,
        task: str,
        step: Dict[str, str],
        observation: str,
    ) -> None:
        """
        Called after each execution step.

        Records patterns for consolidation.
        """
        # Record pattern for duplicate detection
        if "lesson" in step or "error" in observation.lower():
            self.consolidation_trigger.record_pattern(observation[:100])

    async def after_task(
        self,
        task: str,
        trajectory: List[Dict[str, str]],
        outcome: str,
        user_id: Optional[str] = None,
        domain: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Called after task completion.

        Handles memory writing and consolidation checks.
        """
        result = {
            "memory_job_id": "",
            "should_consolidate": False,
            "consolidation_triggers": [],
            "pruned_count": 0,
        }

        # 1. Queue background memory write (non-blocking)
        result["memory_job_id"] = self.background_writer.queue_memory_write(
            task=task,
            trajectory=trajectory,
            outcome=outcome,
        )

        # 2. Check if consolidation should be triggered
        should_consolidate, triggers = self.consolidation_trigger.should_consolidate()
        result["should_consolidate"] = should_consolidate
        result["consolidation_triggers"] = triggers

        # 3. If consolidation triggered, prune afterward
        if should_consolidate:
            # In practice, this would trigger GEPA and then prune
            # For now, just note that pruning should happen after consolidation
            pass

        return result

    def prune_after_consolidation(
        self,
        consolidated_rules: Set[str],
    ) -> int:
        """Call this after GEPA consolidation completes."""
        return self.pruner.prune(consolidated_rules)

    def _format_context(self, trajectory: List[Dict[str, str]]) -> str:
        """Format trajectory as context string."""
        if not trajectory:
            return "No actions yet"

        lines = []
        for step in trajectory[-3:]:  # Last 3 steps
            lines.append(f"Action: {step.get('action', 'N/A')}")
            lines.append(f"Result: {step.get('observation', 'N/A')[:100]}")
        return "\n".join(lines)

    def get_prompt_context(
        self,
        user_id: Optional[str] = None,
        domain: Optional[str] = None,
    ) -> str:
        """
        Get all learned context for prompt injection.

        Combines:
        - Scoped memories
        - Procedures
        - Consolidated rules
        """
        sections = []

        # Procedures
        proc_section = self.procedural_store.get_all_as_prompt_section()
        if proc_section:
            sections.append(proc_section)

        # Scoped memories
        scoped = self.scope_manager.retrieve("", user_id=user_id)
        if scoped:
            sections.append("=== SCOPED KNOWLEDGE ===")
            for mem in scoped[:5]:
                sections.append(f"• [{mem.scope.value}] {mem.content}")

        return "\n\n".join(sections)

    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive statistics."""
        return {
            "scope_manager": self.scope_manager.get_stats(),
            "procedures": len(self.procedural_store.procedures),
            "session_compaction": self.session_compactor.get_stats(),
            "reactive_retrieval": self.reactive_retriever.get_stats(),
            "background_writer": self.background_writer.get_stats(),
            "consolidation_triggers": len(
                self.consolidation_trigger.detected_conflicts
            ),
            "pruner": self.pruner.get_stats(),
        }

    def shutdown(self):
        """Clean shutdown of background processes."""
        self.background_writer.stop_background_worker()


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================


def create_context_engineering_manager(
    memory_store: MemoryStore,
    llm_client: Optional[Callable] = None,
    **config,
) -> ContextEngineeringManager:
    """Factory function to create a configured ContextEngineeringManager."""
    return ContextEngineeringManager(
        memory_store=memory_store,
        llm_client=llm_client,
        config=config,
    )
