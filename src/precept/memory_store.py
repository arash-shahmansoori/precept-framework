"""
Episodic Memory Storage for GemEvo Framework.

Implements Evo-Memory style experience storage with:
- Semantic indexing for efficient retrieval
- Experience compression and summarization
- Automatic pruning of redundant/low-value memories
- Domain-aware memory organization

Based on Evo-Memory paper's memory architecture.
"""

import fcntl
import hashlib
import json
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

import numpy as np


class ExperienceType(Enum):
    """Types of experiences that can be stored."""

    SUCCESS = "success"  # Successful task completion
    FAILURE = "failure"  # Failed attempt with lessons learned
    STRATEGY = "strategy"  # General strategy/approach
    EDGE_CASE = "edge_case"  # Unusual situation handling
    DOMAIN_RULE = "domain_rule"  # Domain-specific rules discovered


class MemoryPriority(Enum):
    """Priority levels for memory retention."""

    CRITICAL = 4  # Never prune automatically
    HIGH = 3  # Prune only under severe pressure
    MEDIUM = 2  # Standard pruning rules
    LOW = 1  # Prune aggressively


@dataclass
class Experience:
    """
    Represents a single experience/memory unit.

    Following Evo-Memory's structure for experience storage.
    """

    id: str
    task_description: str
    goal: str
    trajectory: List[Dict[str, str]]  # List of {action, observation, thought}
    outcome: str  # success/failure/partial
    correctness: float  # 0.0 to 1.0

    # Learned insights
    strategy_used: str
    lessons_learned: List[str]
    skills_demonstrated: List[str]

    # Metadata
    experience_type: ExperienceType
    priority: MemoryPriority
    domain: str
    timestamp: float = field(default_factory=time.time)
    retrieval_count: int = 0
    usefulness_score: float = 0.5  # Updated based on usage feedback

    # Compressed representation
    embedding: Optional[np.ndarray] = None
    summary: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "id": self.id,
            "task_description": self.task_description,
            "goal": self.goal,
            "trajectory": self.trajectory,
            "outcome": self.outcome,
            "correctness": self.correctness,
            "strategy_used": self.strategy_used,
            "lessons_learned": self.lessons_learned,
            "skills_demonstrated": self.skills_demonstrated,
            "experience_type": self.experience_type.value,
            "priority": self.priority.value,
            "domain": self.domain,
            "timestamp": self.timestamp,
            "retrieval_count": self.retrieval_count,
            "usefulness_score": self.usefulness_score,
            "summary": self.summary,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Experience":
        """Create Experience from dictionary."""
        return cls(
            id=data["id"],
            task_description=data["task_description"],
            goal=data["goal"],
            trajectory=data["trajectory"],
            outcome=data["outcome"],
            correctness=data["correctness"],
            strategy_used=data["strategy_used"],
            lessons_learned=data["lessons_learned"],
            skills_demonstrated=data["skills_demonstrated"],
            experience_type=ExperienceType(data["experience_type"]),
            priority=MemoryPriority(data["priority"]),
            domain=data["domain"],
            timestamp=data.get("timestamp", time.time()),
            retrieval_count=data.get("retrieval_count", 0),
            usefulness_score=data.get("usefulness_score", 0.5),
            summary=data.get("summary"),
        )

    def generate_id(self) -> str:
        """Generate unique ID from content."""
        content = f"{self.task_description}:{self.goal}:{self.strategy_used}"
        return hashlib.md5(content.encode()).hexdigest()[:12]

    def get_retrieval_text(self) -> str:
        """Get text representation for retrieval matching."""
        parts = [
            f"Task: {self.task_description}",
            f"Goal: {self.goal}",
            f"Strategy: {self.strategy_used}",
            f"Outcome: {self.outcome}",
        ]
        if self.lessons_learned:
            parts.append(f"Lessons: {'; '.join(self.lessons_learned)}")
        if self.summary:
            parts.append(f"Summary: {self.summary}")
        return "\n".join(parts)


@dataclass
class EpisodicMemory:
    """
    Collection of experiences organized as episodic memory.

    Supports:
    - Temporal organization (recent vs. old)
    - Domain-based clustering
    - Skill-based indexing
    """

    experiences: List[Experience] = field(default_factory=list)
    domain_index: Dict[str, List[str]] = field(
        default_factory=dict
    )  # domain -> [exp_ids]
    skill_index: Dict[str, List[str]] = field(
        default_factory=dict
    )  # skill -> [exp_ids]
    strategy_index: Dict[str, List[str]] = field(
        default_factory=dict
    )  # strategy -> [exp_ids]

    # FIX: Increased from 1000 to 2000 to reduce aggressive pruning
    max_size: int = 2000

    def add_experience(self, exp: Experience) -> None:
        """Add an experience to episodic memory."""
        self.experiences.append(exp)

        # Update domain index
        if exp.domain not in self.domain_index:
            self.domain_index[exp.domain] = []
        self.domain_index[exp.domain].append(exp.id)

        # Update skill index
        for skill in exp.skills_demonstrated:
            if skill not in self.skill_index:
                self.skill_index[skill] = []
            self.skill_index[skill].append(exp.id)

        # Update strategy index
        strategy_key = self._normalize_strategy(exp.strategy_used)
        if strategy_key not in self.strategy_index:
            self.strategy_index[strategy_key] = []
        self.strategy_index[strategy_key].append(exp.id)

        # Prune if needed
        if len(self.experiences) > self.max_size:
            self._prune_old_memories()

    def get_by_domain(self, domain: str) -> List[Experience]:
        """Get all experiences for a domain."""
        exp_ids = self.domain_index.get(domain, [])
        return [e for e in self.experiences if e.id in exp_ids]

    def get_by_skill(self, skill: str) -> List[Experience]:
        """Get all experiences demonstrating a skill."""
        exp_ids = self.skill_index.get(skill, [])
        return [e for e in self.experiences if e.id in exp_ids]

    def get_by_strategy(self, strategy: str) -> List[Experience]:
        """Get all experiences using a strategy."""
        strategy_key = self._normalize_strategy(strategy)
        exp_ids = self.strategy_index.get(strategy_key, [])
        return [e for e in self.experiences if e.id in exp_ids]

    def get_recent(self, n: int = 10) -> List[Experience]:
        """Get n most recent experiences."""
        sorted_exps = sorted(self.experiences, key=lambda e: e.timestamp, reverse=True)
        return sorted_exps[:n]

    def get_most_useful(self, n: int = 10) -> List[Experience]:
        """Get n most useful experiences based on usefulness score."""
        sorted_exps = sorted(
            self.experiences, key=lambda e: e.usefulness_score, reverse=True
        )
        return sorted_exps[:n]

    def _normalize_strategy(self, strategy: str) -> str:
        """Normalize strategy string for indexing."""
        return strategy.lower().strip()[:100]

    def _prune_old_memories(self) -> None:
        """Prune lowest-priority, oldest, least-useful memories.

        FIX: Reduced pruning aggressiveness:
        1. Protect experiences less than 24 hours old
        2. Give failure experiences higher scores (they're more valuable for learning)
        3. Only prune if significantly over capacity (20% buffer)
        """
        # Don't prune unless significantly over capacity
        if len(self.experiences) <= self.max_size * 1.2:
            return

        # Score each memory for pruning
        protected = []  # Recent experiences to always keep
        scores = []
        current_time = time.time()

        # FIX: 24-hour protection for recent experiences
        protection_window = 86400  # 24 hours in seconds

        for exp in self.experiences:
            age_seconds = current_time - exp.timestamp

            # FIX: Protect recent experiences (less than 24 hours old)
            if age_seconds < protection_window:
                protected.append(exp)
                continue

            # Lower score = more likely to prune
            age_factor = 1.0 / (
                1 + (current_time - exp.timestamp) / (86400 * 30)
            )  # 30 days decay
            priority_factor = exp.priority.value / 4.0
            usefulness_factor = exp.usefulness_score
            retrieval_factor = min(1.0, exp.retrieval_count / 10.0)

            # FIX: Failure experiences get a 1.5x boost (they're more valuable for learning)
            failure_boost = 1.5 if exp.experience_type.value == "failure" else 1.0

            score = failure_boost * (
                0.3 * age_factor
                + 0.3 * priority_factor
                + 0.2 * usefulness_factor
                + 0.2 * retrieval_factor
            )
            scores.append((exp, score))

        # Calculate how many we can keep from non-protected
        max_prunable = max(0, self.max_size - len(protected))

        # Sort by score and keep top max_prunable
        scores.sort(key=lambda x: x[1], reverse=True)
        self.experiences = protected + [s[0] for s in scores[:max_prunable]]

        # Rebuild indices
        self._rebuild_indices()

    def _rebuild_indices(self) -> None:
        """Rebuild all indices from experiences."""
        self.domain_index = {}
        self.skill_index = {}
        self.strategy_index = {}

        for exp in self.experiences:
            if exp.domain not in self.domain_index:
                self.domain_index[exp.domain] = []
            self.domain_index[exp.domain].append(exp.id)

            for skill in exp.skills_demonstrated:
                if skill not in self.skill_index:
                    self.skill_index[skill] = []
                self.skill_index[skill].append(exp.id)

            strategy_key = self._normalize_strategy(exp.strategy_used)
            if strategy_key not in self.strategy_index:
                self.strategy_index[strategy_key] = []
            self.strategy_index[strategy_key].append(exp.id)


class SemanticMemoryIndex:
    """
    Semantic index for efficient experience retrieval.

    Uses embedding-based similarity for finding relevant experiences.
    """

    def __init__(
        self,
        embedding_fn: Optional[Callable[[str], np.ndarray]] = None,
        similarity_threshold: float = 0.7,
    ):
        self.embedding_fn = embedding_fn
        self.similarity_threshold = similarity_threshold
        self.embeddings: Dict[str, np.ndarray] = {}
        self.exp_id_to_text: Dict[str, str] = {}

    def index_experience(self, exp: Experience) -> None:
        """Index an experience for semantic retrieval."""
        text = exp.get_retrieval_text()
        self.exp_id_to_text[exp.id] = text

        if self.embedding_fn:
            embedding = self.embedding_fn(text)
            self.embeddings[exp.id] = embedding
            exp.embedding = embedding

    def search(
        self,
        query: str,
        top_k: int = 5,
        domain_filter: Optional[str] = None,
    ) -> List[Tuple[str, float]]:
        """
        Search for similar experiences.

        Returns list of (exp_id, similarity_score) tuples.
        """
        if not self.embedding_fn or not self.embeddings:
            # Fallback to simple text matching
            return self._text_search(query, top_k)

        query_embedding = self.embedding_fn(query)
        similarities = []

        for exp_id, exp_embedding in self.embeddings.items():
            sim = self._cosine_similarity(query_embedding, exp_embedding)
            if sim >= self.similarity_threshold:
                similarities.append((exp_id, sim))

        # Sort by similarity
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:top_k]

    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Calculate cosine similarity between two vectors."""
        dot_product = np.dot(a, b)
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot_product / (norm_a * norm_b)

    def _text_search(self, query: str, top_k: int) -> List[Tuple[str, float]]:
        """Simple text-based search fallback."""
        query_words = set(query.lower().split())
        scores = []

        for exp_id, text in self.exp_id_to_text.items():
            text_words = set(text.lower().split())
            overlap = len(query_words & text_words)
            total = len(query_words | text_words)
            score = overlap / total if total > 0 else 0
            scores.append((exp_id, score))

        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]


class MemoryStore:
    """
    Main memory store combining episodic memory with semantic indexing.

    Provides the "Runtime Database" for the GemEvo framework.
    """

    def __init__(
        self,
        storage_path: Optional[Path] = None,
        embedding_fn: Optional[Callable[[str], np.ndarray]] = None,
        max_memories: int = 1000,
    ):
        # Convert string path to Path object if needed
        if storage_path is not None and isinstance(storage_path, str):
            storage_path = Path(storage_path)
        self.storage_path = storage_path
        self.episodic_memory = EpisodicMemory(max_size=max_memories)
        self.semantic_index = SemanticMemoryIndex(embedding_fn=embedding_fn)

        # Statistics tracking
        self.stats = {
            "total_added": 0,
            "total_retrieved": 0,
            "total_pruned": 0,
            "successful_retrievals": 0,
        }

        # Load existing memories if storage path exists
        if self.storage_path and self.storage_path.exists():
            self.load()

    def store_experience(
        self,
        task_description: str,
        goal: str,
        trajectory: List[Dict[str, str]],
        outcome: str,
        correctness: float,
        strategy_used: str,
        lessons_learned: List[str],
        skills_demonstrated: List[str],
        domain: str = "general",
        experience_type: ExperienceType = ExperienceType.SUCCESS,
        priority: MemoryPriority = MemoryPriority.MEDIUM,
        summary: Optional[str] = None,
    ) -> Experience:
        """
        Store a new experience in memory.

        This is the main entry point for adding experiences after task completion.
        """
        # Create experience
        exp = Experience(
            id="",  # Will be generated
            task_description=task_description,
            goal=goal,
            trajectory=trajectory,
            outcome=outcome,
            correctness=correctness,
            strategy_used=strategy_used,
            lessons_learned=lessons_learned,
            skills_demonstrated=skills_demonstrated,
            experience_type=experience_type,
            priority=priority,
            domain=domain,
            summary=summary,
        )
        exp.id = exp.generate_id()

        # Add to episodic memory
        self.episodic_memory.add_experience(exp)

        # Index for semantic retrieval
        self.semantic_index.index_experience(exp)

        self.stats["total_added"] += 1

        return exp

    def retrieve_relevant(
        self,
        query: str,
        top_k: int = 5,
        domain: Optional[str] = None,
        include_recent: bool = True,
    ) -> List[Experience]:
        """
        Retrieve relevant experiences for a given query/task.

        Combines semantic similarity with recency and domain filtering.
        """
        # Get semantically similar experiences
        semantic_results = self.semantic_index.search(query, top_k=top_k * 2)

        # Get experience objects
        exp_map = {e.id: e for e in self.episodic_memory.experiences}
        retrieved = []

        for exp_id, score in semantic_results:
            if exp_id in exp_map:
                exp = exp_map[exp_id]
                # Apply domain filter if specified
                if domain and exp.domain != domain:
                    continue
                retrieved.append(exp)
                exp.retrieval_count += 1

        # Add recent experiences if requested
        if include_recent:
            recent = self.episodic_memory.get_recent(n=3)
            for exp in recent:
                if exp not in retrieved:
                    retrieved.append(exp)
                    exp.retrieval_count += 1

        # Limit to top_k
        retrieved = retrieved[:top_k]

        self.stats["total_retrieved"] += len(retrieved)

        return retrieved

    def update_usefulness(self, exp_id: str, feedback: float) -> None:
        """
        Update usefulness score based on feedback.

        Called after an experience is used to help solve a task.
        feedback: -1.0 to 1.0 (negative = hindered, positive = helped)

        Counting logic:
        - feedback > 0: Memory helped (counts as successful retrieval)
        - feedback <= 0: Memory hindered or was neutral
        """
        for exp in self.episodic_memory.experiences:
            if exp.id == exp_id:
                # Exponential moving average update
                alpha = 0.3
                exp.usefulness_score = (
                    alpha * (0.5 + feedback * 0.5) + (1 - alpha) * exp.usefulness_score
                )
                # Count as successful if feedback is positive (helped the task)
                if feedback > 0:
                    self.stats["successful_retrievals"] += 1
                break

    def get_frequent_strategies(
        self, min_count: int = 5
    ) -> List[Tuple[str, int, float]]:
        """
        Get frequently used strategies with success rates.

        Returns: List of (strategy, count, avg_correctness)
        Used for memory consolidation.
        """
        strategy_stats: Dict[str, List[float]] = {}

        for exp in self.episodic_memory.experiences:
            strategy = exp.strategy_used
            if strategy not in strategy_stats:
                strategy_stats[strategy] = []
            strategy_stats[strategy].append(exp.correctness)

        results = []
        for strategy, correctness_scores in strategy_stats.items():
            if len(correctness_scores) >= min_count:
                avg_correctness = sum(correctness_scores) / len(correctness_scores)
                results.append((strategy, len(correctness_scores), avg_correctness))

        # Sort by count then correctness
        results.sort(key=lambda x: (x[1], x[2]), reverse=True)
        return results

    def get_frequent_lessons(self, min_count: int = 3) -> List[Tuple[str, int]]:
        """
        Get frequently learned lessons.

        Returns: List of (lesson, count)
        Used for memory consolidation.
        """
        lesson_counts: Dict[str, int] = {}

        for exp in self.episodic_memory.experiences:
            for lesson in exp.lessons_learned:
                lesson_counts[lesson] = lesson_counts.get(lesson, 0) + 1

        results = [
            (lesson, count)
            for lesson, count in lesson_counts.items()
            if count >= min_count
        ]
        results.sort(key=lambda x: x[1], reverse=True)
        return results

    def prune_consolidated(self, consolidated_items: Set[str]) -> int:
        """
        Prune memories that have been consolidated into prompts.

        Returns number of experiences pruned.
        """
        original_count = len(self.episodic_memory.experiences)

        # Remove experiences whose lessons/strategies are now consolidated
        self.episodic_memory.experiences = [
            exp
            for exp in self.episodic_memory.experiences
            if not self._is_consolidated(exp, consolidated_items)
        ]

        self.episodic_memory._rebuild_indices()

        pruned_count = original_count - len(self.episodic_memory.experiences)
        self.stats["total_pruned"] += pruned_count

        return pruned_count

    def _is_consolidated(self, exp: Experience, consolidated_items: Set[str]) -> bool:
        """Check if an experience's lessons are consolidated."""
        # Don't prune critical or high-priority memories
        if exp.priority in [MemoryPriority.CRITICAL, MemoryPriority.HIGH]:
            return False

        # Check if strategy or lessons are in consolidated set
        if exp.strategy_used in consolidated_items:
            return True

        consolidated_lessons = sum(
            1 for lesson in exp.lessons_learned if lesson in consolidated_items
        )
        if consolidated_lessons >= len(exp.lessons_learned) * 0.7:
            return True

        return False

    def save(self) -> None:
        """Save memory store to disk (thread-safe with file locking)."""
        if not self.storage_path:
            return

        self.storage_path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "experiences": [e.to_dict() for e in self.episodic_memory.experiences],
            "stats": self.stats,
            "timestamp": time.time(),
        }

        # Use exclusive file locking for concurrent-safe writes
        with open(self.storage_path, "w") as f:
            try:
                fcntl.flock(f.fileno(), fcntl.LOCK_EX)
                json.dump(data, f, indent=2)
                f.flush()
            finally:
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)

    def load(self) -> None:
        """Load memory store from disk (thread-safe with file locking)."""
        if not self.storage_path or not self.storage_path.exists():
            return

        # Use shared file locking for concurrent-safe reads
        with open(self.storage_path, "r") as f:
            try:
                fcntl.flock(f.fileno(), fcntl.LOCK_SH)
                data = json.load(f)
            except json.JSONDecodeError:
                return
            finally:
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)

        self.episodic_memory.experiences = [
            Experience.from_dict(e) for e in data.get("experiences", [])
        ]
        self.episodic_memory._rebuild_indices()

        for exp in self.episodic_memory.experiences:
            self.semantic_index.index_experience(exp)

        self.stats.update(data.get("stats", {}))

    def get_stats(self) -> Dict[str, Any]:
        """Get memory store statistics."""
        return {
            **self.stats,
            "current_size": len(self.episodic_memory.experiences),
            "domains": list(self.episodic_memory.domain_index.keys()),
            "skills_indexed": len(self.episodic_memory.skill_index),
            "strategies_indexed": len(self.episodic_memory.strategy_index),
        }
