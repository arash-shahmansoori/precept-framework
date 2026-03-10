"""
Pareto-Memory Integration for GemEvo Framework.

Bridges COMPASS Pareto frontier with Evo-Memory's experience storage.

Key capabilities:
1. Store specialized prompts from Pareto frontier in memory
2. Route tasks to appropriate prompt versions based on task type
3. Use memory to inform prompt selection and evolution
4. Track prompt performance across task domains

This enables the agent to retrieve not just facts, but the specific
prompt version best suited for each task type.
"""

import hashlib
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

import numpy as np
from pydantic import BaseModel, Field

from .memory_store import MemoryStore
from .llm_clients import precept_llm_client


class TaskType(Enum):
    """Types of tasks for prompt routing."""
    
    RETRIEVAL = "retrieval"
    REASONING = "reasoning"
    CODING = "coding"
    CREATIVE = "creative"
    MULTI_HOP = "multi_hop"
    FACTUAL = "factual"
    ANALYTICAL = "analytical"
    GENERAL = "general"


@dataclass
class PromptVersion:
    """
    A version of a system prompt optimized for specific tasks.
    
    From COMPASS Pareto frontier, each version excels at different objectives.
    """
    
    id: str
    prompt_text: str
    
    # What this version is optimized for
    optimized_for: List[TaskType]
    strengths: List[str]
    weaknesses: List[str]
    
    # Performance metrics (from COMPASS evaluation)
    pareto_scores: Dict[str, float]  # objective_name -> score
    average_score: float
    
    # Task-specific performance tracking
    task_performance: Dict[str, List[float]] = field(default_factory=dict)  # task_type -> scores
    
    # Metadata
    generation: int = 0
    created_at: float = field(default_factory=time.time)
    selection_count: int = 0
    
    def record_performance(self, task_type: str, score: float) -> None:
        """Record performance on a task type."""
        if task_type not in self.task_performance:
            self.task_performance[task_type] = []
        self.task_performance[task_type].append(score)
    
    def get_task_score(self, task_type: str) -> float:
        """Get average score for a task type."""
        if task_type not in self.task_performance:
            return self.average_score  # Default to overall average
        scores = self.task_performance[task_type]
        return sum(scores) / len(scores) if scores else self.average_score
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "optimized_for": [t.value for t in self.optimized_for],
            "strengths": self.strengths,
            "weaknesses": self.weaknesses,
            "pareto_scores": self.pareto_scores,
            "average_score": self.average_score,
            "task_performance": self.task_performance,
            "generation": self.generation,
            "selection_count": self.selection_count,
        }


class TaskClassification(BaseModel):
    """Classification of a task for routing."""
    
    primary_type: str = Field(description="Primary task type")
    secondary_types: List[str] = Field(description="Secondary task characteristics")
    complexity: str = Field(description="Complexity level: simple, moderate, complex")
    domain: str = Field(description="Domain of the task")
    key_requirements: List[str] = Field(description="Key requirements for this task")


class TaskTypeRouter:
    """
    Routes tasks to appropriate prompt versions.
    
    Uses memory and task analysis to select the best prompt.
    """
    
    def __init__(
        self,
        llm_client: Optional[Callable] = None,
        use_llm_classification: bool = True,
    ):
        # Use actual OpenAI LLM client by default - NO MOCKS
        self.llm_client = llm_client or precept_llm_client
        self.use_llm_classification = use_llm_classification
        
        # Routing rules learned from experience
        self.routing_rules: Dict[str, str] = {}  # pattern -> task_type
        
        # System prompt for classification
        self.classification_system_prompt = """You are an expert at classifying tasks.

Analyze the given task and determine:
1. The primary type of task (retrieval, reasoning, coding, creative, multi_hop, factual, analytical, general)
2. Secondary characteristics
3. Complexity level
4. Domain
5. Key requirements

Be precise and consistent in your classifications."""
    
    async def classify_task(self, task: str, goal: str) -> TaskClassification:
        """
        Classify a task for prompt routing.
        
        Uses LLM if available, falls back to heuristics.
        """
        if self.use_llm_classification and self.llm_client:
            return await self._llm_classify(task, goal)
        else:
            return self._heuristic_classify(task, goal)
    
    async def _llm_classify(self, task: str, goal: str) -> TaskClassification:
        """Classify using LLM."""
        try:
            classification = await self.llm_client(
                system_prompt=self.classification_system_prompt,
                user_prompt=f"Task: {task}\nGoal: {goal}",
                response_model=TaskClassification,
            )
            return classification
        except Exception as e:
            print(f"LLM classification failed: {e}")
            return self._heuristic_classify(task, goal)
    
    def _heuristic_classify(self, task: str, goal: str) -> TaskClassification:
        """Classify using heuristics."""
        text = f"{task} {goal}".lower()
        
        # Detect task type from keywords
        primary_type = TaskType.GENERAL.value
        secondary_types = []
        
        if any(kw in text for kw in ["retrieve", "find", "search", "look up", "get"]):
            primary_type = TaskType.RETRIEVAL.value
        elif any(kw in text for kw in ["reason", "think", "analyze", "deduce", "infer"]):
            primary_type = TaskType.REASONING.value
        elif any(kw in text for kw in ["code", "program", "implement", "function", "debug"]):
            primary_type = TaskType.CODING.value
        elif any(kw in text for kw in ["create", "generate", "write", "compose", "design"]):
            primary_type = TaskType.CREATIVE.value
        elif any(kw in text for kw in ["multiple", "several", "combine", "compare", "across"]):
            secondary_types.append(TaskType.MULTI_HOP.value)
        elif any(kw in text for kw in ["fact", "true", "false", "verify", "check"]):
            primary_type = TaskType.FACTUAL.value
        
        # Detect complexity
        complexity = "moderate"
        word_count = len(text.split())
        if word_count < 20:
            complexity = "simple"
        elif word_count > 100 or "complex" in text or "difficult" in text:
            complexity = "complex"
        
        return TaskClassification(
            primary_type=primary_type,
            secondary_types=secondary_types,
            complexity=complexity,
            domain="general",
            key_requirements=[],
        )
    
    def add_routing_rule(self, pattern: str, task_type: str) -> None:
        """Add a learned routing rule."""
        self.routing_rules[pattern.lower()] = task_type
    
    def get_task_type_from_rules(self, task: str) -> Optional[str]:
        """Check if any learned rules match this task."""
        task_lower = task.lower()
        for pattern, task_type in self.routing_rules.items():
            if pattern in task_lower:
                return task_type
        return None


class ParetoMemoryManager:
    """
    Manages Pareto frontier prompts integrated with memory.
    
    This is the key integration point between COMPASS and Evo-Memory.
    """
    
    def __init__(
        self,
        memory_store: MemoryStore,
        llm_client: Optional[Callable] = None,
        task_router: Optional[TaskTypeRouter] = None,
    ):
        self.memory_store = memory_store
        # Use actual OpenAI LLM client by default - NO MOCKS
        self.llm_client = llm_client or precept_llm_client
        self.task_router = task_router or TaskTypeRouter(self.llm_client)
        
        # Prompt versions from COMPASS Pareto frontier
        self.prompt_versions: Dict[str, PromptVersion] = {}
        
        # Current active prompt
        self.active_prompt_id: Optional[str] = None
        
        # Selection history for learning
        self.selection_history: List[Dict[str, Any]] = []
        
        # Performance tracking
        self.prompt_task_scores: Dict[str, Dict[str, List[float]]] = {}  # prompt_id -> task_type -> scores
    
    def register_pareto_prompt(
        self,
        prompt_text: str,
        optimized_for: List[TaskType],
        pareto_scores: Dict[str, float],
        strengths: Optional[List[str]] = None,
        weaknesses: Optional[List[str]] = None,
        generation: int = 0,
    ) -> PromptVersion:
        """
        Register a prompt version from COMPASS Pareto frontier.
        
        This stores the prompt for task-specific retrieval.
        """
        prompt_id = self._generate_prompt_id(prompt_text)
        
        avg_score = sum(pareto_scores.values()) / len(pareto_scores) if pareto_scores else 0.0
        
        version = PromptVersion(
            id=prompt_id,
            prompt_text=prompt_text,
            optimized_for=optimized_for,
            strengths=strengths or [],
            weaknesses=weaknesses or [],
            pareto_scores=pareto_scores,
            average_score=avg_score,
            generation=generation,
        )
        
        self.prompt_versions[prompt_id] = version
        
        return version
    
    async def select_prompt_for_task(
        self,
        task: str,
        goal: str,
        domain: Optional[str] = None,
    ) -> PromptVersion:
        """
        Select the best prompt version for a given task.
        
        Uses task classification and memory to find the optimal prompt.
        """
        if not self.prompt_versions:
            raise ValueError("No prompt versions registered")
        
        # Classify the task
        classification = await self.task_router.classify_task(task, goal)
        
        # Find best matching prompt
        best_prompt = self._find_best_prompt(classification)
        
        # Update selection tracking
        best_prompt.selection_count += 1
        self.selection_history.append({
            "task": task,
            "classification": classification.dict(),
            "selected_prompt_id": best_prompt.id,
            "timestamp": time.time(),
        })
        
        self.active_prompt_id = best_prompt.id
        
        return best_prompt
    
    def _find_best_prompt(self, classification: TaskClassification) -> PromptVersion:
        """Find the best prompt for a classification."""
        primary_type = classification.primary_type
        
        # Score each prompt
        scored_prompts = []
        for prompt in self.prompt_versions.values():
            score = self._score_prompt_for_task(prompt, classification)
            scored_prompts.append((prompt, score))
        
        # Sort by score
        scored_prompts.sort(key=lambda x: x[1], reverse=True)
        
        return scored_prompts[0][0]
    
    def _score_prompt_for_task(
        self,
        prompt: PromptVersion,
        classification: TaskClassification,
    ) -> float:
        """
        Score a prompt for a given task classification.
        
        Uses GEPA-compliant empirical scoring based on historical performance.
        No arbitrary bonuses - all scores derived from actual task outcomes.
        
        From GEPA paper: "The system evaluates prompts based on their
        empirical performance on tasks, not heuristic features."
        """
        # Primary score: Historical performance on this task type (empirical)
        historical_score = prompt.get_task_score(classification.primary_type)
        
        # If we have historical data, use it directly (empirical scoring)
        if prompt.task_performance.get(classification.primary_type):
            # Pure empirical: average of past performance on this task type
            task_scores = prompt.task_performance[classification.primary_type]
            if len(task_scores) >= 3:
                # Enough data - use empirical mean
                return sum(task_scores) / len(task_scores)
        
        # If no historical data, use Pareto scores (also empirical from evaluation)
        primary_type = TaskType(classification.primary_type)
        
        # Use Pareto score for matching objective if available
        pareto_score_key = self._task_type_to_pareto_objective(primary_type)
        if pareto_score_key and pareto_score_key in prompt.pareto_scores:
            return prompt.pareto_scores[pareto_score_key]
        
        # Fall back to average of all Pareto scores (still empirical)
        if prompt.pareto_scores:
            return sum(prompt.pareto_scores.values()) / len(prompt.pareto_scores)
        
        # Last resort: use historical score or average
        return historical_score if historical_score > 0 else prompt.average_score
    
    def _task_type_to_pareto_objective(self, task_type: TaskType) -> Optional[str]:
        """Map task type to relevant Pareto objective."""
        mapping = {
            TaskType.GENERAL: "task_success_rate",
            TaskType.REASONING: "solution_quality",
            TaskType.RETRIEVAL: "task_success_rate",
            TaskType.ACTION: "step_efficiency",
            TaskType.ANALYSIS: "solution_quality",
            TaskType.CREATIVE: "solution_quality",
        }
        return mapping.get(task_type)
    
    def record_task_outcome(
        self,
        prompt_id: str,
        task_type: str,
        score: float,
    ) -> None:
        """
        Record the outcome of using a prompt for a task.
        
        This updates the prompt's performance tracking.
        """
        if prompt_id in self.prompt_versions:
            self.prompt_versions[prompt_id].record_performance(task_type, score)
        
        # Also track in our aggregated store
        if prompt_id not in self.prompt_task_scores:
            self.prompt_task_scores[prompt_id] = {}
        if task_type not in self.prompt_task_scores[prompt_id]:
            self.prompt_task_scores[prompt_id][task_type] = []
        self.prompt_task_scores[prompt_id][task_type].append(score)
    
    def get_prompt_recommendations(
        self,
        task_type: str,
        n: int = 3,
    ) -> List[Tuple[PromptVersion, float]]:
        """
        Get top N prompt recommendations for a task type.
        
        Returns list of (prompt, expected_score) tuples.
        """
        recommendations = []
        
        for prompt in self.prompt_versions.values():
            expected_score = prompt.get_task_score(task_type)
            recommendations.append((prompt, expected_score))
        
        recommendations.sort(key=lambda x: x[1], reverse=True)
        return recommendations[:n]
    
    def get_underperforming_prompts(
        self,
        threshold: float = 0.5,
    ) -> List[PromptVersion]:
        """Get prompts that are underperforming."""
        underperforming = []
        for prompt in self.prompt_versions.values():
            if prompt.average_score < threshold:
                underperforming.append(prompt)
            elif prompt.task_performance:
                # Check if any task type has low performance
                for scores in prompt.task_performance.values():
                    if scores and sum(scores) / len(scores) < threshold:
                        underperforming.append(prompt)
                        break
        return underperforming
    
    def get_prompt_diversity_report(self) -> Dict[str, Any]:
        """
        Generate a report on Pareto prompt diversity.
        
        Useful for understanding the prompt population.
        """
        if not self.prompt_versions:
            return {"status": "no_prompts"}
        
        # Coverage analysis
        type_coverage: Dict[str, int] = {}
        for prompt in self.prompt_versions.values():
            for task_type in prompt.optimized_for:
                type_coverage[task_type.value] = type_coverage.get(task_type.value, 0) + 1
        
        # Performance spread
        scores = [p.average_score for p in self.prompt_versions.values()]
        
        # Usage distribution
        selection_counts = [p.selection_count for p in self.prompt_versions.values()]
        
        return {
            "total_prompts": len(self.prompt_versions),
            "type_coverage": type_coverage,
            "score_range": {
                "min": min(scores),
                "max": max(scores),
                "avg": sum(scores) / len(scores),
            },
            "selection_stats": {
                "total_selections": sum(selection_counts),
                "most_selected": max(selection_counts) if selection_counts else 0,
                "least_selected": min(selection_counts) if selection_counts else 0,
            },
            "uncovered_types": [
                t.value for t in TaskType
                if t.value not in type_coverage
            ],
        }
    
    def export_for_compass(self) -> List[Dict[str, Any]]:
        """
        Export prompt performance data for COMPASS next evolution cycle.
        
        This feeds learned information back to the "Compiler".
        """
        export_data = []
        
        for prompt in self.prompt_versions.values():
            export_data.append({
                "prompt_id": prompt.id,
                "prompt_text": prompt.prompt_text,
                "generation": prompt.generation,
                "pareto_scores": prompt.pareto_scores,
                "runtime_performance": prompt.task_performance,
                "selection_count": prompt.selection_count,
                "strengths": prompt.strengths,
                "weaknesses": prompt.weaknesses,
            })
        
        return export_data
    
    def import_from_compass(
        self,
        candidates: List[Dict[str, Any]],
    ) -> int:
        """
        Import new prompt candidates from COMPASS evolution.
        
        Updates the Pareto frontier with newly evolved prompts.
        """
        imported = 0
        
        for candidate in candidates:
            # Parse task types
            optimized_for = []
            for type_str in candidate.get("optimized_for", ["general"]):
                try:
                    optimized_for.append(TaskType(type_str))
                except ValueError:
                    optimized_for.append(TaskType.GENERAL)
            
            if not optimized_for:
                optimized_for = [TaskType.GENERAL]
            
            self.register_pareto_prompt(
                prompt_text=candidate["prompt_text"],
                optimized_for=optimized_for,
                pareto_scores=candidate.get("scores", {}),
                strengths=candidate.get("strengths", []),
                weaknesses=candidate.get("weaknesses", []),
                generation=candidate.get("generation", 0),
            )
            imported += 1
        
        return imported
    
    def _generate_prompt_id(self, prompt_text: str) -> str:
        """Generate unique ID for a prompt."""
        return hashlib.md5(prompt_text.encode()).hexdigest()[:12]
    
    def get_stats(self) -> Dict[str, Any]:
        """Get manager statistics."""
        return {
            "total_prompts": len(self.prompt_versions),
            "total_selections": len(self.selection_history),
            "active_prompt": self.active_prompt_id,
            "diversity_report": self.get_prompt_diversity_report(),
        }

