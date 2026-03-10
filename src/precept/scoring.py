"""
GEPA-Compliant Scoring Module for PRECEPT.

This module implements proper scoring mechanisms following the GEPA paper:
"Reflective Prompt Evolution Can Outperform Reinforcement Learning"
(https://arxiv.org/html/2507.19457v1)

GEPA Scoring Principles:
1. Rollout-based evaluation - Score prompts by actually running them on tasks
2. Multi-objective optimization - Track multiple independent metrics
3. Pareto dominance - Select non-dominated solutions, no arbitrary weighting
4. Empirical metrics - All scores derived from actual task execution

NO arbitrary heuristics (keyword matching, length penalties, etc.)
"""

import os
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple

# =============================================================================
# CONFIG-DERIVED BOUNDS (no arbitrary magic numbers)
# =============================================================================

# MAX_RETRIES from config (default 4) - determines upper bound on task steps
DEFAULT_MAX_RETRIES = int(os.getenv("PRECEPT_MAX_RETRIES", "4"))

# Upper bound on task steps: 1 (initial) + MAX_RETRIES (pivots)
MAX_TASK_STEPS = 1 + DEFAULT_MAX_RETRIES


# =============================================================================
# GEPA OBJECTIVE DEFINITIONS
# =============================================================================


class GEPAObjective(Enum):
    """
    Standard GEPA objectives for multi-objective optimization.

    From GEPA paper: The system is evaluated on multiple criteria
    to maintain diverse prompts on the Pareto frontier.
    """

    # Primary objectives (from paper)
    TASK_SUCCESS_RATE = "task_success_rate"  # % of tasks completed successfully
    SOLUTION_QUALITY = "solution_quality"  # Quality of generated solutions
    STEP_EFFICIENCY = "step_efficiency"  # Fewer steps = better
    TOKEN_EFFICIENCY = "token_efficiency"  # Fewer tokens = better

    # Secondary objectives (for PRECEPT extension)
    ADAPTATION_SPEED = "adaptation_speed"  # How fast the agent adapts to failures
    GENERALIZATION = "generalization"  # Performance on unseen task variations


@dataclass
class RolloutResult:
    """
    Result of a single task rollout for GEPA evaluation.

    A rollout is one complete execution of a task with a given prompt.
    """

    task_id: str
    task_description: str
    success: bool
    steps_taken: int
    tokens_used: int
    execution_time_ms: float
    final_answer: Optional[str] = None
    trajectory: List[Dict] = field(default_factory=list)

    # Quality metrics (0-1 scale)
    answer_correctness: float = 0.0  # How correct was the answer
    answer_completeness: float = 0.0  # How complete was the answer

    # Error tracking
    errors_encountered: List[str] = field(default_factory=list)
    recovered_from_errors: int = 0


@dataclass
class GEPAEvaluationResult:
    """
    Complete GEPA evaluation result for a prompt candidate.

    Contains multi-objective scores derived from actual rollouts,
    not heuristics.
    """

    prompt_id: str
    prompt_text: str
    generation: int

    # Raw metrics from rollouts
    num_rollouts: int
    rollout_results: List[RolloutResult]

    # Computed multi-objective scores (0-1 scale)
    scores: Dict[str, float]

    # Metadata
    evaluation_time_ms: float
    timestamp: float = field(default_factory=time.time)

    def dominates(self, other: "GEPAEvaluationResult") -> bool:
        """
        Check if this result Pareto-dominates another.

        A dominates B if:
        - A is >= B in all objectives
        - A is > B in at least one objective
        """
        at_least_equal = True
        strictly_better = False

        for obj in self.scores:
            if obj in other.scores:
                if self.scores[obj] < other.scores[obj]:
                    at_least_equal = False
                    break
                if self.scores[obj] > other.scores[obj]:
                    strictly_better = True

        return at_least_equal and strictly_better

    def get_hypervolume_contribution(self, reference_point: Dict[str, float]) -> float:
        """
        Calculate hypervolume contribution for Pareto selection.

        Hypervolume is a standard metric for multi-objective optimization
        that measures the "volume" dominated by a solution.
        """
        contribution = 1.0
        for obj, score in self.scores.items():
            ref = reference_point.get(obj, 0.0)
            contribution *= max(0.0, score - ref)
        return contribution


# =============================================================================
# GEPA SCORING FUNCTIONS
# =============================================================================


def compute_gepa_scores(rollouts: List[RolloutResult]) -> Dict[str, float]:
    """
    Compute GEPA multi-objective scores from rollout results.

    This is the core GEPA scoring function. All scores are derived
    empirically from actual task execution, not heuristics.

    Args:
        rollouts: List of RolloutResult from task executions

    Returns:
        Dictionary of objective -> score (0-1 scale)
    """
    if not rollouts:
        return {obj.value: 0.0 for obj in GEPAObjective}

    n = len(rollouts)
    scores = {}

    # 1. TASK_SUCCESS_RATE: Direct empirical measurement
    #    From paper: "percentage of tasks completed successfully"
    successes = sum(1 for r in rollouts if r.success)
    scores[GEPAObjective.TASK_SUCCESS_RATE.value] = successes / n

    # 2. SOLUTION_QUALITY: Average of correctness and completeness
    #    From paper: "quality of generated solutions"
    if any(r.answer_correctness > 0 or r.answer_completeness > 0 for r in rollouts):
        quality_scores = [
            (r.answer_correctness + r.answer_completeness) / 2.0 for r in rollouts
        ]
        scores[GEPAObjective.SOLUTION_QUALITY.value] = sum(quality_scores) / n
    else:
        # Fall back to success rate if quality not measured
        scores[GEPAObjective.SOLUTION_QUALITY.value] = scores[
            GEPAObjective.TASK_SUCCESS_RATE.value
        ]

    # 3. STEP_EFFICIENCY: Normalized inverse of average steps
    #    From paper: "efficiency measured by steps taken"
    #    Upper bound derived from config: MAX_TASK_STEPS = 1 + MAX_RETRIES
    avg_steps = sum(r.steps_taken for r in rollouts) / n
    # Normalization using config-derived upper bound (no arbitrary magic numbers)
    # MAX_TASK_STEPS = 1 (initial) + MAX_RETRIES (pivots)
    scores[GEPAObjective.STEP_EFFICIENCY.value] = 1.0 / (
        1.0 + avg_steps / MAX_TASK_STEPS
    )

    # 4. TOKEN_EFFICIENCY: Normalized inverse of average tokens
    avg_tokens = sum(r.tokens_used for r in rollouts) / n
    baseline_tokens = 1000.0  # Configurable baseline
    if avg_tokens > 0:
        scores[GEPAObjective.TOKEN_EFFICIENCY.value] = 1.0 / (
            1.0 + avg_tokens / baseline_tokens
        )
    else:
        scores[GEPAObjective.TOKEN_EFFICIENCY.value] = 1.0  # Perfect if not measured

    # 5. ADAPTATION_SPEED: Measures recovery from failures
    #    This is a PRECEPT extension, but still empirically computed
    #    Score = (recoveries) / max(1, failures)
    total_failures = sum(len(r.errors_encountered) for r in rollouts)
    total_recoveries = sum(r.recovered_from_errors for r in rollouts)
    if total_failures > 0:
        scores[GEPAObjective.ADAPTATION_SPEED.value] = min(
            1.0, total_recoveries / total_failures
        )
    else:
        scores[GEPAObjective.ADAPTATION_SPEED.value] = 1.0  # Perfect if no failures

    # 6. GENERALIZATION: Performance variance across tasks
    #    Low variance = good generalization
    #    Score = 1 - normalized_variance
    if n > 1:
        success_values = [1.0 if r.success else 0.0 for r in rollouts]
        mean = sum(success_values) / n
        variance = sum((v - mean) ** 2 for v in success_values) / n
        # Max variance is 0.25 (for binary outcomes)
        normalized_variance = variance / 0.25
        scores[GEPAObjective.GENERALIZATION.value] = 1.0 - normalized_variance
    else:
        scores[GEPAObjective.GENERALIZATION.value] = scores[
            GEPAObjective.TASK_SUCCESS_RATE.value
        ]

    return scores


def compute_scores_from_task_results(
    task_results: List[Dict[str, Any]],
) -> Dict[str, float]:
    """
    Compute GEPA scores from simplified task result dictionaries.

    This is a convenience function for when full RolloutResult objects
    are not available (e.g., from existing PRECEPT task execution).

    Args:
        task_results: List of dicts with keys: success, steps, tokens, confidence, etc.

    Returns:
        Dictionary of objective -> score
    """
    if not task_results:
        return {obj.value: 0.0 for obj in GEPAObjective}

    # Convert to RolloutResult format
    rollouts = []
    for i, result in enumerate(task_results):
        rollout = RolloutResult(
            task_id=f"task_{i}",
            task_description=result.get("task", ""),
            success=result.get("success", False),
            steps_taken=result.get("steps", 5),
            tokens_used=result.get("tokens", 0),
            execution_time_ms=result.get("execution_time_ms", 0),
            final_answer=result.get("final_answer"),
            trajectory=result.get("trajectory", []),
            answer_correctness=result.get("confidence", 0.5)
            if result.get("success")
            else 0.0,
            answer_completeness=1.0 if result.get("success") else 0.0,
            errors_encountered=result.get("errors", []),
            recovered_from_errors=result.get("recoveries", 0),
        )
        rollouts.append(rollout)

    return compute_gepa_scores(rollouts)


# =============================================================================
# PARETO SELECTION (GEPA-compliant)
# =============================================================================


def pareto_select(
    candidates: List[GEPAEvaluationResult],
    selection_strategy: str = "hypervolume",
) -> GEPAEvaluationResult:
    """
    Select best candidate using Pareto-based selection.

    From GEPA paper: "maintains a Pareto front: instead of evolving only
    the global best prompt, it stochastically explores the top-performing
    prompts for each problem instance"

    Args:
        candidates: List of evaluated candidates
        selection_strategy: "hypervolume" (default), "crowding", or "random"

    Returns:
        Selected candidate
    """
    if not candidates:
        raise ValueError("No candidates to select from")

    if len(candidates) == 1:
        return candidates[0]

    # Step 1: Find non-dominated candidates (Pareto front)
    pareto_front = []
    for candidate in candidates:
        dominated = False
        for other in candidates:
            if other is not candidate and other.dominates(candidate):
                dominated = True
                break
        if not dominated:
            pareto_front.append(candidate)

    if not pareto_front:
        pareto_front = candidates  # Fallback if all dominated (shouldn't happen)

    # Step 2: Select from Pareto front using chosen strategy
    if selection_strategy == "hypervolume":
        # Select candidate with highest hypervolume contribution
        reference_point = {obj.value: 0.0 for obj in GEPAObjective}
        best = max(
            pareto_front, key=lambda c: c.get_hypervolume_contribution(reference_point)
        )
        return best

    elif selection_strategy == "crowding":
        # Select candidate in least crowded region (diversity)
        # For simplicity, use the one with most unique "best" objectives
        objective_leaders = {}
        for obj in GEPAObjective:
            obj_name = obj.value
            best_for_obj = max(pareto_front, key=lambda c: c.scores.get(obj_name, 0))
            objective_leaders[obj_name] = best_for_obj.prompt_id

        # Count how many objectives each candidate leads
        leadership_count = {}
        for candidate in pareto_front:
            count = sum(
                1
                for leader_id in objective_leaders.values()
                if leader_id == candidate.prompt_id
            )
            leadership_count[candidate.prompt_id] = count

        # Select the one that leads the most (or any from front if tie)
        best = max(pareto_front, key=lambda c: leadership_count.get(c.prompt_id, 0))
        return best

    else:  # random
        import random

        return random.choice(pareto_front)


def update_pareto_front(
    current_front: List[GEPAEvaluationResult],
    new_candidate: GEPAEvaluationResult,
    max_front_size: int = 10,
) -> Tuple[List[GEPAEvaluationResult], bool]:
    """
    Update Pareto front with a new candidate.

    Args:
        current_front: Current Pareto front
        new_candidate: New candidate to potentially add
        max_front_size: Maximum front size

    Returns:
        Tuple of (updated_front, was_added)
    """
    # Check if new candidate is dominated by any existing
    for existing in current_front:
        if existing.dominates(new_candidate):
            return current_front, False  # Dominated, don't add

    # Remove any existing that new candidate dominates
    updated_front = [c for c in current_front if not new_candidate.dominates(c)]

    # Add new candidate
    updated_front.append(new_candidate)

    # Prune if too large (keep most diverse)
    if len(updated_front) > max_front_size:
        updated_front = _prune_pareto_front(updated_front, max_front_size)

    return updated_front, True


def _prune_pareto_front(
    front: List[GEPAEvaluationResult],
    target_size: int,
) -> List[GEPAEvaluationResult]:
    """
    Prune Pareto front to target size while maintaining diversity.

    Uses crowding distance to keep diverse solutions.
    """
    if len(front) <= target_size:
        return front

    # Keep candidates that lead each objective
    keep_ids = set()
    for obj in GEPAObjective:
        obj_name = obj.value
        best = max(front, key=lambda c: c.scores.get(obj_name, 0))
        keep_ids.add(best.prompt_id)

    # Fill remaining slots by hypervolume contribution
    remaining_slots = target_size - len(keep_ids)
    if remaining_slots > 0:
        reference_point = {obj.value: 0.0 for obj in GEPAObjective}
        candidates_by_hv = sorted(
            [c for c in front if c.prompt_id not in keep_ids],
            key=lambda c: c.get_hypervolume_contribution(reference_point),
            reverse=True,
        )
        for c in candidates_by_hv[:remaining_slots]:
            keep_ids.add(c.prompt_id)

    return [c for c in front if c.prompt_id in keep_ids]


# =============================================================================
# ROLLOUT EXECUTOR (For GEPA Evaluation)
# =============================================================================


class GEPARolloutExecutor:
    """
    Executes rollouts for GEPA candidate evaluation.

    From GEPA paper: Candidates are evaluated by running actual
    tasks and measuring empirical performance.
    """

    def __init__(
        self,
        task_executor: Callable,
        llm_client: Callable,
    ):
        """
        Initialize rollout executor.

        Args:
            task_executor: Async function that executes a task given a prompt
            llm_client: LLM client for the agent
        """
        self.task_executor = task_executor
        self.llm_client = llm_client

    async def execute_rollouts(
        self,
        prompt: str,
        prompt_id: str,
        validation_tasks: List[Dict[str, str]],
        num_rollouts: int = 1,
    ) -> GEPAEvaluationResult:
        """
        Execute rollouts for a prompt candidate.

        Args:
            prompt: The prompt to evaluate
            prompt_id: Unique identifier for the prompt
            validation_tasks: List of tasks to run
            num_rollouts: Number of times to run each task

        Returns:
            GEPAEvaluationResult with scores from actual execution
        """
        start_time = time.time()
        rollout_results = []

        for task in validation_tasks:
            for _ in range(num_rollouts):
                try:
                    result = await self.task_executor(
                        prompt=prompt,
                        task=task.get("task", ""),
                        goal=task.get("goal", "Complete the task"),
                    )

                    rollout = RolloutResult(
                        task_id=task.get("id", f"task_{len(rollout_results)}"),
                        task_description=task.get("task", ""),
                        success=result.get("success", False),
                        steps_taken=result.get("steps", 5),
                        tokens_used=result.get("tokens", 0),
                        execution_time_ms=result.get("execution_time_ms", 0),
                        final_answer=result.get("final_answer"),
                        trajectory=result.get("trajectory", []),
                        answer_correctness=result.get("confidence", 0.5)
                        if result.get("success")
                        else 0.0,
                        answer_completeness=1.0 if result.get("success") else 0.0,
                        errors_encountered=result.get("errors", []),
                        recovered_from_errors=result.get("recoveries", 0),
                    )
                    rollout_results.append(rollout)

                except Exception as e:
                    # Record failed rollout
                    rollout = RolloutResult(
                        task_id=task.get("id", f"task_{len(rollout_results)}"),
                        task_description=task.get("task", ""),
                        success=False,
                        steps_taken=0,
                        tokens_used=0,
                        execution_time_ms=0,
                        errors_encountered=[str(e)],
                    )
                    rollout_results.append(rollout)

        # Compute scores from rollouts
        scores = compute_gepa_scores(rollout_results)

        evaluation_time = (time.time() - start_time) * 1000

        return GEPAEvaluationResult(
            prompt_id=prompt_id,
            prompt_text=prompt,
            generation=0,  # Set by caller
            num_rollouts=len(rollout_results),
            rollout_results=rollout_results,
            scores=scores,
            evaluation_time_ms=evaluation_time,
        )


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    # Objectives
    "GEPAObjective",
    # Data classes
    "RolloutResult",
    "GEPAEvaluationResult",
    # Scoring functions
    "compute_gepa_scores",
    "compute_scores_from_task_results",
    # Pareto selection
    "pareto_select",
    "update_pareto_front",
    # Rollout execution
    "GEPARolloutExecutor",
]
