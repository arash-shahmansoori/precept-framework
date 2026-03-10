"""
GEPA (Genetic-Pareto Evolution): Reflective Prompt Evolution Engine.

Based on the GEPA paper: "Reflective Prompt Evolution Can Outperform Reinforcement Learning"
(https://arxiv.org/html/2507.19457v1)

Key Principles:
1. Reflective Prompt Mutation - learn from trajectory analysis
2. Pareto-based Candidate Selection - maintain diverse frontier
3. Multi-objective Evolutionary Search - optimize multiple metrics
4. System-aware Crossover - combine complementary lessons

This module provides:
- GEPAReflection: Diagnoses problems from trajectories
- GEPAMutation: Generates improved prompt variants
- GEPAParetoCandidate: Candidates on the Pareto frontier
- GEPAEvolutionEngine: Core evolution engine
"""

import hashlib
import random
from typing import Any, Callable, Dict, List, Optional

from pydantic import BaseModel, Field


# =============================================================================
# GEPA PYDANTIC MODELS
# =============================================================================

class GEPAReflection(BaseModel):
    """
    GEPA Reflective Analysis - diagnoses problems from trajectory.
    
    From GEPA paper: "reflects on them in natural language to diagnose
    problems, propose and test prompt updates"
    """
    diagnosis: str = Field(description="What went wrong or could be improved")
    root_cause: str = Field(description="Root cause of the issue")
    suggested_fix: str = Field(description="Specific fix to address the issue")
    confidence: float = Field(description="Confidence in this diagnosis (0-1)")
    affected_objectives: List[str] = Field(
        default_factory=list,
        description="Which objectives this affects"
    )


class GEPAMutation(BaseModel):
    """
    GEPA Prompt Mutation - generates improved prompt variant.
    
    From GEPA paper: "the candidate prompt is derived from an ancestor,
    accumulating high-level lessons derived from observations"
    """
    mutated_prompt: str = Field(description="The mutated prompt text")
    mutation_type: str = Field(description="Type: addition, removal, rewrite, merge")
    lessons_incorporated: List[str] = Field(
        default_factory=list,
        description="Lessons from experience"
    )
    expected_improvement: str = Field(description="Expected improvement")
    parent_prompt_id: str = Field(default="", description="ID of the parent prompt")

    @property
    def changes_made(self) -> List[str]:
        """Alias for lessons_incorporated for backward compatibility."""
        return self.lessons_incorporated


class GEPAParetoCandidate(BaseModel):
    """
    A candidate on the Pareto frontier.
    
    From GEPA paper: "maintains a Pareto front: instead of evolving only
    the global best prompt, it stochastically explores the top-performing
    prompts for each problem instance"
    """
    prompt_id: str
    prompt_text: str
    scores: Dict[str, float]  # objective -> score
    generation: int
    parent_id: Optional[str] = None
    mutation_history: List[str] = Field(default_factory=list)
    
    def dominates(self, other: "GEPAParetoCandidate") -> bool:
        """
        Check if this candidate Pareto-dominates another.
        
        Candidate A dominates B if:
        - A is at least as good as B in all objectives
        - A is strictly better than B in at least one objective
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
    
    def get_average_score(self) -> float:
        """Get average score across all objectives."""
        if not self.scores:
            return 0.0
        return sum(self.scores.values()) / len(self.scores)


class GEPAConfig(BaseModel):
    """Configuration for GEPA Evolution Engine."""
    
    # Objectives to optimize
    objectives: List[str] = Field(
        default=[
            "success_rate",
            "step_efficiency",
            "adaptation_speed",
            "rule_generalization",
        ]
    )
    
    # Pareto selection
    max_pareto_front_size: int = 10
    selection_noise: float = 0.2  # Random noise for stochastic selection
    
    # Mutation
    min_reflection_confidence: float = 0.5
    max_lessons_per_mutation: int = 5
    
    # Evolution
    generations_to_keep: int = 5


# =============================================================================
# GEPA EVOLUTION ENGINE
# =============================================================================

class GEPAEvolutionEngine:
    """
    GEPA-style evolutionary engine for PRECEPT.
    
    Implements the key GEPA principles from the paper:
    1. Reflective Prompt Mutation
    2. Pareto-based Candidate Selection
    3. Multi-objective Optimization
    4. System-aware Crossover
    
    Usage:
        engine = GEPAEvolutionEngine(llm_client=my_llm_client)
        
        # Initialize with base prompt
        engine.initialize_pareto_front(base_prompt)
        
        # Run evolution cycle
        reflection = await engine.reflect_on_trajectory(trajectory, task, success, prompt)
        mutation = await engine.mutate_prompt(parent, parent_id, [reflection], rules)
        scores = engine.evaluate_candidate(task_results)
        candidate = engine.create_candidate(mutation, scores, parent_id)
        added = engine.update_pareto_front(candidate)
    """
    
    def __init__(
        self,
        llm_client: Callable,
        config: Optional[GEPAConfig] = None,
        learned_rules_getter: Optional[Callable[[], List[str]]] = None,
    ):
        """
        Initialize GEPA Evolution Engine.
        
        Args:
            llm_client: Async LLM client for reflection and mutation
            config: Optional GEPA configuration
            learned_rules_getter: Optional callable to get learned rules for evaluation
        """
        self.llm_client = llm_client
        self.config = config or GEPAConfig()
        self.objectives = self.config.objectives
        self.learned_rules_getter = learned_rules_getter or (lambda: [])
        
        # Pareto frontier
        self.pareto_front: List[GEPAParetoCandidate] = []
        
        # Evolution history
        self.generation = 0
        self.mutation_history: List[Dict] = []
        
        # Statistics
        self.stats = {
            "total_mutations": 0,
            "successful_mutations": 0,
            "pareto_updates": 0,
            "reflections_performed": 0,
            "candidates_dominated": 0,
        }
    
    def initialize_pareto_front(
        self,
        base_prompt: str,
        initial_scores: Optional[Dict[str, float]] = None,
    ) -> GEPAParetoCandidate:
        """
        Initialize Pareto front with a base prompt.
        
        Args:
            base_prompt: The initial system prompt
            initial_scores: Optional initial scores (defaults to 0.5 for all)
        
        Returns:
            The created base candidate
        """
        scores = initial_scores or {obj: 0.5 for obj in self.objectives}
        
        base_candidate = GEPAParetoCandidate(
            prompt_id=self._generate_prompt_id(base_prompt),
            prompt_text=base_prompt,
            scores=scores,
            generation=0,
            parent_id=None,
            mutation_history=[],
        )
        
        self.pareto_front.append(base_candidate)
        return base_candidate
    
    async def reflect_on_trajectory(
        self,
        trajectory: List[Dict],
        task: str,
        success: bool,
        prompt_used: str,
    ) -> GEPAReflection:
        """
        GEPA Reflective Analysis - diagnose problems from trajectory.
        
        From paper: "Given any AI system containing one or more LLM prompts,
        GEPA samples system-level trajectories (e.g., reasoning, tool calls,
        and tool outputs) and reflects on them in natural language to diagnose
        problems"
        
        Args:
            trajectory: List of trajectory steps (thought, action, observation)
            task: The task description
            success: Whether the task succeeded
            prompt_used: The prompt that was used
        
        Returns:
            GEPAReflection with diagnosis and suggested fix
        """
        # Format trajectory for analysis
        trajectory_text = "\n".join([
            f"Step {i+1}: {step.get('thought', '')} → {step.get('action', '')} → {step.get('observation', '')}"
            for i, step in enumerate(trajectory)
        ])
        
        reflection_prompt = f"""Analyze this AI agent trajectory and diagnose any issues:

TASK: {task}
OUTCOME: {'SUCCESS' if success else 'FAILURE'}

PROMPT USED:
{prompt_used[:500]}...

TRAJECTORY:
{trajectory_text}

Diagnose:
1. What went wrong or could be improved?
2. What is the root cause?
3. What specific fix would address this?
4. Which objectives does this affect ({', '.join(self.objectives)})?
"""
        
        try:
            reflection = await self.llm_client(
                system_prompt="You are an expert at analyzing AI agent behavior and diagnosing problems.",
                user_prompt=reflection_prompt,
                response_model=GEPAReflection,
            )
            self.stats["reflections_performed"] += 1
            return reflection
        except Exception as e:
            import logging
            logging.getLogger("precept.gepa").warning(f"GEPA Reflection failed: {e}")
            return GEPAReflection(
                diagnosis="Unable to diagnose",
                root_cause="Unknown",
                suggested_fix="No fix available",
                confidence=0.0,
                affected_objectives=[],
            )
    
    async def mutate_prompt(
        self,
        parent_prompt: str,
        parent_id: str,
        reflections: List[GEPAReflection],
        learned_rules: List[str],
    ) -> GEPAMutation:
        """
        GEPA Reflective Prompt Mutation.
        
        From paper: "In each mutation, the candidate prompt is derived from
        an ancestor, accumulating high-level lessons derived from observations
        and LLM feedback."
        
        Args:
            parent_prompt: The parent prompt to mutate
            parent_id: ID of the parent prompt
            reflections: List of reflections to incorporate
            learned_rules: List of learned rules to add
        
        Returns:
            GEPAMutation with the mutated prompt
        """
        # Compile lessons from reflections
        lessons = []
        for r in reflections:
            if r.confidence >= self.config.min_reflection_confidence:
                lessons.append(f"- {r.suggested_fix} (from: {r.diagnosis})")
        
        # Limit lessons
        lessons = lessons[:self.config.max_lessons_per_mutation]
        
        # Add learned rules
        for rule in learned_rules:
            lessons.append(f"- LEARNED RULE: {rule}")
        
        mutation_prompt = f"""Generate an improved version of this prompt by incorporating lessons learned:

ORIGINAL PROMPT:
{parent_prompt}

LESSONS TO INCORPORATE:
{chr(10).join(lessons) if lessons else "No specific lessons - improve generally."}

IMPROVEMENT GOALS:
- Make rules more explicit and actionable
- Add learned constraints directly to instructions
- Improve clarity and structure
- Maintain the core capabilities

Generate a mutated prompt that incorporates these lessons. The mutation should be substantial enough to test new behaviors but not so different as to lose proven strategies.
"""
        
        try:
            mutation = await self.llm_client(
                system_prompt="You are an expert prompt engineer. Generate improved prompts that incorporate lessons learned.",
                user_prompt=mutation_prompt,
                response_model=GEPAMutation,
            )
            mutation.parent_prompt_id = parent_id
            self.stats["total_mutations"] += 1
            return mutation
        except Exception as e:
            import logging
            logging.getLogger("precept.gepa").warning(f"GEPA Mutation failed: {e}")
            # Fallback: simple rule injection
            mutated = parent_prompt
            if learned_rules:
                mutated += "\n\n### LEARNED RULES ###\n"
                for rule in learned_rules:
                    mutated += f"• {rule}\n"
            
            return GEPAMutation(
                mutated_prompt=mutated,
                mutation_type="addition",
                lessons_incorporated=learned_rules,
                expected_improvement="Added learned rules to prompt",
                parent_prompt_id=parent_id,
            )
    
    def evaluate_candidate(
        self,
        task_results: List[Dict],
    ) -> Dict[str, float]:
        """
        Multi-objective evaluation following GEPA principles.
        
        Uses the GEPA-compliant scoring module for proper empirical evaluation.
        All scores are derived from actual task execution results, not heuristics.
        
        From GEPA paper: "The system is evaluated on multiple criteria
        to maintain diverse prompts on the Pareto frontier."
        
        Args:
            task_results: List of task results with success, steps, etc.
        
        Returns:
            Dictionary of objective -> score (all empirically computed)
        """
        # Import GEPA-compliant scoring
        from .scoring import compute_scores_from_task_results, GEPAObjective
        
        if not task_results:
            return {obj: 0.0 for obj in self.objectives}
        
        # Use GEPA-compliant scoring function
        gepa_scores = compute_scores_from_task_results(task_results)
        
        # Map GEPA objectives to our objective names for compatibility
        objective_mapping = {
            GEPAObjective.TASK_SUCCESS_RATE.value: "success_rate",
            GEPAObjective.STEP_EFFICIENCY.value: "step_efficiency",
            GEPAObjective.ADAPTATION_SPEED.value: "adaptation_speed",
            GEPAObjective.GENERALIZATION.value: "rule_generalization",
        }
        
        # Build scores dict with our objective names
        scores = {}
        for gepa_obj, our_obj in objective_mapping.items():
            if our_obj in self.objectives:
                scores[our_obj] = gepa_scores.get(gepa_obj, 0.0)
        
        # Fill any missing objectives with their GEPA equivalents
        for obj in self.objectives:
            if obj not in scores:
                # Try to find a matching GEPA objective
                for gepa_obj_name, score in gepa_scores.items():
                    if obj.lower() in gepa_obj_name.lower() or gepa_obj_name.lower() in obj.lower():
                        scores[obj] = score
                        break
                else:
                    # Use success rate as fallback for unmapped objectives
                    scores[obj] = gepa_scores.get(GEPAObjective.TASK_SUCCESS_RATE.value, 0.0)
        
        return scores
    
    def create_candidate(
        self,
        mutation: GEPAMutation,
        scores: Dict[str, float],
        parent_id: str,
    ) -> GEPAParetoCandidate:
        """
        Create a new Pareto candidate from a mutation.
        
        Args:
            mutation: The mutation result
            scores: Evaluation scores
            parent_id: Parent prompt ID
        
        Returns:
            New GEPAParetoCandidate
        """
        return GEPAParetoCandidate(
            prompt_id=self._generate_prompt_id(mutation.mutated_prompt),
            prompt_text=mutation.mutated_prompt,
            scores=scores,
            generation=self.generation + 1,
            parent_id=parent_id,
            mutation_history=mutation.lessons_incorporated,
        )
    
    def update_pareto_front(
        self,
        candidate: GEPAParetoCandidate,
    ) -> bool:
        """
        Pareto-based candidate selection.
        
        From paper: "To avoid the local optima that afflict greedy prompt
        updates, GEPA maintains a Pareto front: instead of evolving only
        the global best prompt, it stochastically explores the top-performing
        prompts for each problem instance"
        
        Args:
            candidate: The candidate to potentially add
        
        Returns:
            True if candidate was added to Pareto front
        """
        # Check if candidate is dominated by any existing
        for existing in self.pareto_front:
            if existing.dominates(candidate):
                return False  # Dominated, don't add
        
        # Remove any existing that candidate dominates
        original_size = len(self.pareto_front)
        self.pareto_front = [
            p for p in self.pareto_front
            if not candidate.dominates(p)
        ]
        dominated_count = original_size - len(self.pareto_front)
        self.stats["candidates_dominated"] += dominated_count
        
        # Add candidate to front
        self.pareto_front.append(candidate)
        self.stats["pareto_updates"] += 1
        
        # Prune if too large
        if len(self.pareto_front) > self.config.max_pareto_front_size:
            self._prune_pareto_front()
        
        return True
    
    def select_parent_from_pareto(self) -> Optional[GEPAParetoCandidate]:
        """
        Stochastically select a parent from the Pareto front.
        
        From paper: "it stochastically explores the top-performing prompts"
        
        Returns:
            Selected candidate or None if front is empty
        """
        if not self.pareto_front:
            return None
        
        # Weight by average score with some randomness
        weights = []
        for candidate in self.pareto_front:
            avg_score = candidate.get_average_score()
            weights.append(avg_score + random.random() * self.config.selection_noise)
        
        total = sum(weights)
        if total == 0:
            return random.choice(self.pareto_front)
        
        # Weighted random selection
        r = random.random() * total
        cumsum = 0
        for candidate, weight in zip(self.pareto_front, weights):
            cumsum += weight
            if r <= cumsum:
                return candidate
        
        return self.pareto_front[-1]
    
    def get_best_candidate(self) -> Optional[GEPAParetoCandidate]:
        """Get the candidate with highest average score."""
        if not self.pareto_front:
            return None
        
        return max(self.pareto_front, key=lambda c: c.get_average_score())
    
    def advance_generation(self):
        """Advance to next generation."""
        self.generation += 1
    
    def get_diversity_report(self) -> Dict[str, Any]:
        """Report on Pareto front diversity."""
        if not self.pareto_front:
            return {"status": "empty", "front_size": 0}
        
        # Score ranges per objective
        score_ranges = {}
        for obj in self.objectives:
            scores = [c.scores.get(obj, 0) for c in self.pareto_front]
            if scores:
                score_ranges[obj] = {
                    "min": min(scores),
                    "max": max(scores),
                    "avg": sum(scores) / len(scores),
                }
        
        return {
            "status": "active",
            "front_size": len(self.pareto_front),
            "generations_represented": sorted(set(c.generation for c in self.pareto_front)),
            "score_ranges": score_ranges,
            "stats": self.stats.copy(),
        }
    
    def _generate_prompt_id(self, prompt: str) -> str:
        """Generate unique ID for a prompt."""
        return hashlib.md5(prompt.encode()).hexdigest()[:12]
    
    def _prune_pareto_front(self):
        """Prune Pareto front to max size, keeping diverse candidates."""
        if len(self.pareto_front) <= self.config.max_pareto_front_size:
            return
        
        # Keep candidates with best scores for each objective
        keep = set()
        for obj in self.objectives:
            best = max(self.pareto_front, key=lambda c: c.scores.get(obj, 0))
            keep.add(best.prompt_id)
        
        # Fill remaining slots with highest average scores
        remaining = [c for c in self.pareto_front if c.prompt_id not in keep]
        remaining.sort(key=lambda c: c.get_average_score(), reverse=True)
        
        slots_left = self.config.max_pareto_front_size - len(keep)
        for c in remaining[:slots_left]:
            keep.add(c.prompt_id)
        
        self.pareto_front = [c for c in self.pareto_front if c.prompt_id in keep]


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def create_gepa_engine(
    llm_client: Callable,
    objectives: Optional[List[str]] = None,
    learned_rules_getter: Optional[Callable[[], List[str]]] = None,
) -> GEPAEvolutionEngine:
    """
    Factory function to create a GEPA Evolution Engine.
    
    Args:
        llm_client: Async LLM client
        objectives: Optional list of objectives to optimize
        learned_rules_getter: Optional callable to get learned rules
    
    Returns:
        Configured GEPAEvolutionEngine
    """
    config = GEPAConfig()
    if objectives:
        config.objectives = objectives
    
    return GEPAEvolutionEngine(
        llm_client=llm_client,
        config=config,
        learned_rules_getter=learned_rules_getter,
    )

