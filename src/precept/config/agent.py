"""
Agent Configuration for PRECEPT.

Configuration for PRECEPT agent behavior including retry limits, learning intervals,
memory limits, and feature flags.

Usage:
    from precept.config.agent import AgentConfig

    config = AgentConfig()
    config = AgentConfig(max_attempts=5, verbose_llm=True)
    config = AgentConfig.from_env()
"""

import os
from dataclasses import dataclass


@dataclass
class AgentConfig:
    """Configuration for PRECEPT agent behavior."""

    # =========================================================================
    # RETRY BUDGETS - SINGLE SOURCE OF TRUTH
    # =========================================================================
    #
    # max_retries is THE authoritative value for retry budget.
    # Both PRECEPT and baselines use this same value for fair comparison.
    #
    # PRECEPT: 1 initial attempt + max_retries pivots = max_retries + 1 total
    # Baselines: max_attempts iterations (where max_attempts = max_retries + 1)
    #
    # Example with max_retries=2:
    #   PRECEPT:   1 + 2 = 3 total attempts
    #   Baselines: 3 total attempts (max_attempts = 3)
    #
    # PRECEPT's ADVANTAGE (after learning):
    # - First time: Uses retries to find solution (blind exploration)
    # - After learning: Applies learned rule directly → 0 retries needed!
    # =========================================================================

    max_retries: int = 2  # SINGLE SOURCE OF TRUTH: Retry budget for all agents

    # Internal concurrency
    max_internal_workers: int = 3

    # Learning intervals
    consolidation_interval: int = 3
    compass_evolution_interval: int = 2
    failure_threshold: int = 2

    # Memory limits
    max_memories: int = 1000
    max_reflections_per_type: int = 20

    # =========================================================================
    # CONTEXT LIMITS FOR LLM REASONING
    # =========================================================================
    # These control how much context is passed to the LLM during reasoning.
    # Set to 0 for unlimited (include all available context).
    #
    # SMART FILTERING: When max_rules_chars > 0, we first try to filter rules
    # by relevance to the current task. Only if no relevant rules are found
    # do we fall back to truncation.
    # =========================================================================
    max_rules_chars: int = 0  # 0 = unlimited (include all rules)
    max_memories_chars: int = 2000  # Reasonable limit for memories
    enable_smart_rule_filtering: bool = True  # Filter by task relevance

    # Feature flags
    enable_llm_reasoning: bool = True
    force_llm_reasoning: bool = False
    enable_compass_optimization: bool = True
    verbose_llm: bool = False
    enable_dynamic_tier_resolution: bool = False

    # =========================================================================
    # HYBRID PARSING - Combines rule-based + LLM for complex tasks
    # =========================================================================
    # When enabled, if rule-based parsing is uncertain (confidence < 0.8),
    # PRECEPT uses OpenAI structured outputs to assist with parsing.
    # This helps with complex/ambiguous task descriptions.
    # =========================================================================
    enable_hybrid_parsing: bool = False  # Off by default (rule-based is faster)

    # =========================================================================
    # RETRY BEHAVIOR FLAGS - Configurable pruning aggressiveness
    # =========================================================================
    # These flags control how aggressively PRECEPT prunes the search space.
    # Default (all False): Aggressive pruning (fewer steps, may miss solutions)
    # All True: Exhaustive search (more steps, finds more solutions)
    #
    # Trade-off: Aggressive pruning saves API calls but may reduce success rate
    # =========================================================================

    # Option A: Disable "EXHAUSTED" early exit
    # When True: Ignore LLM "EXHAUSTED" signal, continue with random exploration
    # When False (default): Stop immediately when LLM says all options exhausted
    disable_exhausted_exit: bool = False

    # Option B: Only prune HARD constraints (not SOFT)
    # When True: SOFT errors (e.g., "server busy") can still be retried
    # When False (default): Both HARD and SOFT errors are pruned
    soft_constraints_retriable: bool = False

    # Option C: Enable random fallback when no remaining options
    # When True: Fall back to random choice from ALL options (like Full Reflexion)
    # When False (default): Stop when all options are pruned
    enable_random_fallback: bool = False

    # =========================================================================
    # COMPOSITIONAL GENERALIZATION - Atomic Constraint Stacking
    # =========================================================================
    # PRECEPT's key advantage: O(1) compositional adaptation.
    # Instead of learning O(2^N) composite rules, learn N atomic precepts once
    # and let the LLM synthesize composite solutions at runtime.
    #
    # Mechanism:
    # 1. Decompose complex conditions into atomic constraints via probing
    # 2. Retrieve individual precepts for each atomic constraint
    # 3. Stack constraints in the LLM context (Refine Layer)
    # 4. LLM synthesizes composite solution satisfying all constraints
    #
    # Example:
    #   Condition A+B → Retrieve Precept(A) + Precept(B) → LLM synthesizes X+Y
    #
    # DEFAULT: OFF - Enable explicitly for Experiment 6 only.
    # This ensures Experiments 1-5 behave exactly as before.
    # =========================================================================

    # Enable atomic constraint stacking for compositional generalization
    enable_compositional_generalization: bool = False  # OFF by default

    # Store rules at atomic (component) level, not just composite
    enable_atomic_precept_storage: bool = False  # OFF by default

    # Maximum number of atomic constraints to stack in LLM context
    max_stacked_constraints: int = 10

    # Minimum confidence threshold for atomic precepts
    atomic_precept_min_confidence: float = 0.3

    # Enable conflict detection between stacked constraints
    enable_constraint_conflict_detection: bool = False  # OFF by default

    @classmethod
    def from_env(cls) -> "AgentConfig":
        """Load agent config from environment variables."""
        max_retries_env = os.getenv("PRECEPT_MAX_RETRIES")
        if max_retries_env is not None:
            max_retries = int(max_retries_env)
        else:
            # Backward compatibility: PRECEPT_MAX_ATTEMPTS was total attempts,
            # while max_retries counts only retries after the first attempt.
            legacy_attempts = os.getenv("PRECEPT_MAX_ATTEMPTS")
            max_retries = max(0, int(legacy_attempts) - 1) if legacy_attempts else 2

        return cls(
            max_retries=max_retries,
            max_internal_workers=int(os.getenv("PRECEPT_INTERNAL_WORKERS", "3")),
            consolidation_interval=int(
                os.getenv("PRECEPT_CONSOLIDATION_INTERVAL", "3")
            ),
            enable_llm_reasoning=os.getenv("PRECEPT_LLM_REASONING", "true").lower()
            == "true",
            verbose_llm=os.getenv("PRECEPT_VERBOSE", "false").lower() == "true",
            enable_dynamic_tier_resolution=os.getenv(
                "PRECEPT_DYNAMIC_TIER_RESOLUTION", "false"
            ).lower()
            == "true",
            # Retry behavior flags
            disable_exhausted_exit=os.getenv(
                "PRECEPT_DISABLE_EXHAUSTED_EXIT", "false"
            ).lower()
            == "true",
            soft_constraints_retriable=os.getenv(
                "PRECEPT_SOFT_CONSTRAINTS_RETRIABLE", "false"
            ).lower()
            == "true",
            enable_random_fallback=os.getenv(
                "PRECEPT_ENABLE_RANDOM_FALLBACK", "false"
            ).lower()
            == "true",
            # Compositional generalization flags (OFF by default)
            enable_compositional_generalization=os.getenv(
                "PRECEPT_COMPOSITIONAL_GENERALIZATION", "false"
            ).lower()
            == "true",
            enable_atomic_precept_storage=os.getenv(
                "PRECEPT_ATOMIC_PRECEPT_STORAGE", "false"
            ).lower()
            == "true",
            enable_constraint_conflict_detection=os.getenv(
                "PRECEPT_CONSTRAINT_CONFLICT_DETECTION", "false"
            ).lower()
            == "true",
        )


__all__ = ["AgentConfig"]
