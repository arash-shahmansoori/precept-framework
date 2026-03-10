#!/usr/bin/env python3
"""
PRECEPT + AutoGen + MCP: Full Integration Demo (Multi-Domain)

PRECEPT = Planning Resilience via Experience, Context Engineering & Probing Trajectories

EVALUATION METHODOLOGY:
- TRAINING PHASE: PRECEPT + Full Reflexion learn patterns
- TEST PHASE: Compare PRECEPT vs 3 Baselines (4-Way Comparison)

BASELINES:
1. LLMBaselineAgent (Adapted ReAct): Error feedback only, no reflection
2. ReflexionBaselineAgent (Adapted Reflexion): Within-task reflection only
3. FullReflexionBaselineAgent (Full Reflexion): Cross-episode memory (same task type)

Usage:
    # Basic usage (defaults: --train 6 --test 4, static knowledge enabled)
    uv run examples/precept_autogen_mcp_full.py --domain logistics

    # Custom train/test split with seed for reproducibility
    uv run examples/precept_autogen_mcp_full.py --domain logistics --train 8 --test 4 --seed 42

    # Concurrent training ("Tesla Fleet" mode - 4x faster)
    uv run examples/precept_autogen_mcp_full.py --domain logistics --concurrent-training --training-workers 4

    # Concurrent testing (faster test phase, control parallelism with --workers)
    uv run examples/precept_autogen_mcp_full.py --domain logistics --concurrent --workers 4

    # Full concurrent mode (both training and testing parallelized)
    uv run examples/precept_autogen_mcp_full.py -d logistics -ct -tw 4 -c -w 4 --seed 42

    # ═══════════════════════════════════════════════════════════════════════════
    # STATIC KNOWLEDGE CONFIGURATION (Simple on/off)
    # ═══════════════════════════════════════════════════════════════════════════

    # Test WITH static knowledge (default) - ALL agents get static facts
    # PRECEPT's advantage: cutting-edge CONFLICT RESOLUTION
    uv run examples/precept_autogen_mcp_full.py -d logistics --static-knowledge

    # Test WITHOUT static knowledge - pure dynamic learning comparison
    uv run examples/precept_autogen_mcp_full.py -d logistics --no-static-knowledge

    # All domains
    uv run examples/precept_autogen_mcp_full.py --domain coding --train 8 --test 4
    uv run examples/precept_autogen_mcp_full.py --domain devops --train 6 --test 4
    uv run examples/precept_autogen_mcp_full.py --domain finance --train 6 --test 4

    # List available domains
    uv run examples/precept_autogen_mcp_full.py --list

╔═══════════════════════════════════════════════════════════════════════════════════╗
║                          PRECEPT STAGES IMPLEMENTED                               ║
╠═══════════════════════════════════════════════════════════════════════════════════╣
║                                                                                   ║
║  1. ✅ DOMAIN STRATEGY PATTERN (domain_strategies/)                               ║
║       • DomainStrategy base class for learning agents                             ║
║       • BaselineDomainStrategy for fair comparison                                ║
║       • 6 domains: Logistics, Coding, DevOps, Finance, Booking, Integration       ║
║                                                                                   ║
║  2. ✅ DYNAMIC RULE PARSER (rule_parser.py)                                       ║
║       • DynamicRuleParser - NO hardcoded knowledge                                ║
║       • Learns from rule text dynamically                                         ║
║                                                                                   ║
║  3. ✅ COMPASS ADVANTAGES (complexity_analyzer.py)                                ║
║       • PRECEPTComplexityAnalyzer (ML-based complexity detection)                 ║
║       • SmartRolloutStrategy (adaptive rollout allocation)                        ║
║       • MultiStrategyCoordinator (multi-strategy coordination)                    ║
║                                                                                   ║
║  4. ✅ AUTOGEN PRECEPT AGENT (precept_agent.py)                                   ║
║       • Generic agent with Strategy Pattern                                       ║
║       • Works with ANY domain by injecting DomainStrategy                         ║
║                                                                                   ║
║  5. ✅ BASELINE AGENTS (baseline_agents.py)                                       ║
║       • 3 baselines for comprehensive comparison                                  ║
║       • LLMBaselineAgent, ReflexionBaselineAgent, FullReflexionBaselineAgent      ║
║                                                                                   ║
║  6. ✅ SCENARIO GENERATORS (scenario_generators/)                                 ║
║       • All 6 domains supported                                                   ║
║       • Integrates with black_swan_gen.py                                         ║
║                                                                                   ║
╚═══════════════════════════════════════════════════════════════════════════════════╝
"""

import argparse
import asyncio
import json
import os
import sys
import time as time_module
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

# Suppress asyncio shutdown warnings from MCP library
# These are harmless errors that occur during event loop cleanup
warnings.filterwarnings(
    "ignore",
    message=".*cancel scope.*",
    category=RuntimeWarning,
)
warnings.filterwarnings(
    "ignore",
    message=".*Event loop is closed.*",
    category=RuntimeWarning,
)

# Add examples to path for config imports
examples_dir = Path(__file__).parent
sys.path.insert(0, str(examples_dir))

# Add src to path for precept imports
project_root = examples_dir.parent
sys.path.insert(0, str(project_root / "src"))

from dotenv import load_dotenv  # noqa: E402

load_dotenv()

# ═══════════════════════════════════════════════════════════════════════════════
# EXPERIMENT CONFIGURATION - From examples/config/
# ═══════════════════════════════════════════════════════════════════════════════

from config import (  # noqa: E402, I001
    DATA_DIR,
    DOMAIN_ICONS,
    SCENARIO_GENERATORS,
    SERVER_SCRIPT,
    STATS_PATH,
    ExperimentDisplay,
    ProgressTracker,
)

# ═══════════════════════════════════════════════════════════════════════════════
# MODULAR PRECEPT IMPORTS - All from src/precept/
# ═══════════════════════════════════════════════════════════════════════════════

from precept import (  # noqa: E402
    ExecutionTracer,
    ExpeL_BaselineAgent,
    FullReflexionBaselineAgent,
    # LLMBaselineAgent,  # Commented out - unfair comparison (no training)
    PRECEPTAgent,
    # ReflexionBaselineAgent,  # Commented out - unfair comparison (no training)
    check_api_availability,
    clear_expel_insights,
    get_baseline_strategy,
    get_domain_strategy,
    list_available_domains,
)
from precept.config import BaselineConfig, PreceptConfig, get_logger, setup_logging  # noqa: E402

# Import async client reset for clean state between experiments
try:
    from models.openai_api import reset_async_client
except ImportError:

    def reset_async_client():
        """Fallback if not available."""
        pass

# ═══════════════════════════════════════════════════════════════════════════════
# LOGGING SETUP
# ═══════════════════════════════════════════════════════════════════════════════

# Initialize logging, display, and progress tracking
logger = get_logger("precept.experiment")
display = ExperimentDisplay(logger)
progress_tracker = ProgressTracker()


def load_stats() -> Dict:
    """Load cumulative stats."""
    if STATS_PATH.exists():
        with open(STATS_PATH, "r") as f:
            return json.load(f)
    return {
        "total_runs": 0,
        "total_tasks": 0,
        "precept_successes": 0,
        "baseline_successes": 0,
    }


def save_stats(stats: Dict):
    """Save cumulative stats."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with open(STATS_PATH, "w") as f:
        json.dump(stats, f, indent=2)


def _get_domain_test_generator(domain: str, num_train: int, num_test: int):
    """
    Get the domain-specific test scenario generator.

    This function returns the appropriate generator class for each domain,
    initialized with the given train/test parameters.

    Args:
        domain: Domain name (logistics, booking, coding, devops, finance, integration)
        num_train: Number of training samples
        num_test: Number of test samples

    Returns:
        Domain-specific scenario generator instance, or None if not available
    """
    train_ratio = (
        num_train / (num_train + num_test) if (num_train + num_test) > 0 else 0.6
    )
    num_samples = num_train + num_test

    try:
        if domain == "logistics":
            from precept.scenario_generators.logistics import LogisticsScenarioGenerator

            return LogisticsScenarioGenerator(
                num_samples=num_samples, train_ratio=train_ratio
            )
        elif domain == "booking":
            from precept.scenario_generators.booking import BookingScenarioGenerator

            return BookingScenarioGenerator(
                num_samples=num_samples, train_ratio=train_ratio
            )
        elif domain == "coding":
            from precept.scenario_generators.coding import CodingScenarioGenerator

            return CodingScenarioGenerator(
                num_samples=num_samples, train_ratio=train_ratio
            )
        elif domain == "devops":
            from precept.scenario_generators.devops import DevOpsScenarioGenerator

            return DevOpsScenarioGenerator(
                num_samples=num_samples, train_ratio=train_ratio
            )
        elif domain == "finance":
            from precept.scenario_generators.finance import FinanceScenarioGenerator

            return FinanceScenarioGenerator(
                num_samples=num_samples, train_ratio=train_ratio
            )
        elif domain == "integration":
            from precept.scenario_generators.integration import (
                IntegrationScenarioGenerator,
            )

            return IntegrationScenarioGenerator(
                num_samples=num_samples, train_ratio=train_ratio
            )
        else:
            logger.warning(f"Unknown domain: {domain}")
            return None
    except ImportError as e:
        logger.warning(f"Could not import generator for domain {domain}: {e}")
        return None


def save_experiment_results(
    domain: str,
    train_scenarios: List[Dict],
    test_scenarios: List[Dict],
    llm_metrics: "AgentMetrics",
    reflexion_metrics: "AgentMetrics",
    full_reflexion_metrics: "AgentMetrics",
    expel_metrics: "AgentMetrics",  # ExpeL baseline (Zhao et al., 2023)
    precept_metrics: "AgentMetrics",
    precept_stats: Dict,
    pruning_stats: Dict,
    train_elapsed: float,
    test_elapsed: float,
    concurrent_training: bool,
    concurrent_testing: bool,
    seed: int = None,
    enable_static_knowledge: bool = True,
    conflict_stats: Dict = None,
    max_retries: int = None,
    test_mode: str = "matched",
    dual_mode_results: Dict = None,  # Contains metrics for both modes when test_mode="both"
) -> Path:
    """
    Save comprehensive experiment results to a timestamped JSON file.

    Returns:
        Path to the saved results file
    """
    from datetime import datetime

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = DATA_DIR / f"experiment_results_{domain}_{timestamp}.json"

    # Determine static knowledge mode string
    static_kb_mode = "enabled" if enable_static_knowledge else "disabled"

    # Calculate effective max_retries (default is 2)
    effective_max_retries = max_retries if max_retries is not None else 2

    results = {
        "metadata": {
            "domain": domain,
            "timestamp": timestamp,
            "seed": seed,
            "train_tasks": len(train_scenarios),
            "test_tasks": len(test_scenarios),
            "max_retries": effective_max_retries,
            "max_attempts": effective_max_retries + 1,  # Total attempts = 1 + retries
            "concurrent_training": concurrent_training,
            "concurrent_testing": concurrent_testing,
            "train_elapsed_seconds": round(train_elapsed, 2),
            "test_elapsed_seconds": round(test_elapsed, 2),
            "static_knowledge_enabled": enable_static_knowledge,
            "static_knowledge_mode": static_kb_mode,
            "test_mode": test_mode,
        },
        "agents": {
            "llm_baseline": {
                "name": llm_metrics.name,
                "success_rate": round(llm_metrics.success_rate, 4),
                "total_successes": llm_metrics.total_successes,
                "total_tasks": llm_metrics.total_tasks,
                "avg_steps": round(llm_metrics.avg_steps, 2),
                "llm_calls": llm_metrics.llm_calls,
                "llm_accuracy": round(llm_metrics.llm_accuracy, 4),
                # New metrics
                "first_try_success_rate": round(llm_metrics.first_try_success_rate, 4),
                "first_try_successes": llm_metrics.first_try_successes,
                "learning_efficiency": round(llm_metrics.learning_efficiency, 4),
                "cold_start_success_rate": round(
                    llm_metrics.cold_start_success_rate, 4
                ),
                "cold_start_successes": llm_metrics.cold_start_successes,
                "cold_start_total": llm_metrics.cold_start_total,
            },
            "reflexion": {
                "name": reflexion_metrics.name,
                "success_rate": round(reflexion_metrics.success_rate, 4),
                "total_successes": reflexion_metrics.total_successes,
                "total_tasks": reflexion_metrics.total_tasks,
                "avg_steps": round(reflexion_metrics.avg_steps, 2),
                "llm_calls": reflexion_metrics.llm_calls,
                "llm_accuracy": round(reflexion_metrics.llm_accuracy, 4),
                "reflections_generated": reflexion_metrics.reflections_generated,
                # New metrics
                "first_try_success_rate": round(
                    reflexion_metrics.first_try_success_rate, 4
                ),
                "first_try_successes": reflexion_metrics.first_try_successes,
                "learning_efficiency": round(reflexion_metrics.learning_efficiency, 4),
                "cold_start_success_rate": round(
                    reflexion_metrics.cold_start_success_rate, 4
                ),
                "cold_start_successes": reflexion_metrics.cold_start_successes,
                "cold_start_total": reflexion_metrics.cold_start_total,
            },
            "full_reflexion": {
                "name": full_reflexion_metrics.name,
                "success_rate": round(full_reflexion_metrics.success_rate, 4),
                "total_successes": full_reflexion_metrics.total_successes,
                "total_tasks": full_reflexion_metrics.total_tasks,
                "avg_steps": round(full_reflexion_metrics.avg_steps, 2),
                "llm_calls": full_reflexion_metrics.llm_calls,
                "llm_accuracy": round(full_reflexion_metrics.llm_accuracy, 4),
                "reflections_generated": full_reflexion_metrics.reflections_generated,
                "reflections_reused": full_reflexion_metrics.reflections_reused,
                "accumulated_memories": full_reflexion_metrics.accumulated_memories,
                # P₁ metrics (overall)
                "first_try_success_rate": round(
                    full_reflexion_metrics.first_try_success_rate, 4
                ),
                "first_try_successes": full_reflexion_metrics.first_try_successes,
                # P₁ metrics by category (for fair comparison)
                "p1_with_rules": round(full_reflexion_metrics.p1_with_rules, 4),
                "p1_with_rules_total": full_reflexion_metrics.p1_with_rules_total,
                "p1_inherent": round(full_reflexion_metrics.p1_inherent, 4),
                "p1_inherent_total": full_reflexion_metrics.p1_inherent_total,
                "p1_other": round(full_reflexion_metrics.p1_other, 4),
                "p1_other_total": full_reflexion_metrics.p1_other_total,
                # Other metrics
                "learning_efficiency": round(
                    full_reflexion_metrics.learning_efficiency, 4
                ),
                "cold_start_success_rate": round(
                    full_reflexion_metrics.cold_start_success_rate, 4
                ),
                "cold_start_successes": full_reflexion_metrics.cold_start_successes,
                "cold_start_total": full_reflexion_metrics.cold_start_total,
            },
            "expel": {
                "name": expel_metrics.name,
                "success_rate": round(expel_metrics.success_rate, 4),
                "total_successes": expel_metrics.total_successes,
                "total_tasks": expel_metrics.total_tasks,
                "avg_steps": round(expel_metrics.avg_steps, 2),
                "llm_calls": expel_metrics.llm_calls,
                "llm_accuracy": round(expel_metrics.llm_accuracy, 4),
                "insights_extracted": getattr(expel_metrics, "insights_extracted", 0),
                "insights_retrieved": getattr(expel_metrics, "insights_retrieved", 0),
                # P₁ metrics (overall)
                "first_try_success_rate": round(
                    expel_metrics.first_try_success_rate, 4
                ),
                "first_try_successes": expel_metrics.first_try_successes,
                # P₁ metrics by category (for fair comparison)
                "p1_with_rules": round(expel_metrics.p1_with_rules, 4),
                "p1_with_rules_total": expel_metrics.p1_with_rules_total,
                "p1_inherent": round(expel_metrics.p1_inherent, 4),
                "p1_inherent_total": expel_metrics.p1_inherent_total,
                "p1_other": round(expel_metrics.p1_other, 4),
                "p1_other_total": expel_metrics.p1_other_total,
                # Other metrics
                "learning_efficiency": round(expel_metrics.learning_efficiency, 4),
                "cold_start_success_rate": round(
                    expel_metrics.cold_start_success_rate, 4
                ),
                "cold_start_successes": expel_metrics.cold_start_successes,
                "cold_start_total": expel_metrics.cold_start_total,
            },
            "precept": {
                "name": precept_metrics.name,
                "success_rate": round(precept_metrics.success_rate, 4),
                "total_successes": precept_metrics.total_successes,
                "total_tasks": precept_metrics.total_tasks,
                "avg_steps": round(precept_metrics.avg_steps, 2),
                "rules_learned": precept_metrics.rules_learned,
                "llm_reasoning_calls": precept_stats.get("total_calls", 0),
                "llm_reasoning_successes": precept_stats.get("successes", 0),
                "has_compass_evolution": precept_metrics.has_compass_evolution,
                # P₁ metrics (overall)
                "first_try_success_rate": round(
                    precept_metrics.first_try_success_rate, 4
                ),
                "first_try_successes": precept_metrics.first_try_successes,
                # P₁ metrics (categorized by training outcome)
                "p1_with_rules": round(precept_metrics.p1_with_rules, 4),
                "p1_with_rules_successes": precept_metrics.p1_with_rules_successes,
                "p1_with_rules_total": precept_metrics.p1_with_rules_total,
                "p1_inherent": round(precept_metrics.p1_inherent, 4),
                "p1_inherent_successes": precept_metrics.p1_inherent_successes,
                "p1_inherent_total": precept_metrics.p1_inherent_total,
                "p1_other": round(precept_metrics.p1_other, 4),
                "p1_other_successes": precept_metrics.p1_other_successes,
                "p1_other_total": precept_metrics.p1_other_total,
                # Legacy (backward compatibility)
                "p1_without_rules": round(precept_metrics.p1_without_rules, 4),
                "p1_without_rules_successes": precept_metrics.p1_without_rules_successes,
                "p1_without_rules_total": precept_metrics.p1_without_rules_total,
                # Other metrics
                "learning_efficiency": round(precept_metrics.learning_efficiency, 4),
                "cold_start_success_rate": round(
                    precept_metrics.cold_start_success_rate, 4
                ),
                "cold_start_successes": precept_metrics.cold_start_successes,
                "cold_start_total": precept_metrics.cold_start_total,
            },
        },
        "precept_advantages": {
            # Success rate advantages
            "vs_llm_baseline": round(
                precept_metrics.success_rate - llm_metrics.success_rate, 4
            ),
            "vs_reflexion": round(
                precept_metrics.success_rate - reflexion_metrics.success_rate, 4
            ),
            "vs_full_reflexion": round(
                precept_metrics.success_rate - full_reflexion_metrics.success_rate, 4
            ),
            # Steps efficiency
            "steps_saved_vs_llm": round(
                llm_metrics.avg_steps - precept_metrics.avg_steps, 2
            ),
            "steps_saved_vs_reflexion": round(
                reflexion_metrics.avg_steps - precept_metrics.avg_steps, 2
            ),
            "steps_saved_vs_full_reflexion": round(
                full_reflexion_metrics.avg_steps - precept_metrics.avg_steps, 2
            ),
            # NEW: First-try success advantages
            "first_try_advantage_vs_llm": round(
                precept_metrics.first_try_success_rate
                - llm_metrics.first_try_success_rate,
                4,
            ),
            "first_try_advantage_vs_reflexion": round(
                precept_metrics.first_try_success_rate
                - reflexion_metrics.first_try_success_rate,
                4,
            ),
            "first_try_advantage_vs_full_reflexion": round(
                precept_metrics.first_try_success_rate
                - full_reflexion_metrics.first_try_success_rate,
                4,
            ),
            # NEW: Learning efficiency advantages
            "learning_efficiency_advantage_vs_llm": round(
                precept_metrics.learning_efficiency - llm_metrics.learning_efficiency, 4
            ),
            "learning_efficiency_advantage_vs_full_reflexion": round(
                precept_metrics.learning_efficiency
                - full_reflexion_metrics.learning_efficiency,
                4,
            ),
            # NEW: Cold-start advantages
            "cold_start_advantage_vs_llm": round(
                precept_metrics.cold_start_success_rate
                - llm_metrics.cold_start_success_rate,
                4,
            ),
            "cold_start_advantage_vs_full_reflexion": round(
                precept_metrics.cold_start_success_rate
                - full_reflexion_metrics.cold_start_success_rate,
                4,
            ),
        },
        "pruning_stats": {
            "total_constraints": pruning_stats.get("total_constraints", 0),
            "hard_constraints": pruning_stats.get("hard_constraints", 0),
            "soft_constraints": pruning_stats.get("soft_constraints", 0),
            "dumb_retries_prevented": pruning_stats.get("dumb_retries_prevented", 0),
            "pruning_efficiency": round(
                pruning_stats.get("dumb_retries_prevented", 0)
                / max(1, pruning_stats.get("total_constraints", 1)),
                4,
            ),
        },
        "conflict_resolution": {
            "enabled": enable_static_knowledge,
            "mode": static_kb_mode,
            "stats": conflict_stats or {},
            "precept_capabilities": [
                "Bayesian uncertainty quantification",
                "Evidence-based prioritization",
                "Anomaly detection",
                "Dynamic reliability learning",
                "Thompson sampling exploration",
                "Multi-strategy resolution",
            ]
            if enable_static_knowledge
            else [],
            "baseline_capabilities": [
                "Static knowledge access (read-only)",
                "No conflict detection",
                "No resolution strategy",
            ]
            if enable_static_knowledge
            else [],
        },
        "winner": (
            "precept"
            if precept_metrics.success_rate
            >= max(
                llm_metrics.success_rate,
                reflexion_metrics.success_rate,
                full_reflexion_metrics.success_rate,
            )
            else "baseline"
        ),
    }

    # Add dual-mode results if running in "both" mode
    if dual_mode_results:
        results["dual_mode_results"] = dual_mode_results

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)

    logger.info(f"📁 Experiment results saved to: {results_file}")
    return results_file


async def graceful_disconnect(agents: List, timeout: float = 2.0):
    """
    Gracefully disconnect all agents with timeout and error suppression.

    This is critical for parallel execution to prevent "Event loop is closed" errors.
    When running multiple experiments in parallel, proper cleanup prevents:
    - Stale HTTP connections bound to wrong event loops
    - asyncio.locks.Event objects bound to closed loops
    - MCP server process zombies

    Args:
        agents: List of agents to disconnect
        timeout: Maximum time to wait for each disconnect
    """
    for agent in agents:
        try:
            # Use wait_for with timeout to prevent hanging
            await asyncio.wait_for(agent.disconnect(), timeout=timeout)
        except asyncio.TimeoutError:
            logger.debug(f"Disconnect timeout for {agent.__class__.__name__}")
        except (RuntimeError, asyncio.CancelledError, GeneratorExit):
            # These are expected during asyncio shutdown - suppress silently
            pass
        except Exception:
            # Suppress all cleanup errors - we're shutting down anyway
            pass


# =============================================================================
# METRICS DATACLASS
# =============================================================================


@dataclass
class AgentMetrics:
    """Comprehensive metrics for an agent."""

    name: str
    success_rate: float
    total_successes: int
    total_tasks: int
    avg_steps: float
    llm_calls: int
    llm_calls_per_task: float
    llm_accuracy: float
    # Learning-specific
    rules_learned: int = 0
    reflections_generated: int = 0
    reflections_reused: int = 0
    accumulated_memories: int = 0
    # Special metrics
    has_cross_task_learning: bool = False
    has_cross_episode_memory: bool = False
    has_within_task_reflection: bool = False
    has_deterministic_pruning: bool = False
    has_compass_evolution: bool = False
    # =========================================================================
    # NEW METRICS: First-Try, Learning Efficiency, Cold-Start
    # =========================================================================
    # First-Try Success (P₁ₐₗₗ): Overall first-try success on ALL scenarios
    first_try_success_rate: float = 0.0
    first_try_successes: int = 0
    # =========================================================================
    # P₁ METRICS (SIMPLIFIED)
    # =========================================================================
    # With fair filtering, the main P₁ = first_try_success_rate on filtered scenarios
    # Legacy fields kept for backward compatibility but set to 0
    p1_with_rules: float = 0.0
    p1_with_rules_successes: int = 0
    p1_with_rules_total: int = 0
    p1_inherent: float = 0.0
    p1_inherent_successes: int = 0
    p1_inherent_total: int = 0
    p1_other: float = 0.0
    p1_other_successes: int = 0
    p1_other_total: int = 0
    p1_without_rules: float = 0.0
    p1_without_rules_successes: int = 0
    p1_without_rules_total: int = 0
    # Learning Efficiency: Success rate normalized by training samples
    # Higher = more efficient learner (achieves more with less training)
    learning_efficiency: float = 0.0
    # Cold-Start Success: Success on FIRST encounter of each task type
    # Tests generalization without prior experience of that specific type
    cold_start_success_rate: float = 0.0
    cold_start_successes: int = 0
    cold_start_total: int = 0


def calculate_metrics(
    results: List[Dict[str, Any]],
    agent_stats: Dict[str, Any],
    agent_name: str,
    scenarios: List[Dict[str, Any]] = None,
    num_train: int = 10,
) -> AgentMetrics:
    """
    Calculate comprehensive metrics from results.

    Args:
        results: List of per-task results with 'success', 'steps', and 'attempts'
        agent_stats: Agent-specific statistics
        agent_name: Name of the agent
        scenarios: List of test scenarios (for cold-start calculation)
        num_train: Number of training samples (for learning efficiency)

    Returns:
        AgentMetrics with all computed metrics
    """
    successes = sum(1 for r in results if r["success"])
    total = len(results)
    avg_steps = sum(r["steps"] for r in results) / total if total > 0 else 0
    success_rate = successes / total if total > 0 else 0

    # =========================================================================
    # NEW METRIC 1: First-Try Success Rate (P₁)
    # CONSISTENT DEFINITION: Success WITHOUT ERROR RECOVERY
    # This is semantically correct across ALL methods:
    #   - PRECEPT: first_try = True if action succeeded without _handle_error_recovery
    #   - ExpeL: first_try = True if success on first attempt (attempts == 1)
    #   - Full Reflexion: first_try = True if success on first attempt (attempts == 1)
    # =========================================================================
    first_try_successes = sum(1 for r in results if r.get("first_try", False))
    first_try_success_rate = first_try_successes / total if total > 0 else 0

    # =========================================================================
    # P₁ METRICS (SIMPLIFIED)
    # =========================================================================
    # With fair filtering, we only test on scenarios where training succeeded.
    # The first_try_success_rate IS the main P₁ metric for all scenarios.
    # No complex category breakdown needed - all test scenarios are "fair".
    # =========================================================================

    # For backward compatibility, set category metrics to 0
    p1_with_rules = 0.0
    p1_with_rules_successes = 0
    p1_with_rules_total = 0
    p1_inherent = 0.0
    p1_inherent_successes = 0
    p1_inherent_total = 0
    p1_other = 0.0
    p1_other_successes = 0
    p1_other_total = 0
    p1_without_rules = 0.0
    p1_without_rules_successes = 0
    p1_without_rules_total = 0

    # =========================================================================
    # NEW METRIC 2: Learning Efficiency
    # Success rate normalized by training samples (baseline: 10 samples)
    # Higher = more efficient (achieves high success with less training)
    # =========================================================================
    learning_efficiency = success_rate * (10.0 / num_train) if num_train > 0 else 0

    # =========================================================================
    # NEW METRIC 3: Cold-Start Success Rate
    # Success on FIRST encounter of each task type
    # =========================================================================
    cold_start_successes = 0
    cold_start_total = 0

    if scenarios and len(scenarios) == len(results):
        # Track first encounter of each task type
        seen_types = set()
        for scenario, result in zip(scenarios, results):
            # Get task type identifier
            task_type = scenario.get("tests_learning") or scenario.get(
                "black_swan_type", "unknown"
            )

            if task_type not in seen_types:
                # This is the FIRST encounter of this task type
                seen_types.add(task_type)
                cold_start_total += 1
                if result["success"]:
                    cold_start_successes += 1

    cold_start_success_rate = (
        cold_start_successes / cold_start_total if cold_start_total > 0 else 0
    )

    return AgentMetrics(
        name=agent_name,
        success_rate=success_rate,
        total_successes=successes,
        total_tasks=total,
        avg_steps=avg_steps,
        llm_calls=agent_stats.get("llm_calls", 0),
        llm_calls_per_task=agent_stats.get("llm_calls_per_task", 0),
        llm_accuracy=agent_stats.get("llm_accuracy", 0),
        rules_learned=agent_stats.get("rules_learned", 0),
        reflections_generated=agent_stats.get("reflections_generated", 0),
        reflections_reused=agent_stats.get("reflections_reused", 0),
        accumulated_memories=sum(agent_stats.get("memory_stats", {}).values())
        if "memory_stats" in agent_stats
        else 0,
        # P₁ metrics (overall)
        first_try_success_rate=first_try_success_rate,
        first_try_successes=first_try_successes,
        # P₁ metrics (categorized by training outcome)
        p1_with_rules=p1_with_rules,
        p1_with_rules_successes=p1_with_rules_successes,
        p1_with_rules_total=p1_with_rules_total,
        p1_inherent=p1_inherent,
        p1_inherent_successes=p1_inherent_successes,
        p1_inherent_total=p1_inherent_total,
        p1_other=p1_other,
        p1_other_successes=p1_other_successes,
        p1_other_total=p1_other_total,
        # Legacy (backward compatibility)
        p1_without_rules=p1_without_rules,
        p1_without_rules_successes=p1_without_rules_successes,
        p1_without_rules_total=p1_without_rules_total,
        # Other metrics
        learning_efficiency=learning_efficiency,
        cold_start_success_rate=cold_start_success_rate,
        cold_start_successes=cold_start_successes,
        cold_start_total=cold_start_total,
    )


# =============================================================================
# HELPER: Run Tests on Scenario Set (for dual-mode support)
# =============================================================================


async def run_test_phase(
    scenarios: List[Dict],
    precept_agent,
    expel_baseline,
    full_reflexion,
    tracer,
    mode_name: str = "matched",
) -> tuple:
    """
    Run all agents on a set of test scenarios and return results.

    Args:
        scenarios: List of test scenarios to run
        precept_agent: PRECEPT agent instance
        expel_baseline: ExpeL agent instance
        full_reflexion: Full Reflexion agent instance
        tracer: Experiment tracer for logging
        mode_name: Name of the test mode (for logging)

    Returns:
        Tuple of (precept_results, expel_results, fr_results, elapsed_time)
    """
    import time as time_module

    precept_results = []
    expel_results = []
    fr_results = []

    test_start = time_module.time()

    logger.info(f"\n🧪 [{mode_name.upper()}] Testing on {len(scenarios)} scenarios...")

    last_compass_state: Dict[str, Any] = {}

    for i, scenario in enumerate(scenarios, 1):
        task = scenario["task"]
        tests = scenario.get("tests_learning", "pattern")

        # ═══════════════════════════════════════════════════════════════════
        # EXTRACT MULTI-CONDITION METADATA from scenario
        # This passes condition_key to agents without it being in task string
        # ═══════════════════════════════════════════════════════════════════
        multi_condition = scenario.get("multi_condition", {})
        metadata = {
            "condition_key": multi_condition.get("condition_key")
            or scenario.get("condition_key"),
            "conditions": multi_condition.get("conditions", []),
            "expected_solution": multi_condition.get("expected_solution"),
        }

        # Start tracing
        precept_trace = tracer.start_task(i, task, f"testing_{mode_name}", "precept")
        expel_trace = tracer.start_task(i, task, f"testing_{mode_name}", "expel")
        fr_trace = tracer.start_task(i, task, f"testing_{mode_name}", "full_reflexion")

        # Run all agents with metadata (condition_key for multi-condition enforcement)
        # NOTE: ExpeL now uses training=True during testing for fair comparison
        # This allows ExpeL to extract insights during testing (online learning)
        precept_result = await precept_agent.run_task(task, metadata=metadata)
        expel_result = await expel_baseline.run_task(
            task,
            training=True,
            metadata=metadata,  # Enable learning during testing
        )
        fr_result = await full_reflexion.run_task(task, metadata=metadata)

        # Record traces
        precept_trace.add_event(
            "task_complete",
            {
                "success": precept_result.get("success", False),
                "task_steps": precept_result.get("task_steps", 0),
                "overhead_steps": precept_result.get("overhead_steps", 0),
                "strategy": precept_result.get("strategy", ""),
                "response": str(precept_result.get("response", ""))[:200],
                "tests_learning": tests,
                "test_mode": mode_name,
                "condition_key": metadata.get("condition_key"),
                "conditions": metadata.get("conditions", []),
                "expected_solution": metadata.get("expected_solution"),
                "rule_learned": precept_result.get("rule_learned", False),
                "learned_rule_key": precept_result.get("learned_rule_key", ""),
                "learned_solution": precept_result.get("learned_solution", ""),
                "learned_via": precept_result.get("learned_via", ""),
            },
        )
        if precept_result.get("rule_learned"):
            precept_trace.add_event(
                "rule_learned",
                {
                    "condition_key": precept_result.get("learned_rule_key")
                    or metadata.get("condition_key"),
                    "solution": precept_result.get("learned_solution", ""),
                    "via": precept_result.get("learned_via", ""),
                    "phase": f"testing_{mode_name}",
                },
            )
        await _maybe_log_compass_compilation_status(
            precept_agent, precept_trace, last_compass_state
        )
        tracer.end_task(precept_trace, precept_result)
        precept_results.append(precept_result)

        expel_trace.add_event(
            "task_complete",
            {
                "success": expel_result.get("success", False),
                "total_insights": expel_result.get("total_insights", 0),
                "insights_retrieved": expel_result.get("insights_retrieved", 0),
                "response": str(expel_result.get("response", ""))[:200],
                "tests_learning": tests,
                "test_mode": mode_name,
                "condition_key": metadata.get("condition_key"),
                "conditions": metadata.get("conditions", []),
                "expected_solution": metadata.get("expected_solution"),
            },
        )
        tracer.end_task(expel_trace, expel_result)
        expel_results.append(expel_result)

        fr_trace.add_event(
            "task_complete",
            {
                "success": fr_result.get("success", False),
                "accumulated_reflections": fr_result.get("accumulated_reflections", 0),
                "response": str(fr_result.get("response", ""))[:200],
                "tests_learning": tests,
                "test_mode": mode_name,
                "condition_key": metadata.get("condition_key"),
                "conditions": metadata.get("conditions", []),
                "expected_solution": metadata.get("expected_solution"),
            },
        )
        tracer.end_task(fr_trace, fr_result)
        fr_results.append(fr_result)

        # ═══════════════════════════════════════════════════════════════════════
        # DETAILED PER-TASK LOGGING FOR LEARNING CURVE ANALYSIS
        # This enables tracking metrics by encounter number for each condition_key
        # ═══════════════════════════════════════════════════════════════════════
        condition_key = scenario.get("multi_condition", {}).get("condition_key", "")
        logger.info(
            f"  📊 [{mode_name.upper()} Test {i}/{len(scenarios)}] "
            f"key={condition_key[:30]}... | "
            f"PRECEPT: {'✓' if precept_result.get('success') else '✗'} "
            f"(P₁={'Y' if precept_result.get('first_try') else 'N'}, steps={precept_result.get('steps', 0)}) | "
            f"ExpeL: {'✓' if expel_result.get('success') else '✗'} "
            f"(P₁={'Y' if expel_result.get('first_try') else 'N'}) | "
            f"FullRef: {'✓' if fr_result.get('success') else '✗'} "
            f"(P₁={'Y' if fr_result.get('first_try') else 'N'})"
        )

    elapsed = time_module.time() - test_start
    logger.info(f"  ⏱️  [{mode_name.upper()}] Testing completed in {elapsed:.1f}s")

    return precept_results, expel_results, fr_results, elapsed


async def _maybe_log_compass_compilation_status(
    precept_agent, precept_trace, last_state: Dict[str, Any]
) -> None:
    """Log COMPASS compilation status when it changes (for trace visibility)."""
    if not precept_agent or not getattr(precept_agent, "mcp_client", None):
        return
    if not precept_trace or not getattr(precept_trace, "add_event", None):
        return

    try:
        raw = await precept_agent.mcp_client.call_tool(
            "get_compass_compilation_status", {}
        )
        status = json.loads(raw) if isinstance(raw, str) else raw
    except Exception:
        return

    state = status.get("state", {}) if isinstance(status, dict) else {}
    current = {
        "generation": state.get("generation"),
        "score": state.get("score"),
        "updated_at": state.get("updated_at"),
    }

    if current and current != last_state:
        precept_trace.add_event(
            "compass_compilation_status",
            {
                "state": state,
                "config": status.get("config", {}),
                "stats": status.get("stats", {}),
            },
        )
        last_state.clear()
        last_state.update(current)


def extract_mode_metrics(
    precept_results: List,
    expel_results: List,
    fr_results: List,
    scenarios: List,
    num_train: int,
    precept_stats: Dict,
    expel_stats: Dict,
    fr_stats: Dict,
) -> Dict:
    """
    Extract summary metrics for a test mode.

    Returns a dict with key metrics for each agent.
    """
    # Calculate metrics for each agent
    precept_m = calculate_metrics(
        precept_results, precept_stats, "PRECEPT", scenarios, num_train
    )
    expel_m = calculate_metrics(
        expel_results, expel_stats, "ExpeL", scenarios, num_train
    )
    fr_m = calculate_metrics(
        fr_results, fr_stats, "Full Reflexion", scenarios, num_train
    )

    return {
        "precept": {
            "success_rate": round(precept_m.success_rate, 4),
            "first_try_success_rate": round(precept_m.first_try_success_rate, 4),
            "first_try_successes": precept_m.first_try_successes,
            "total_tasks": len(scenarios),
            "avg_steps": round(precept_m.avg_steps, 2),
        },
        "expel": {
            "success_rate": round(expel_m.success_rate, 4),
            "first_try_success_rate": round(expel_m.first_try_success_rate, 4),
            "first_try_successes": expel_m.first_try_successes,
            "total_tasks": len(scenarios),
            "avg_steps": round(expel_m.avg_steps, 2),
        },
        "full_reflexion": {
            "success_rate": round(fr_m.success_rate, 4),
            "first_try_success_rate": round(fr_m.first_try_success_rate, 4),
            "first_try_successes": fr_m.first_try_successes,
            "total_tasks": len(scenarios),
            "avg_steps": round(fr_m.avg_steps, 2),
        },
    }


# =============================================================================
# MAIN TEST (Domain-Agnostic) - 4-WAY COMPARISON
# =============================================================================


async def run_domain_test(
    domain: str,
    num_train: int = 6,
    num_test: int = 4,
    concurrent_testing: bool = False,
    concurrent_training: bool = False,
    max_workers: int = 4,
    training_workers: int = 2,
    agent_internal_workers: int = 3,
    seed: int = None,
    enable_static_knowledge: bool = True,
    max_retries: int = None,
    detailed_logs: bool = False,
    trace_file: str = None,
    num_conditions: int = 1,
    train_num_conditions: int = None,
    test_num_conditions: int = None,
    test_mode: str = "matched",
    args: argparse.Namespace = None,  # Pass args for extra flags
):
    """
    Run the PRECEPT vs 3 Baselines comparison for ANY domain.

    Args:
        domain: Domain to test (logistics, coding, devops, etc.)
        num_train: Number of training tasks (sequential for learning)
        num_test: Number of test tasks
        concurrent_testing: If True, run agents in parallel per test task.
                           Faster but disables continuous learning during testing.
        concurrent_training: If True, run training tasks in parallel ("Tesla Fleet" mode).
                            Multiple agents learn simultaneously, sharing the Evo-Memory.
                            GEPA consolidation runs as a batch job after all training.
        max_workers: Maximum number of agents to run concurrently during testing (1-4).
                    1 = sequential, 2 = pairs, 3 = three at a time, 4 = all parallel
        training_workers: Maximum number of training tasks to run in parallel.
                         Higher = faster training but more API calls. Default: 2
        num_conditions: Number of conditions per scenario (1-10). Default: 1.
                       Higher values challenge baselines exponentially.
        train_num_conditions: Override num_conditions for training phase only.
                             For compositional generalization: train on simpler conditions.
        test_num_conditions: Override num_conditions for testing phase only.
                            For compositional generalization: test on complex conditions.
        agent_internal_workers: Max concurrent internal operations PER AGENT.
                               Controls parallelism of LLM calls, MCP tools within each agent.
        seed: Random seed for reproducibility (passed to save_experiment_results).
        enable_static_knowledge: If True, ingest static knowledge for ALL agents.
                                If False, no agents get static knowledge (pure dynamic learning).
        max_retries: Maximum number of retries allowed (same for all agents).
                    - None: Uses default (2)
                    - 1: Near first-try only (1 initial + 1 retry = 2 attempts)
                    - 2: Balanced (1 initial + 2 retries = 3 attempts) [default]
                    - 4: Lenient (1 initial + 4 retries = 5 attempts)
        detailed_logs: If True, save step-by-step execution traces for analysis.
        trace_file: Path for detailed trace file. Default: data/trace_{domain}_{timestamp}.json
        condition_aware_baselines: ABLATION MODE. If True, show condition codes in ExpeL insights
                                   and Reflexion reflections. This gives baselines PRECEPT-like
                                   condition→solution mappings to measure how much of PRECEPT's
                                   advantage comes from structured rule learning vs just having
                                   condition information.
    """
    icon = DOMAIN_ICONS.get(domain, "🔬")

    # Print experiment header
    display.print_header(domain, icon)

    # Check API
    if not check_api_availability():
        logger.error("OpenAI API not available. Set OPENAI_API_KEY in .env")
        return
    display.print_api_status(True)

    # Load stats
    stats = load_stats()

    # ═══════════════════════════════════════════════════════════════════════════
    # EXECUTION TRACER (Detailed step-by-step logging)
    # ═══════════════════════════════════════════════════════════════════════════
    tracer = ExecutionTracer(
        domain=domain,
        enabled=detailed_logs,
    )
    if detailed_logs:
        tracer.set_metadata(
            {
                "num_train": num_train,
                "num_test": num_test,
                "concurrent_testing": concurrent_testing,
                "concurrent_training": concurrent_training,
                "max_workers": max_workers,
                "training_workers": training_workers,
                "seed": seed,
                "enable_static_knowledge": enable_static_knowledge,
                "max_retries": max_retries,
            }
        )
        logger.info("📝 Detailed execution tracing: ENABLED")
    logger.info(f"📊 Previous runs: {stats['total_runs']}")

    # Get strategies for the selected domain
    logger.info(f"🚀 Initializing 4 agents for {domain.upper()} domain...")

    # Log max_retries setting
    effective_max_retries = max_retries if max_retries is not None else 2
    logger.info(
        f"🔄 Max retries: {effective_max_retries} (total attempts = {effective_max_retries + 1})"
    )

    precept_strategy = get_domain_strategy(domain, max_retries=max_retries)
    baseline_strategy = get_baseline_strategy(domain, max_retries=max_retries)

    # Create baseline config with max_attempts = max_retries + 1
    # (1 initial attempt + max_retries retries)
    baseline_max_attempts = (max_retries if max_retries is not None else 2) + 1
    baseline_config = BaselineConfig(
        max_attempts=baseline_max_attempts,
        max_internal_workers=agent_internal_workers,
        model=args.model,  # Use CLI-specified model
    )
    logger.info(f"📋 Baseline max_attempts: {baseline_max_attempts}")
    logger.info(f"📋 LLM model: {args.model}")

    # ═══════════════════════════════════════════════════════════════════════════
    # CREATE ALL 4 AGENTS (with configurable internal concurrency)
    # ═══════════════════════════════════════════════════════════════════════════

    # Create PRECEPT config with max_retries (single source of truth)
    precept_config = PreceptConfig()
    precept_config.llm.model = args.model  # Use CLI-specified model
    precept_config.agent.max_retries = effective_max_retries
    precept_config.agent.max_internal_workers = agent_internal_workers
    precept_config.agent.enable_llm_reasoning = True
    # Enable verbose LLM logging when detailed_logs is True (--verbose flag)
    precept_config.agent.verbose_llm = detailed_logs

    # COMPOSITIONAL GENERALIZATION: Read from environment variables
    # These are set by run_exp6_compositional_generalization.py
    if os.getenv("PRECEPT_COMPOSITIONAL_GENERALIZATION", "false").lower() == "true":
        precept_config.agent.enable_compositional_generalization = True
        logger.info("🧬 Compositional generalization: ENABLED")
    if os.getenv("PRECEPT_ATOMIC_PRECEPT_STORAGE", "false").lower() == "true":
        precept_config.agent.enable_atomic_precept_storage = True
        logger.info("⚛️ Atomic precept storage: ENABLED")
    if os.getenv("PRECEPT_CONSTRAINT_CONFLICT_DETECTION", "false").lower() == "true":
        precept_config.agent.enable_constraint_conflict_detection = True
        logger.info("⚔️ Constraint conflict detection: ENABLED")
    logger.info(
        f"📋 PRECEPT max_retries: {precept_config.agent.max_retries} (from config)"
    )

    # Set embedding model as environment variable for vector stores
    # This affects ChromaDB initialization in both PRECEPT MCP server and ExpeL
    os.environ["PRECEPT_EMBEDDING_MODEL"] = args.embedding_model
    logger.info(f"📋 Embedding model: {args.embedding_model}")

    # 1. PRECEPT Agent: Full learning stack
    precept_agent = PRECEPTAgent(
        domain_strategy=precept_strategy,
        config=precept_config,
        server_script=SERVER_SCRIPT,
    )
    await precept_agent.connect()

    # ═══════════════════════════════════════════════════════════════════════════
    # ABLATION: Configure PRECEPT's hybrid retrieval mode
    # When --hybrid-retrieval is set, PRECEPT uses BM25 + semantic for Tier 2
    # ═══════════════════════════════════════════════════════════════════════════
    if args.hybrid_retrieval:
        try:
            _hybrid_result = await precept_agent.mcp_client.call_tool(
                "configure_hybrid_retrieval", {"enabled": True}
            )
            logger.info("  🔧 PRECEPT hybrid retrieval: ENABLED")
        except Exception as e:
            logger.warning(f"  ⚠️ Could not enable PRECEPT hybrid retrieval: {e}")

    # ═══════════════════════════════════════════════════════════════════════════
    # FAIR COMPARISON: Clear PRECEPT's learned data (like Full Reflexion does)
    # This ensures both agents start from scratch for each experiment run.
    # ═══════════════════════════════════════════════════════════════════════════
    if args.preserve_learned_rules:
        logger.info(
            "  🔒 PRECEPT reset skipped: preserving learned rules and domain mappings"
        )
    else:
        try:
            clear_result = await precept_agent.mcp_client.clear_learned_data(
                clear_rules=True,
                clear_experiences=False,  # Keep experiences (vector DB) for semantic search
                clear_domain_mappings=True,
            )
            logger.info(f"  🧹 PRECEPT reset for fair comparison: {clear_result}")
        except Exception as e:
            logger.warning(f"  ⚠️ Could not clear PRECEPT data: {e}")

    # ═══════════════════════════════════════════════════════════════════════════
    # NOTE: LLM Baseline and Adapted Reflexion are COMMENTED OUT because they
    # do not participate in training (T_train = 0) and thus do not provide a
    # FAIR comparison. Only Full Reflexion (which trains alongside PRECEPT)
    # is a fair baseline. See docs/THEORETICAL_BOUNDS.md for details.
    # ═══════════════════════════════════════════════════════════════════════════

    # # 2. LLM Baseline: Error feedback only (Adapted ReAct)
    # # UNFAIR: Does not participate in training phase
    # llm_baseline = LLMBaselineAgent(
    #     baseline_strategy=baseline_strategy,
    #     config=baseline_config,
    #     server_script=SERVER_SCRIPT,
    #     verbose=False,
    #     max_internal_workers=agent_internal_workers,
    # )
    # await llm_baseline.connect()

    # # 3. Reflexion Baseline: Within-task reflection (Adapted Reflexion)
    # # UNFAIR: Does not participate in training phase (within-task only)
    # reflexion_baseline = ReflexionBaselineAgent(
    #     baseline_strategy=baseline_strategy,
    #     config=baseline_config,
    #     server_script=SERVER_SCRIPT,
    #     verbose=False,
    #     max_internal_workers=agent_internal_workers,
    # )
    # await reflexion_baseline.connect()

    # 2. Full Reflexion Baseline: Cross-episode memory (Full paper)
    # FAIR: Participates in training phase alongside PRECEPT
    # ═══════════════════════════════════════════════════════════════════════
    # BUGFIX: Gate memory clearing on --preserve-learned-rules.
    # When preserving, load previously saved reflections from disk instead
    # of clearing. This gives baselines the same cross-subprocess persistence
    # that PRECEPT gets via precept_learned_rules.json.
    # ═══════════════════════════════════════════════════════════════════════
    if args.preserve_learned_rules:
        from precept.baseline_functions import load_reflection_memory
        loaded = load_reflection_memory()
        logger.info(f"  🔒 Full Reflexion reset skipped: loaded {loaded} reflections from disk")
    else:
        FullReflexionBaselineAgent.clear_memory()
    full_reflexion = FullReflexionBaselineAgent(
        baseline_strategy=baseline_strategy,
        config=baseline_config,
        server_script=SERVER_SCRIPT,
        verbose=detailed_logs,  # Enable verbose logging when --verbose flag is set
        max_internal_workers=agent_internal_workers,
        condition_enhanced_retrieval=args.condition_enhanced_retrieval,  # ABLATION
        hybrid_retrieval=args.hybrid_retrieval,  # ABLATION: BM25 + semantic
        improved_baselines=args.improved_baselines,  # IMPROVED: Metadata filtering
    )
    await full_reflexion.connect()

    # 3. ExpeL Baseline: Experiential Learning (Zhao et al., 2023)
    # FAIR: Participates in training phase - extracts generalizable insights
    # ═══════════════════════════════════════════════════════════════════════
    # BUGFIX: Same persistence fix for ExpeL insights.
    # ═══════════════════════════════════════════════════════════════════════
    if args.preserve_learned_rules:
        from precept.baseline_functions import load_expel_insights
        loaded = load_expel_insights()
        logger.info(f"  🔒 ExpeL reset skipped: loaded {loaded} insights from disk")
    else:
        clear_expel_insights()
    expel_baseline = ExpeL_BaselineAgent(
        baseline_strategy=baseline_strategy,
        config=baseline_config,
        server_script=SERVER_SCRIPT,
        verbose=detailed_logs,
        max_internal_workers=agent_internal_workers,
        condition_enhanced_retrieval=args.condition_enhanced_retrieval,  # ABLATION
        hybrid_retrieval=args.hybrid_retrieval,  # ABLATION: BM25 + semantic
        improved_baselines=args.improved_baselines,  # IMPROVED: Metadata filtering
    )
    await expel_baseline.connect()

    display.print_agents_initialized()

    # ═══════════════════════════════════════════════════════════════════════════
    # PHASE 0: STATIC KNOWLEDGE INGESTION (Simple on/off)
    # ═══════════════════════════════════════════════════════════════════════════
    # When enabled: ALL agents (PRECEPT + baselines) get static knowledge
    # When disabled: NO agents get static knowledge (pure dynamic learning)
    #
    # KEY DIFFERENCE: PRECEPT has cutting-edge CONFLICT RESOLUTION
    # Baselines can only read static knowledge naively (no conflict detection)
    # ═══════════════════════════════════════════════════════════════════════════
    static_kb = None

    logger.info("")
    logger.info("=" * 85)
    if enable_static_knowledge:
        logger.info("📚 PHASE 0: STATIC KNOWLEDGE INGESTION (All Agents)")
    else:
        logger.info("📚 PHASE 0: STATIC KNOWLEDGE DISABLED (Pure Dynamic Learning)")
    logger.info("=" * 85)

    # Preserve random state so SK generation doesn't alter the task
    # generation seed.  This ensures WITH-SK and NO-SK runs produce the
    # same train/test split for a given seed.
    import random as _rng_mod
    _rng_state_before_sk = _rng_mod.getstate()

    if not enable_static_knowledge:
        logger.info("  ℹ️ Static knowledge ingestion is DISABLED")
        logger.info(
            "  📊 All agents will rely ONLY on dynamic learning from experience"
        )
        logger.info("  💡 This isolates test-time learning capabilities for comparison")
    else:
        # Import dynamic static knowledge generator
        from precept.config import generate_dynamic_static_knowledge

        # Start with empty list
        all_knowledge = []

        # 1. Load base static knowledge from JSON file (if exists)
        static_kb_path = DATA_DIR / "static_knowledge" / f"{domain}_kb.json"
        if static_kb_path.exists():
            try:
                with open(static_kb_path, "r") as f:
                    base_kb = json.load(f)
                all_knowledge.extend(base_kb)
                logger.info(
                    f"  📖 Loaded {len(base_kb)} base items from {static_kb_path.name}"
                )
            except Exception as e:
                logger.warning(f"  ⚠️ Could not load base static knowledge: {e}")

        # 2. Generate DYNAMIC conflicting knowledge based on num_conditions
        logger.info(
            f"  🔄 Generating dynamic conflicts for num_conditions={num_conditions}..."
        )
        try:
            dynamic_conflicts = generate_dynamic_static_knowledge(
                domain=domain,
                num_conditions=num_conditions,
            )
            all_knowledge.extend(dynamic_conflicts)
            logger.info(
                f"  ✓ Generated {len(dynamic_conflicts)} dynamic conflict items"
            )

            # Log conflict breakdown
            full_conflicts = sum(
                1
                for k in dynamic_conflicts
                if k.get("metadata", {}).get("conflict_type") == "full"
            )
            partial_subset = sum(
                1
                for k in dynamic_conflicts
                if k.get("metadata", {}).get("conflict_type") == "partial_subset"
            )
            partial_superset = sum(
                1
                for k in dynamic_conflicts
                if k.get("metadata", {}).get("conflict_type") == "partial_superset"
            )
            outdated = sum(
                1
                for k in dynamic_conflicts
                if k.get("metadata", {}).get("conflict_type") == "outdated"
            )
            agreement = sum(
                1
                for k in dynamic_conflicts
                if k.get("metadata", {}).get("conflict_type") == "agreement"
            )

            logger.info(
                f"     Full conflicts: {full_conflicts} (same {num_conditions} conditions, different solutions)"
            )
            logger.info(
                f"     Partial (subset): {partial_subset} (fewer conditions - TRAP for baselines)"
            )
            logger.info(
                f"     Partial (superset): {partial_superset} (more conditions)"
            )
            logger.info(f"     Outdated: {outdated} (dynamic should override)")
            logger.info(f"     Agreement: {agreement} (confidence boost)")
        except Exception as e:
            logger.warning(f"  ⚠️ Dynamic conflict generation failed: {e}")

        # 3. Combine and ingest
        if all_knowledge:
            static_kb = all_knowledge
            logger.info(f"  📊 Total static knowledge items: {len(static_kb)}")

            kb_json = json.dumps(static_kb)

            # Ingest for ALL agents (PRECEPT + baselines)
            logger.info("  📖 Ingesting static knowledge for ALL agents...")

            # PRECEPT
            try:
                result = await precept_agent.mcp_client.call_tool(
                    "ingest_static_knowledge",
                    {
                        "knowledge_items": kb_json,
                        "domain": domain,
                        "source": f"dynamic_{num_conditions}C_{domain}",
                    },
                )
                logger.info(f"  ✓ PRECEPT: {result}")
            except Exception as e:
                logger.warning(f"  ⚠️ PRECEPT ingestion failed: {e}")

            # Full Reflexion Baseline
            try:
                result = await full_reflexion.mcp_client.call_tool(
                    "ingest_static_knowledge",
                    {
                        "knowledge_items": kb_json,
                        "domain": domain,
                        "source": f"dynamic_{num_conditions}C_{domain}",
                    },
                )
                logger.info(f"  ✓ Full Reflexion: {result}")
            except Exception as e:
                logger.warning(f"  ⚠️ Full Reflexion ingestion failed: {e}")

            # ExpeL Baseline: inject SK into ExpeL's own insight store
            # for fair comparison (ExpeL uses an isolated retrieval pipeline)
            try:
                from precept.baseline_functions import add_expel_insight
                expel_sk_count = 0
                for item in static_kb:
                    content = item.get("content", "") if isinstance(item, dict) else str(item)
                    conditions = []
                    if isinstance(item, dict):
                        if "conditions" in item:
                            conditions = item["conditions"]
                        elif "condition" in item:
                            conditions = [item["condition"]]
                    add_expel_insight(
                        {
                            "insight": content,
                            "task": f"Static knowledge for {domain}",
                            "conditions": conditions,
                            "type": "static_kb",
                            "confidence": "high",
                            "source": "static_knowledge",
                        },
                        condition_enhanced=args.condition_enhanced_retrieval,
                        improved_baselines=args.improved_baselines,
                    )
                    expel_sk_count += 1
                logger.info(f"  ✓ ExpeL: Injected {expel_sk_count} static knowledge items into insight store")
            except Exception as e:
                logger.warning(f"  ⚠️ ExpeL SK ingestion failed: {e}")

            logger.info("")
            logger.info("  ═══════════════════════════════════════════════════════════")
            logger.info("  💡 KEY DIFFERENCE: PRECEPT vs Baselines")
            logger.info("  ═══════════════════════════════════════════════════════════")
            logger.info("  ALL agents have access to the SAME static knowledge.")
            logger.info(
                f"  Static knowledge is AWARE of num_conditions={num_conditions}"
            )
            logger.info("")
            logger.info("  CONFLICT TYPES GENERATED:")
            logger.info(
                f"    • Full conflicts (same {num_conditions} conditions, different solutions)"
            )
            logger.info("    • Partial subset (fewer conditions - TRAP for baselines)")
            logger.info("    • Partial superset (more conditions)")
            logger.info("    • Outdated (dynamic should override)")
            logger.info("")
            logger.info("  PRECEPT's UNIQUE ADVANTAGES:")
            logger.info(
                "    ✓ Cutting-edge CONFLICT RESOLUTION between static & dynamic"
            )
            logger.info("    ✓ EXACT condition matching (won't apply partial matches)")
            logger.info("    ✓ Bayesian uncertainty quantification for reliability")
            logger.info("    ✓ Evidence-based prioritization (anomaly detection)")
            logger.info("    ✓ Dynamic reliability learning from outcomes")
            logger.info("")
            logger.info("  Full Reflexion TRAPS:")
            logger.info("    • May apply partial matches (subset conditions)")
            logger.info("    • No structured conflict detection")
            logger.info("    • Relies on LLM interpretation of natural language")
            logger.info("  ═══════════════════════════════════════════════════════════")
        else:
            logger.info("  ℹ️ No knowledge items generated (check domain configuration)")

    # Restore random state to ensure identical task generation regardless
    # of whether SK generation consumed random numbers above.
    _rng_mod.setstate(_rng_state_before_sk)

    # Generate scenarios
    scenario_generator = SCENARIO_GENERATORS.get(domain)
    if not scenario_generator:
        logger.error(f"No scenario generator for domain: {domain}")
        return

    # Determine conditions for train and test phases
    effective_train_conditions = train_num_conditions or num_conditions
    effective_test_conditions = test_num_conditions or num_conditions
    is_compositional_mode = effective_train_conditions != effective_test_conditions

    # Check for semantic compositional mode (where P₁ > 0% is achievable)
    is_semantic_compositional = args is not None and getattr(
        args, "semantic_compositional", False
    )

    total_needed = num_train + num_test
    train_ratio = num_train / total_needed

    # All domains now support semantic compositional mode
    SEMANTIC_DOMAINS = (
        "logistics",
        "devops",
        "finance",
        "booking",
        "coding",
        "integration",
    )

    if is_semantic_compositional and domain in SEMANTIC_DOMAINS:
        # ═══════════════════════════════════════════════════════════════════════
        # SEMANTIC COMPOSITIONAL MODE: Solutions are DERIVABLE from atomic precepts
        # This enables P₁ > 0% because LLM can reason: A→X, B→Y ⟹ A+B→X_Y
        # Supported domains: ALL (logistics, devops, finance, booking, coding, integration)
        # ═══════════════════════════════════════════════════════════════════════
        logger.info(
            f"🧠 SEMANTIC COMPOSITIONAL MODE ({domain.upper()}): Solutions derivable from atomic precepts"
        )
        logger.info(
            "   Unlike Black Swan CSPs, composite solutions follow predictable patterns"
        )
        logger.info("   Higher tier wins: tier=3 > tier=2 > tier=1")

        # Get beta (repetitions per atom) from args
        beta = getattr(args, "beta", 1) if args else 1
        filter_by_learned = getattr(args, "filter_by_learned", False) if args else False

        # IMPORTANT: num_train from args is TOTAL training tasks (beta × num_atoms)
        # We need to compute the number of unique atoms to train
        num_unique_atoms = max(1, num_train // beta) if beta > 0 else num_train

        logger.info(f"   Beta={beta} (each atom trained {beta}x for robust learning)")
        logger.info(
            f"   Unique atoms to train: {num_unique_atoms} (total tasks: {num_train})"
        )
        if filter_by_learned:
            logger.info(
                "   🎯 FILTER-BY-LEARNED: Will only test composites where ALL atoms were LEARNED"
            )

        # Get test_num_conditions from args for M-way compositional generalization
        effective_test_num_conditions = (
            getattr(args, "test_num_conditions", 2) if args else 2
        )
        if effective_test_num_conditions is None:
            effective_test_num_conditions = 2  # Default to 2-way

        # Use domain-specific semantic generator
        if domain == "logistics":
            from precept.scenario_generators.logistics import LogisticsScenarioGenerator

            semantic_gen = LogisticsScenarioGenerator(
                num_samples=num_train + num_test, train_ratio=0.5
            )
        elif domain == "devops":
            from precept.scenario_generators.devops import DevOpsScenarioGenerator

            semantic_gen = DevOpsScenarioGenerator(
                num_samples=num_train + num_test, train_ratio=0.5
            )
        elif domain == "finance":
            from precept.scenario_generators.finance import FinanceScenarioGenerator

            semantic_gen = FinanceScenarioGenerator(
                num_samples=num_train + num_test, train_ratio=0.5
            )
        elif domain == "booking":
            from precept.scenario_generators.booking import BookingScenarioGenerator

            semantic_gen = BookingScenarioGenerator(
                num_samples=num_train + num_test, train_ratio=0.5
            )
        elif domain == "coding":
            from precept.scenario_generators.coding import CodingScenarioGenerator

            semantic_gen = CodingScenarioGenerator(
                num_samples=num_train + num_test, train_ratio=0.5
            )
        elif domain == "integration":
            from precept.scenario_generators.integration import (
                IntegrationScenarioGenerator,
            )

            semantic_gen = IntegrationScenarioGenerator(
                num_samples=num_train + num_test, train_ratio=0.5
            )
        else:
            raise ValueError(f"Unknown domain for semantic compositional: {domain}")

        train_scenarios, test_scenarios, semantic_mappings = (
            semantic_gen.generate_semantic_compositional_test(
                num_train=num_unique_atoms,  # Number of unique atomic conditions
                num_test=num_test,
                seed=seed,
                beta=beta,
                filter_by_learned=filter_by_learned,
                test_num_conditions=effective_test_num_conditions,  # Support 1→M generalization
            )
        )

        logger.info(
            f"📊 SEMANTIC: {len(train_scenarios)} atomic train + {len(test_scenarios)} compositional test"
        )
        logger.info(f"   Trained atoms: {semantic_mappings.get('trained_atoms', [])}")
        logger.info(
            f"   Derivation rule: {semantic_mappings.get('derivation_rule', 'N/A')}"
        )

        # Set compositional mode flag for later processing
        is_compositional_mode = True

    elif is_compositional_mode:
        # COMPOSITIONAL GENERALIZATION: Generate train and test separately
        logger.info(
            f"🧬 COMPOSITIONAL MODE: train={effective_train_conditions}C → test={effective_test_conditions}C"
        )

        # Generate TRAINING scenarios with fewer conditions (atomic)
        train_kwargs = {
            "num_samples": num_train,
            "train_ratio": 1.0,  # All training
            "include_generator_samples": False,
            "num_conditions": effective_train_conditions,
            "test_mode": "matched",
            "include_fleet_learning": True,
        }
        if domain == "logistics":
            train_kwargs["include_conflict_resolution"] = True
        train_scenarios = scenario_generator(**train_kwargs)
        train_scenarios = [s for s in train_scenarios if s.get("phase") == "training"][
            :num_train
        ]

        # Generate TEST scenarios with more conditions (composite)
        test_kwargs = {
            "num_samples": num_test,
            "train_ratio": 0.0,  # All testing
            "include_generator_samples": False,
            "num_conditions": effective_test_conditions,
            "test_mode": "random",  # Use random for novel combinations
            "include_fleet_learning": False,
        }
        if domain == "logistics":
            test_kwargs["include_conflict_resolution"] = False
        test_scenarios = scenario_generator(**test_kwargs)
        # Mark all as test phase
        for s in test_scenarios:
            s["phase"] = "test"
        test_scenarios = test_scenarios[:num_test]

        logger.info(
            f"📊 COMPOSITIONAL: {len(train_scenarios)} train ({effective_train_conditions}C) + {len(test_scenarios)} test ({effective_test_conditions}C)"
        )
    else:
        # STANDARD MODE: Same conditions for train and test
        # Build scenario generator kwargs (only pass supported params per domain)
        # Note: num_conditions is now passed to ALL scenarios (including fleet learning)
        # for consistency - Option B implementation
        generator_kwargs = {
            "num_samples": total_needed,
            "train_ratio": train_ratio,
            "include_generator_samples": False,
            "num_conditions": num_conditions,
            "test_mode": test_mode,  # matched = O(1) lookup, random = generalization test
            "include_fleet_learning": True,  # All domains support this
        }

        # Only logistics has conflict_resolution scenarios
        if domain == "logistics":
            generator_kwargs["include_conflict_resolution"] = True

        all_scenarios = scenario_generator(**generator_kwargs)
        logger.info(f"📊 Generated {len(all_scenarios)} {domain} scenarios")

        # Split scenarios
        train_scenarios = [s for s in all_scenarios if s.get("phase") == "training"]
        test_scenarios = [s for s in all_scenarios if s.get("phase") == "test"]

        if not train_scenarios or not test_scenarios:
            mid = len(all_scenarios) // 2
            train_scenarios = all_scenarios[:mid]
            test_scenarios = all_scenarios[mid:]

        train_scenarios = train_scenarios[:num_train]
        test_scenarios = test_scenarios[:num_test]

        if len(train_scenarios) < num_train:
            train_scenarios = (
                train_scenarios * (num_train // len(train_scenarios) + 1)
            )[:num_train]
        if len(test_scenarios) < num_test:
            test_scenarios = (test_scenarios * (num_test // len(test_scenarios) + 1))[
                :num_test
            ]

    logger.info(
        f"   num_samples={total_needed}, train_ratio={train_ratio:.0%} ({num_train} train / {num_test} test)"
    )

    logger.info("=" * 85)
    logger.info(f"📚 TRAIN-TEST EVALUATION for {domain.upper()} domain")
    logger.info(
        f"   Training: {len(train_scenarios)} tasks (PRECEPT + Full Reflexion learn)"
    )
    logger.info(f"   Testing:  {len(test_scenarios)} tasks (4-way comparison)")
    logger.info("=" * 85)

    # Results storage (FAIR COMPARISON: PRECEPT vs Full Reflexion vs ExpeL)
    precept_results = []
    # llm_results = []  # Not used (unfair comparison)
    # reflexion_results = []  # Not used (unfair comparison)
    full_reflexion_results = []
    expel_results = []

    # ═══════════════════════════════════════════════════════════════════════════
    # PHASE 1: TRAINING (PRECEPT + Full Reflexion learn)
    # ═══════════════════════════════════════════════════════════════════════════
    if concurrent_training:
        details = [
            "⚡ CONCURRENT TRAINING MODE ('Tesla Fleet' Architecture)",
            f"   • {training_workers} training tasks run in parallel",
            "   • All agents share Evo-Memory (Vector DB)",
            "   • GEPA consolidation runs as batch job after training",
            "PRECEPT:          Learns rules + triggers COMPASS evolution",
            "Full Reflexion: Builds cross-episode memory",
            "Others:         Skip (no cross-episode learning capability)",
        ]
    else:
        details = [
            "🔄 SEQUENTIAL TRAINING (standard cross-task learning)",
            "PRECEPT:          Learns rules + triggers COMPASS evolution",
            "Full Reflexion: Builds cross-episode memory",
            "Others:         Skip (no cross-episode learning capability)",
        ]
    display.print_phase_header("1: TRAINING", len(train_scenarios), details)

    train_start_time = time_module.time()

    # ═══════════════════════════════════════════════════════════════════════════
    # TRAINING OUTCOME TRACKING (PER-AGENT)
    # ═══════════════════════════════════════════════════════════════════════════
    # TRACK ALL TRAINING CONDITION KEYS (Unified - All Agents Same Test Set)
    # ═══════════════════════════════════════════════════════════════════════════
    # Track ALL condition_keys that appeared in training.
    # Test scenarios will be filtered to only include these keys.
    # ALL agents will be tested on the SAME filtered scenarios.
    # This is the cleanest, fairest comparison - all agents exposed to same
    # training scenarios and tested on same test scenarios.
    # ═══════════════════════════════════════════════════════════════════════════
    training_condition_keys = set()  # All condition keys from training

    if concurrent_training:
        # ═══════════════════════════════════════════════════════════════════════
        # CONCURRENT TRAINING ("Tesla Fleet" Mode)
        # ═══════════════════════════════════════════════════════════════════════
        # Multiple agents learn simultaneously, sharing the Evo-Memory.
        # This works because PRECEPT doesn't use gradient descent - learning is
        # simply writing to a database. No synchronization needed!
        # ═══════════════════════════════════════════════════════════════════════
        training_semaphore = asyncio.Semaphore(training_workers)
        training_results = []

        async def train_task(scenario, task_num, update_progress):
            """Train on a single scenario with semaphore control."""
            async with training_semaphore:
                task = scenario["task"]

                # ═══════════════════════════════════════════════════════════════════
                # EXTRACT MULTI-CONDITION METADATA from scenario
                # This passes condition_key to agents without it being in task string
                # ═══════════════════════════════════════════════════════════════════
                multi_condition = scenario.get("multi_condition", {})
                cond_key = multi_condition.get("condition_key") or scenario.get(
                    "condition_key"
                )
                metadata = {
                    "condition_key": cond_key,
                    "conditions": multi_condition.get("conditions", []),
                    "expected_solution": multi_condition.get("expected_solution")
                    or multi_condition.get("solution"),
                }

                # Start tracing for this training task
                precept_trace = tracer.start_task(task_num, task, "training", "precept")
                fr_trace = tracer.start_task(
                    task_num, task, "training", "full_reflexion"
                )
                expel_trace = tracer.start_task(task_num, task, "training", "expel")

                # All learning agents learn from the same task concurrently (with metadata)
                precept_result, fr_result, expel_result = await asyncio.gather(
                    precept_agent.run_task(task, metadata=metadata),
                    full_reflexion.run_task(task, metadata=metadata),
                    expel_baseline.run_task(task, training=True, metadata=metadata),
                    return_exceptions=True,
                )

                # Handle exceptions
                if isinstance(precept_result, Exception):
                    precept_result = {"success": False, "error": str(precept_result)}
                if isinstance(fr_result, Exception):
                    fr_result = {"success": False, "accumulated_reflections": 0}
                if isinstance(expel_result, Exception):
                    expel_result = {"success": False, "total_insights": 0}

                # Add execution details to traces
                precept_trace.add_event(
                    "task_complete",
                    {
                        "success": precept_result.get("success", False),
                        "task_steps": precept_result.get("task_steps", 0),
                        "overhead_steps": precept_result.get("overhead_steps", 0),
                        "strategy": precept_result.get("strategy", ""),
                        "response": str(precept_result.get("response", ""))[:200],
                        "condition_key": cond_key,
                        "conditions": metadata.get("conditions", []),
                        "expected_solution": metadata.get("expected_solution"),
                        "rule_learned": precept_result.get("rule_learned", False),
                        "learned_rule_key": precept_result.get("learned_rule_key", ""),
                        "learned_solution": precept_result.get("learned_solution", ""),
                        "learned_via": precept_result.get("learned_via", ""),
                    },
                )
                if precept_result.get("rule_learned"):
                    precept_trace.add_event(
                        "rule_learned",
                        {
                            "condition_key": precept_result.get("learned_rule_key")
                            or cond_key,
                            "solution": precept_result.get("learned_solution", ""),
                            "via": precept_result.get("learned_via", ""),
                            "phase": "training",
                        },
                    )
                tracer.end_task(precept_trace, precept_result)

                fr_trace.add_event(
                    "task_complete",
                    {
                        "success": fr_result.get("success", False),
                        "accumulated_reflections": fr_result.get(
                            "accumulated_reflections", 0
                        ),
                        "response": str(fr_result.get("response", ""))[:200],
                    },
                )
                tracer.end_task(fr_trace, fr_result)

                expel_trace.add_event(
                    "task_complete",
                    {
                        "success": expel_result.get("success", False),
                        "total_insights": expel_result.get("total_insights", 0),
                        "insights_extracted": expel_result.get("insights_extracted", 0),
                        "response": str(expel_result.get("response", ""))[:200],
                    },
                )
                tracer.end_task(expel_trace, expel_result)

                # Update progress bar
                update_progress(
                    precept_ok=precept_result.get("success", False),
                    fr_ok=fr_result.get("success", False),
                )

                # Return with condition_key for outcome tracking
                return precept_result, fr_result, expel_result, cond_key

        # Run all training tasks with progress bar
        with progress_tracker.training_progress(
            total=len(train_scenarios),
            description="🎓 Training (concurrent)",
        ) as update_progress:
            training_results = await asyncio.gather(
                *[
                    train_task(scenario, i, update_progress)
                    for i, scenario in enumerate(train_scenarios, 1)
                ]
            )

        # Collect all training condition keys
        for r in training_results:
            precept_res, fr_res, expel_res, cond_key = r
            if cond_key:
                training_condition_keys.add(cond_key)

        logger.info(f"  📊 Training complete: {len(training_results)} scenarios")
        logger.info(
            f"  📚 Unique condition keys in training: {len(training_condition_keys)}"
        )

        # Trigger GEPA consolidation as a batch job after all training
        logger.info("  🔄 Running batch GEPA consolidation (Map-Reduce)...")
        try:
            consolidation_result = (
                await precept_agent.mcp_client.trigger_consolidation()
            )
            logger.info(f"     {consolidation_result[:100]}...")
        except Exception as e:
            logger.warning(f"     Consolidation skipped: {e}")

    else:
        # ═══════════════════════════════════════════════════════════════════════
        # SEQUENTIAL TRAINING (Standard Cross-Task Learning)
        # ═══════════════════════════════════════════════════════════════════════
        with progress_tracker.training_progress(
            total=len(train_scenarios),
            description="🎓 Training (sequential)",
        ) as update_progress:
            for i, scenario in enumerate(train_scenarios, 1):
                task = scenario["task"]
                expected = scenario.get("expected", "N/A")

                # ═══════════════════════════════════════════════════════════════════
                # EXTRACT MULTI-CONDITION METADATA from scenario
                # This passes condition_key to agents without it being in task string
                # ═══════════════════════════════════════════════════════════════════
                multi_condition = scenario.get("multi_condition", {})
                cond_key = multi_condition.get("condition_key") or scenario.get(
                    "condition_key"
                )
                metadata = {
                    "condition_key": cond_key,
                    "conditions": multi_condition.get("conditions", []),
                    "expected_solution": multi_condition.get("expected_solution")
                    or multi_condition.get("solution"),
                }

                # Start tracing for this training task
                precept_trace = tracer.start_task(i, task, "training", "precept")
                fr_trace = tracer.start_task(i, task, "training", "full_reflexion")
                expel_trace = tracer.start_task(i, task, "training", "expel")

                # PRECEPT learns (with metadata for multi-condition enforcement)
                precept_result = await precept_agent.run_task(task, metadata=metadata)

                # Add execution details to trace
                precept_trace.add_event(
                    "task_complete",
                    {
                        "success": precept_result.get("success", False),
                        "task_steps": precept_result.get("task_steps", 0),
                        "overhead_steps": precept_result.get("overhead_steps", 0),
                        "strategy": precept_result.get("strategy", ""),
                        "response": str(precept_result.get("response", ""))[:200],
                        "condition_key": cond_key,
                        "conditions": metadata.get("conditions", []),
                        "expected_solution": metadata.get("expected_solution"),
                        "rule_learned": precept_result.get("rule_learned", False),
                        "learned_rule_key": precept_result.get("learned_rule_key", ""),
                        "learned_solution": precept_result.get("learned_solution", ""),
                        "learned_via": precept_result.get("learned_via", ""),
                    },
                )
                if precept_result.get("rule_learned"):
                    precept_trace.add_event(
                        "rule_learned",
                        {
                            "condition_key": precept_result.get("learned_rule_key")
                            or cond_key,
                            "solution": precept_result.get("learned_solution", ""),
                            "via": precept_result.get("learned_via", ""),
                            "phase": "training",
                        },
                    )
                tracer.end_task(precept_trace, precept_result)

                # Full Reflexion builds memory (with metadata for multi-condition enforcement)
                fr_result = await full_reflexion.run_task(task, metadata=metadata)

                # Add execution details to trace
                fr_trace.add_event(
                    "task_complete",
                    {
                        "success": fr_result.get("success", False),
                        "accumulated_reflections": fr_result.get(
                            "accumulated_reflections", 0
                        ),
                        "response": str(fr_result.get("response", ""))[:200],
                    },
                )
                tracer.end_task(fr_trace, fr_result)

                # ExpeL extracts insights (with metadata for multi-condition enforcement)
                expel_result = await expel_baseline.run_task(
                    task, training=True, metadata=metadata
                )

                # Add execution details to trace
                expel_trace.add_event(
                    "task_complete",
                    {
                        "success": expel_result.get("success", False),
                        "total_insights": expel_result.get("total_insights", 0),
                        "insights_extracted": expel_baseline.insights_extracted,
                        "response": str(expel_result.get("response", ""))[:200],
                    },
                )
                tracer.end_task(expel_trace, expel_result)

                # Update progress bar
                update_progress(
                    precept_ok=precept_result["success"],
                    fr_ok=fr_result["success"],
                )

                # Track all training condition keys (unified - all agents same test set)
                if cond_key:
                    training_condition_keys.add(cond_key)

                display.print_train_result(
                    num=i,
                    total=len(train_scenarios),
                    task=task,
                    expected=expected,
                    precept_success=precept_result["success"],
                    fr_success=fr_result["success"],
                    fr_memories=fr_result.get("accumulated_reflections", 0),
                )

    train_elapsed = time_module.time() - train_start_time

    # Training summary
    display.print_training_summary(
        elapsed=train_elapsed,
        concurrent=concurrent_training,
        training_workers=training_workers,
        rules_learned=len(precept_agent.learning_events),
        consolidations=precept_agent.tasks_since_consolidation,
        fr_memories=sum(FullReflexionBaselineAgent.get_memory_stats().values()),
    )

    # ═══════════════════════════════════════════════════════════════════════════
    # CRITICAL: SYNC LEARNED RULES BEFORE TESTING
    # ═══════════════════════════════════════════════════════════════════════════
    # In concurrent training, rules are persisted to disk asynchronously.
    # We MUST reload from disk to ensure all rules are available for testing.
    # This prevents race conditions where some rules aren't yet in memory.
    # ═══════════════════════════════════════════════════════════════════════════
    logger.info("  🔄 Synchronizing learned rules from disk...")
    try:
        sync_result = await precept_agent.mcp_client.reload_learned_rules()
        logger.info(f"  ✓ {sync_result}")
    except Exception as e:
        logger.warning(f"  ⚠ Rule sync warning: {e}")

    # ═══════════════════════════════════════════════════════════════════════════
    # BUGFIX: Save baseline memories to disk for cross-subprocess persistence.
    # When the experiment runs training and testing as separate subprocesses,
    # Python globals (reflection_memory, expel_insight_store) are lost.
    # Saving to JSON here allows the test subprocess to reload them when
    # --preserve-learned-rules is set, giving baselines the same persistence
    # advantage that PRECEPT already has via precept_learned_rules.json.
    # ═══════════════════════════════════════════════════════════════════════════
    try:
        from precept.baseline_functions import save_reflection_memory, save_expel_insights
        save_reflection_memory()
        save_expel_insights()
    except Exception as e:
        logger.warning(f"  ⚠ Baseline memory save warning: {e}")

    # ═══════════════════════════════════════════════════════════════════════════
    # POST-TRAINING: PREPARE TEST SCENARIOS FOR EACH MODE
    # ═══════════════════════════════════════════════════════════════════════════
    # For "both" mode, we prepare TWO sets of test scenarios:
    # 1. MATCHED: Filter test scenarios to only those with keys from training
    # 2. RANDOM: Use unfiltered test scenarios (new random keys)
    #
    # COMPOSITIONAL GENERALIZATION: When train/test conditions differ, skip
    # regeneration and use the pre-generated test scenarios with different
    # num_conditions (e.g., test on 2C when trained on 1C).
    # ═══════════════════════════════════════════════════════════════════════════

    # Determine if in compositional mode (different train/test conditions)
    effective_train_conditions = train_num_conditions or num_conditions
    effective_test_conditions = test_num_conditions or num_conditions
    is_compositional_mode = effective_train_conditions != effective_test_conditions

    # Check for semantic compositional mode (already generated with derivable solutions)
    is_semantic_compositional = args is not None and getattr(
        args, "semantic_compositional", False
    )

    if is_semantic_compositional:
        # SEMANTIC MODE: Test scenarios already generated with derivable solutions
        # Skip regeneration - P₁ > 0% is achievable through LLM reasoning
        logger.info(
            "\n🧠 SEMANTIC COMPOSITIONAL: Using pre-generated derivable test scenarios"
        )
        logger.info(
            "   Test solutions are derivable from atomic precepts via reasoning"
        )
        logger.info("   Example: COLD→reefer, HAZM→hazmat ⟹ COLD+HAZM→reefer_hazmat")

        # Check if we need to filter by actually-learned atoms
        filter_by_learned = getattr(args, "filter_by_learned", False) if args else False

        if filter_by_learned:
            # ═══════════════════════════════════════════════════════════════════
            # PURE COMPOSITIONAL TEST: Only test composites where ALL atoms LEARNED
            # This isolates the true compositional generalization capability
            # ═══════════════════════════════════════════════════════════════════
            logger.info(
                "\n🎯 FILTER-BY-LEARNED: Checking which atoms were actually learned..."
            )

            learned_atoms = set()
            try:
                atomic_precepts_path = DATA_DIR / "precept_atomic_precepts.json"
                if atomic_precepts_path.exists():
                    with open(atomic_precepts_path) as f:
                        atomic_precepts = json.load(f)
                    learned_atoms = set(atomic_precepts.keys())
                    logger.info(
                        f"   ✅ Learned {len(learned_atoms)} atomic precepts: {list(learned_atoms)}"
                    )
                else:
                    logger.warning(
                        "   ⚠️ No atomic precepts file found - using all test scenarios"
                    )
            except Exception as e:
                logger.warning(f"   ⚠️ Could not load atomic precepts: {e}")

            if learned_atoms:
                # Filter test scenarios to only those where ALL constituent atoms were learned
                filtered_tests = []
                skipped_tests = []

                for scenario in test_scenarios:
                    mc = scenario.get("multi_condition", {})
                    conditions = mc.get("conditions", [])

                    if not conditions:
                        # No conditions - keep it
                        filtered_tests.append(scenario)
                    else:
                        # Check if ALL atoms in this composite were learned
                        missing_atoms = [
                            c for c in conditions if c not in learned_atoms
                        ]
                        if missing_atoms:
                            skipped_tests.append(
                                {
                                    "condition_key": mc.get("condition_key", "unknown"),
                                    "missing": missing_atoms,
                                }
                            )
                        else:
                            filtered_tests.append(scenario)

                logger.info("   📊 Filtering results:")
                logger.info(f"      Original test scenarios: {len(test_scenarios)}")
                logger.info(f"      Kept (all atoms learned): {len(filtered_tests)}")
                logger.info(f"      Skipped (missing atoms): {len(skipped_tests)}")

                if skipped_tests:
                    logger.info("   🔍 Skipped composites:")
                    for skip in skipped_tests[:5]:  # Show first 5
                        logger.info(
                            f"      {skip['condition_key']} - missing: {skip['missing']}"
                        )

                if not filtered_tests:
                    logger.warning(
                        "   ⚠️ NO test scenarios passed filtering! Using all tests instead."
                    )
                    logger.warning(
                        "   This means no atomic precepts were fully learned during training."
                    )
                    filtered_tests = test_scenarios

                test_scenarios = filtered_tests

        # Use the (possibly filtered) test scenarios
        random_test_scenarios = test_scenarios
        matched_test_scenarios = []
        learned_rules = {}

        # Set up modes to run - only compositional mode with semantic scenarios
        modes_to_run = [("semantic_compositional", random_test_scenarios)]

        logger.info(
            f"\n🧪 SEMANTIC TEST: {len(random_test_scenarios)} derivable compositional scenarios"
        )

    elif is_compositional_mode:
        logger.info(
            "\n🧬 COMPOSITIONAL GENERALIZATION: Generating tests from learned atoms"
        )
        logger.info(
            f"   Training used {effective_train_conditions}-condition scenarios"
        )
        logger.info(
            f"   Testing uses {effective_test_conditions}-condition scenarios (novel combinations)"
        )

        # ═══════════════════════════════════════════════════════════════════════
        # LOAD LEARNED ATOMIC PRECEPTS
        # ═══════════════════════════════════════════════════════════════════════
        # After training on 1-condition scenarios, we have atomic precepts.
        # Use ONLY these learned atoms to create test composites.
        # ═══════════════════════════════════════════════════════════════════════
        learned_atoms = []
        try:
            atomic_precepts_path = DATA_DIR / "precept_atomic_precepts.json"
            if atomic_precepts_path.exists():
                with open(atomic_precepts_path) as f:
                    atomic_precepts = json.load(f)
                learned_atoms = list(atomic_precepts.keys())
                logger.info(
                    f"   ✅ Loaded {len(learned_atoms)} learned atomic precepts"
                )
                logger.info(f"      Atoms: {learned_atoms}")
            else:
                logger.warning(
                    "   ⚠️ No atomic precepts file found - using training condition keys"
                )
                # Fallback: extract atoms from training condition keys
                for key in training_condition_keys:
                    atoms = key.split("+")
                    learned_atoms.extend(atoms)
                learned_atoms = list(set(learned_atoms))
        except Exception as e:
            logger.warning(f"   ⚠️ Could not load atomic precepts: {e}")
            # Fallback to training keys
            for key in training_condition_keys:
                atoms = key.split("+")
                learned_atoms.extend(atoms)
            learned_atoms = list(set(learned_atoms))

        if not learned_atoms:
            logger.error("   ❌ No learned atoms available for compositional testing!")
            learned_atoms = []  # Will skip test generation

        # ═══════════════════════════════════════════════════════════════════════
        # GENERATE TEST COMPOSITES FROM ONLY LEARNED ATOMS
        # ═══════════════════════════════════════════════════════════════════════
        # This ensures P₁ > 0% is possible because ALL atoms in test composites
        # were learned during training and have associated precepts.
        # ═══════════════════════════════════════════════════════════════════════
        # Use the actual class instance, not the function wrapper
        compositional_scenarios = []
        try:
            if domain == "logistics":
                from precept.scenario_generators.logistics import (
                    LogisticsScenarioGenerator,
                )

                gen_instance = LogisticsScenarioGenerator(
                    num_samples=num_test, train_ratio=0.0
                )
                compositional_scenarios = gen_instance.generate_compositional_test(
                    learned_atoms=learned_atoms,
                    num_test=num_test,
                    num_conditions=effective_test_conditions,
                    include_conflicts=True,
                    seed=seed,
                )
            elif domain == "devops":
                from precept.scenario_generators.devops import DevOpsScenarioGenerator

                gen_instance = DevOpsScenarioGenerator(
                    num_samples=num_test, train_ratio=0.0
                )
                if hasattr(gen_instance, "generate_compositional_test"):
                    compositional_scenarios = gen_instance.generate_compositional_test(
                        learned_atoms=learned_atoms,
                        num_test=num_test,
                        num_conditions=effective_test_conditions,
                        include_conflicts=True,
                        seed=seed,
                    )
            # Add more domains as needed...
        except Exception as e:
            logger.warning(f"   ⚠️ Error generating compositional tests: {e}")
            compositional_scenarios = []

        if compositional_scenarios:
            logger.info(
                f"   ✅ Generated {len(compositional_scenarios)} compositional test scenarios"
            )
            logger.info(
                "      All composites use ONLY learned atoms → P₁ > 0% possible!"
            )
            random_test_scenarios = compositional_scenarios
        else:
            logger.warning(
                "   ⚠️ Could not generate compositional tests - using pre-generated scenarios"
            )
            random_test_scenarios = test_scenarios

        matched_test_scenarios = []
        learned_rules = {}  # Don't load - we're testing generalization, not matching

        # Set up modes to run - only compositional mode
        modes_to_run = [("compositional", random_test_scenarios)]

        logger.info(
            f"\n🧪 COMPOSITIONAL TEST: {len(random_test_scenarios)} novel {effective_test_conditions}-condition scenarios"
        )
        logger.info(
            "   Each composite uses ONLY atoms that were learned during training!"
        )

    # STANDARD MODE: Continue with normal test scenario preparation
    if not is_compositional_mode and not is_semantic_compositional:
        logger.info("\n📚 Training Summary:")
        logger.info(f"   Total training scenarios: {len(train_scenarios)}")
        logger.info(f"   Unique condition keys:    {len(training_condition_keys)}")

    # ═══════════════════════════════════════════════════════════════════════════
    # COLD START SUPPORT (β=0): Pre-generate condition keys without training
    # ═══════════════════════════════════════════════════════════════════════════
    # When β=0 (no training), we need to generate a fixed pool of condition keys
    # for testing. This allows fair comparison of online learning capabilities.
    # Skip for compositional mode (we already have pre-generated test scenarios).
    # ═══════════════════════════════════════════════════════════════════════════
    if (
        not is_compositional_mode
        and not is_semantic_compositional
        and len(train_scenarios) == 0
        and len(training_condition_keys) == 0
        and not getattr(args, "preserve_learned_rules", False)
    ):
        logger.info("\n🧊 COLD START MODE DETECTED (β=0)")
        logger.info("   Pre-generating condition keys for testing...")

        # Use the scenario generator to create test scenarios with fixed keys
        test_generator = _get_domain_test_generator(domain, 0, num_test)
        if test_generator and hasattr(
            test_generator, "generate_multi_condition_scenarios"
        ):
            # Generate scenarios with num_conditions (this creates the fixed key pool)
            cold_start_scenarios = test_generator.generate_multi_condition_scenarios(
                num_training=6,  # Generate 6 unique keys (one per symbol)
                num_test=num_test,
                num_conditions=num_conditions,
                test_mode="matched",
            )

            # Extract the condition keys from generated scenarios
            for scenario in cold_start_scenarios:
                mc = scenario.get("multi_condition", {})
                if mc.get("condition_key"):
                    training_condition_keys.add(mc["condition_key"])

            logger.info(
                f"   Generated {len(training_condition_keys)} unique condition keys"
            )
            logger.info(f"   Keys: {list(training_condition_keys)[:3]}...")
        else:
            logger.warning("   ⚠️ Could not generate cold start keys - using fallback")

    # ═══════════════════════════════════════════════════════════════════════════
    # GENERATE TEST SCENARIOS USING SCENARIO GENERATOR (not hardcoded templates)
    # ═══════════════════════════════════════════════════════════════════════════
    # Use the scenario generator's proper methods to build test scenarios from
    # learned rule keys. This keeps all generation logic in one place.
    # Skip for compositional mode (we already have pre-generated test scenarios).
    # ═══════════════════════════════════════════════════════════════════════════
    if not is_compositional_mode and not is_semantic_compositional:
        matched_test_scenarios = []
        random_test_scenarios = []
        learned_rules = {}  # Full learned rules dict

    # Load learned rules for post-training test generation
    if not is_compositional_mode and not is_semantic_compositional:
        try:
            learned_rules_path = DATA_DIR / "precept_learned_rules.json"
            if learned_rules_path.exists():
                with open(learned_rules_path) as f:
                    learned_rules = json.load(f)
                logger.info(f"  📚 Found {len(learned_rules)} learned rules")
            else:
                logger.warning("  ⚠️ No learned rules file found")
        except Exception as e:
            logger.warning(f"  ⚠️ Could not load learned rules: {e}")

    # ═══════════════════════════════════════════════════════════════════════════
    # GENERATE MATCHED TEST SCENARIOS (exact condition keys from learned rules)
    # ═══════════════════════════════════════════════════════════════════════════
    # CRITICAL FIX: Filter learned rules to ONLY include keys from the current
    # training session's key pool. This prevents stale rules from previous runs
    # from contaminating the test scenarios.
    #
    # For proper MATCHED mode testing:
    # 1. We use the fixed key pool from training (K unique composite keys)
    # 2. Only rules learned FOR THOSE SPECIFIC KEYS should be used in test
    # 3. This ensures 100% match when all expected rules are learned
    # Skip for compositional mode (we use pre-generated composite test scenarios).
    # ═══════════════════════════════════════════════════════════════════════════
    if (
        not is_compositional_mode
        and not is_semantic_compositional
        and test_mode in ("matched", "both")
    ):
        if (
            learned_rules
            and training_condition_keys
            and not getattr(args, "preserve_learned_rules", False)
        ):
            # FILTER: Only use learned rules whose keys are in current training session
            filtered_learned_rules = {
                k: v for k, v in learned_rules.items() if k in training_condition_keys
            }

            # Log filtering info
            num_total = len(learned_rules)
            num_filtered = len(filtered_learned_rules)
            num_excluded = num_total - num_filtered
            if num_excluded > 0:
                logger.info("  🔍 MATCHED mode: Filtered learned rules")
                logger.info(f"     Total in file: {num_total}")
                logger.info(f"     In current training pool: {num_filtered}")
                logger.info(f"     Excluded (stale/old): {num_excluded}")
            else:
                logger.info(
                    f"  ✅ MATCHED mode: All {num_filtered} learned rules match training pool"
                )

            if not filtered_learned_rules:
                logger.warning("  ⚠️ No learned rules match current training keys!")
                logger.info(
                    f"     Training keys: {list(training_condition_keys)[:3]}..."
                )
                logger.info(f"     Learned keys: {list(learned_rules.keys())[:3]}...")
                matched_test_scenarios = test_scenarios.copy()
            else:
                # Use the domain-specific scenario generator's method
                test_generator = _get_domain_test_generator(domain, num_train, num_test)
                if test_generator and hasattr(
                    test_generator, "generate_test_from_learned_keys"
                ):
                    matched_test_scenarios = test_generator.generate_test_from_learned_keys(
                        learned_rule_keys=filtered_learned_rules,  # Use FILTERED rules
                        num_test=num_test,
                        mode="matched",
                        seed=seed if seed else 42,
                        all_training_keys=list(
                            training_condition_keys
                        ),  # Pass training pool
                    )
                    logger.info(
                        "     (using EXACT condition keys for true O(1) lookup test)"
                    )
                else:
                    logger.warning(
                        f"  ⚠️ Domain {domain} does not support generate_test_from_learned_keys"
                    )
                    matched_test_scenarios = test_scenarios.copy()
        elif learned_rules:
            if getattr(args, "preserve_learned_rules", False) and training_condition_keys:
                logger.info(
                    "  🔒 MATCHED mode: Using learned rules directly "
                    "(preserve_learned_rules enabled)"
                )
            # Learned rules exist but no training keys tracked - use all (backward compat)
            test_generator = _get_domain_test_generator(domain, num_train, num_test)
            if test_generator and hasattr(
                test_generator, "generate_test_from_learned_keys"
            ):
                matched_test_scenarios = test_generator.generate_test_from_learned_keys(
                    learned_rule_keys=learned_rules,
                    num_test=num_test,
                    mode="matched",
                    seed=seed if seed else 42,
                )
                logger.info("     (using ALL learned rules - no training key tracking)")
            else:
                matched_test_scenarios = test_scenarios.copy()
        elif training_condition_keys:
            # COLD START: No learned rules but we have pre-generated condition keys
            # Generate test scenarios using the cold start keys
            logger.info(
                "  🧊 COLD START: Using pre-generated condition keys for testing"
            )
            test_generator = _get_domain_test_generator(domain, num_train, num_test)
            if test_generator and hasattr(
                test_generator, "generate_test_from_learned_keys"
            ):
                # Create empty learned_rules dict but use all training keys
                matched_test_scenarios = test_generator.generate_test_from_learned_keys(
                    learned_rule_keys={},  # Empty - no rules learned yet
                    num_test=num_test,
                    mode="matched",
                    seed=seed if seed else 42,
                    all_training_keys=list(training_condition_keys),
                )
                logger.info(
                    f"     Generated {len(matched_test_scenarios)} test scenarios"
                )
                logger.info(
                    f"     Using {len(training_condition_keys)} unique condition keys"
                )
            else:
                matched_test_scenarios = test_scenarios.copy()
        else:
            # No learned rules and no cold start keys - fallback to generator
            logger.warning(
                "  ⚠️ No learned rules or condition keys - using default test scenarios"
            )
            matched_test_scenarios = test_scenarios.copy()

    # ═══════════════════════════════════════════════════════════════════════════
    # GENERATE RANDOM TEST SCENARIOS (partial overlap with ALL training keys)
    # ═══════════════════════════════════════════════════════════════════════════
    # RANDOM mode uses ALL training condition keys, not just learned rules:
    # - Learned rules (success via error recovery)
    # - First-try successes (no rule learned - agent guessed correctly)
    # - Exhausted retries (failed completely - no rule learned)
    # This provides broader coverage of the training distribution.
    # This uses the generate_test_from_learned_keys method available in ALL domains
    # Skip for compositional mode (we use pre-generated composite test scenarios).
    # ═══════════════════════════════════════════════════════════════════════════
    if (
        not is_compositional_mode
        and not is_semantic_compositional
        and test_mode in ("random", "both")
    ):
        if learned_rules or training_condition_keys:
            # Use the domain-specific scenario generator's method with ALL training keys
            test_generator = _get_domain_test_generator(domain, num_train, num_test)

            if test_generator and hasattr(
                test_generator, "generate_test_from_learned_keys"
            ):
                # Log training key coverage
                num_learned = len(learned_rules)
                num_all_training = len(training_condition_keys)
                num_unlearned = num_all_training - num_learned
                logger.info("  📊 Training key breakdown:")
                logger.info(f"     Learned rules: {num_learned}")
                logger.info(f"     First-try/exhausted: {num_unlearned}")
                logger.info(f"     Total training keys: {num_all_training}")

                random_test_scenarios = test_generator.generate_test_from_learned_keys(
                    learned_rule_keys=learned_rules,
                    num_test=num_test,
                    mode="random",
                    seed=(seed if seed else 42) + 100,  # Different seed from MATCHED
                    all_training_keys=list(training_condition_keys),
                )
                logger.info("     (partial matches to test Tier 2/3 similarity search)")
            else:
                logger.warning(
                    f"  ⚠️ Domain {domain} does not support generate_test_from_learned_keys"
                )
                random_test_scenarios = test_scenarios.copy()
        else:
            # Fallback: use default test scenarios
            logger.warning(
                "  ⚠️ No learned rules or training keys - using default scenarios"
            )
            random_test_scenarios = test_scenarios.copy()

    # Determine which modes to run
    # Skip if already set by compositional or semantic compositional mode
    if not is_compositional_mode and not is_semantic_compositional:
        modes_to_run = []
        if test_mode == "matched":
            modes_to_run = [("matched", matched_test_scenarios)]
            logger.info("\n📊 Test Mode: MATCHED (O(1) Lookup Test)")
            logger.info(f"   Filtered test scenarios: {len(matched_test_scenarios)}")
        elif test_mode == "random":
            modes_to_run = [("random", random_test_scenarios)]
            logger.info("\n📊 Test Mode: RANDOM (Generalization Test)")
            logger.info(f"   Random test scenarios: {len(random_test_scenarios)}")
        else:  # "both"
            modes_to_run = [
                ("matched", matched_test_scenarios),
                ("random", random_test_scenarios),
            ]
            logger.info("\n📊 Test Mode: BOTH (Running Matched + Random)")
            logger.info(f"   MATCHED scenarios: {len(matched_test_scenarios)}")
            logger.info(f"   RANDOM scenarios:  {len(random_test_scenarios)}")

    # For backward compatibility, set filtered_test_scenarios to first mode
    filtered_test_scenarios = modes_to_run[0][1] if modes_to_run else test_scenarios

    # ═══════════════════════════════════════════════════════════════════════════
    # PHASE 2: TESTING (Run All Test Modes)
    # ═══════════════════════════════════════════════════════════════════════════
    # Run tests for each mode (matched, random, or both).
    # Collect results and metrics for each mode separately.
    # ═══════════════════════════════════════════════════════════════════════════

    # ═══════════════════════════════════════════════════════════════════════════
    # CRITICAL FIX: Reload learned rules from disk before testing
    # ═══════════════════════════════════════════════════════════════════════════
    # During training, rules are saved to disk but the MCP server's in-memory
    # cache may have stale or conflicting state. Reloading ensures Tier 1 lookups
    # use the LATEST rules from training.
    # ═══════════════════════════════════════════════════════════════════════════
    try:
        reload_result = await precept_agent.mcp_client.reload_learned_rules()
        logger.info(f"  🔄 {reload_result}")
    except Exception as e:
        logger.warning(f"  ⚠️ Could not reload rules: {e}")

    total_test_scenarios = sum(len(scenarios) for _, scenarios in modes_to_run)
    if test_mode == "both":
        details = [
            "🎯 DUAL-MODE TESTING: Running MATCHED + RANDOM",
            f"   MATCHED scenarios: {len(matched_test_scenarios)} (O(1) lookup)",
            f"   RANDOM scenarios:  {len(random_test_scenarios)} (generalization)",
        ]
    else:
        details = [
            f"🎯 TESTING: {test_mode.upper()} mode",
            f"   Test scenarios: {len(filtered_test_scenarios)}",
        ]
    display.print_phase_header(
        "2: TESTING - Unified Fair Comparison",
        total_test_scenarios,
        details,
    )

    test_start_time = time_module.time()

    # Create semaphore for controlled concurrency
    semaphore = asyncio.Semaphore(max_workers)

    async def run_with_semaphore(agent, task_str, agent_name, **kwargs):
        """Run agent task with semaphore to limit concurrency."""
        async with semaphore:
            try:
                return await agent.run_task(task_str, **kwargs)
            except Exception as e:
                return {"success": False, "steps": 0, "error": str(e)}

    # ═══════════════════════════════════════════════════════════════════════════
    # RUN TESTS FOR EACH MODE
    # ═══════════════════════════════════════════════════════════════════════════

    # Store results for each mode
    mode_results = {}  # mode_name -> {precept: [], expel: [], fr: [], elapsed: float}
    dual_mode_results = None  # Will contain metrics for both modes if test_mode="both"

    for mode_name, mode_scenarios in modes_to_run:
        logger.info(f"\n{'=' * 70}")
        logger.info(
            f"🧪 [{mode_name.upper()}] Testing on {len(mode_scenarios)} scenarios"
        )
        logger.info("=" * 70)

        # Run tests for this mode using helper function
        p_results, e_results, fr_results_mode, elapsed = await run_test_phase(
            scenarios=mode_scenarios,
            precept_agent=precept_agent,
            expel_baseline=expel_baseline,
            full_reflexion=full_reflexion,
            tracer=tracer,
            mode_name=mode_name,
        )

        # Store results
        mode_results[mode_name] = {
            "precept": p_results,
            "expel": e_results,
            "full_reflexion": fr_results_mode,
            "elapsed": elapsed,
            "scenarios": mode_scenarios,
        }

        # Print mode summary
        p_successes = sum(1 for r in p_results if r.get("success", False))
        e_successes = sum(1 for r in e_results if r.get("success", False))
        fr_successes = sum(1 for r in fr_results_mode if r.get("success", False))
        total = len(mode_scenarios)

        logger.info(f"\n  ✅ [{mode_name.upper()}] Testing complete:")
        logger.info(
            f"     PRECEPT:        {p_successes}/{total} ({100 * p_successes / max(1, total):.1f}%)"
        )
        logger.info(
            f"     ExpeL:          {e_successes}/{total} ({100 * e_successes / max(1, total):.1f}%)"
        )
        logger.info(
            f"     Full Reflexion: {fr_successes}/{total} ({100 * fr_successes / max(1, total):.1f}%)"
        )

    test_elapsed = time_module.time() - test_start_time

    # Use first mode's results as the primary results (for backward compatibility)
    primary_mode = modes_to_run[0][0]
    precept_results = mode_results[primary_mode]["precept"]
    expel_results = mode_results[primary_mode]["expel"]
    full_reflexion_results = mode_results[primary_mode]["full_reflexion"]

    # Print combined summary
    logger.info(f"\n  ⏱️  Total testing time: {test_elapsed:.1f}s")

    # If running both modes, extract metrics for each and create dual_mode_results
    if test_mode == "both":
        # Get stats for metric calculation
        full_reflexion_stats_temp = full_reflexion.get_stats()
        expel_stats_temp = expel_baseline.get_stats()
        precept_stats_temp = precept_agent.get_llm_reasoning_stats()

        dual_mode_results = {}
        for mode_name in mode_results:
            mode_data = mode_results[mode_name]
            dual_mode_results[mode_name] = extract_mode_metrics(
                precept_results=mode_data["precept"],
                expel_results=mode_data["expel"],
                fr_results=mode_data["full_reflexion"],
                scenarios=mode_data["scenarios"],
                num_train=num_train,
                precept_stats=precept_stats_temp,
                expel_stats=expel_stats_temp,
                fr_stats=full_reflexion_stats_temp,
            )
            dual_mode_results[mode_name]["elapsed_seconds"] = round(
                mode_data["elapsed"], 2
            )
            dual_mode_results[mode_name]["num_scenarios"] = len(mode_data["scenarios"])

        # Print comparison table
        logger.info("\n" + "=" * 85)
        logger.info("📊 DUAL-MODE COMPARISON: MATCHED vs RANDOM")
        logger.info("=" * 85)
        logger.info("")
        logger.info("┌" + "─" * 83 + "┐")
        logger.info(
            f"│ {'Metric':<25} │ {'MATCHED (O(1))':<25} │ {'RANDOM (Generalize)':<25} │"
        )
        logger.info("├" + "─" * 83 + "┤")

        for agent_name in ["precept", "expel", "full_reflexion"]:
            matched_p1 = (
                dual_mode_results["matched"][agent_name]["first_try_success_rate"] * 100
            )
            random_p1 = (
                dual_mode_results["random"][agent_name]["first_try_success_rate"] * 100
            )
            display_name = {
                "precept": "PRECEPT",
                "expel": "ExpeL",
                "full_reflexion": "Full Reflexion",
            }[agent_name]
            logger.info(
                f"│ {display_name + ' P₁':<25} │ {matched_p1:>24.1f}% │ {random_p1:>24.1f}% │"
            )

        logger.info("├" + "─" * 83 + "┤")

        for agent_name in ["precept", "expel", "full_reflexion"]:
            matched_steps = dual_mode_results["matched"][agent_name]["avg_steps"]
            random_steps = dual_mode_results["random"][agent_name]["avg_steps"]
            display_name = {
                "precept": "PRECEPT",
                "expel": "ExpeL",
                "full_reflexion": "Full Reflexion",
            }[agent_name]
            logger.info(
                f"│ {display_name + ' Avg Steps':<25} │ {matched_steps:>25.2f} │ {random_steps:>25.2f} │"
            )

        logger.info("└" + "─" * 83 + "┘")

    # ═══════════════════════════════════════════════════════════════════════════
    # CALCULATE COMPREHENSIVE METRICS (FAIR COMPARISON: PRECEPT vs Full Reflexion)
    # ═══════════════════════════════════════════════════════════════════════════

    # llm_stats = llm_baseline.get_stats()  # UNFAIR - commented out
    # reflexion_stats = reflexion_baseline.get_stats()  # UNFAIR - commented out
    full_reflexion_stats = full_reflexion.get_stats()
    expel_stats = expel_baseline.get_stats()
    precept_stats = precept_agent.get_llm_reasoning_stats()
    pruning_stats = precept_agent.get_pruning_stats()
    # compass_stats used via precept_agent.get_prompt_stats() for has_evolved

    # Calculate metrics for each agent (FAIR COMPARISON only)
    # Pass test_scenarios and num_train for new metrics (first-try, learning efficiency, cold-start)

    # # LLM Baseline (UNFAIR - does not train)
    # llm_metrics = calculate_metrics(
    #     llm_results,
    #     llm_stats,
    #     "LLM Baseline",
    #     scenarios=test_scenarios,
    #     num_train=num_train,
    # )
    # llm_metrics.has_cross_task_learning = False
    # llm_metrics.has_cross_episode_memory = False
    # llm_metrics.has_within_task_reflection = False

    # # Adapted Reflexion (UNFAIR - does not train)
    # reflexion_metrics = calculate_metrics(
    #     reflexion_results,
    #     reflexion_stats,
    #     "Reflexion",
    #     scenarios=test_scenarios,
    #     num_train=num_train,
    # )
    # reflexion_metrics.has_within_task_reflection = True
    # reflexion_metrics.reflections_generated = reflexion_stats.get(
    #     "reflections_generated", 0
    # )

    # Full Reflexion (unified - same scenarios as all agents)
    full_reflexion_metrics = calculate_metrics(
        full_reflexion_results,
        full_reflexion_stats,
        "Full Reflexion",
        scenarios=filtered_test_scenarios,
        num_train=num_train,
    )
    full_reflexion_metrics.has_cross_episode_memory = True
    full_reflexion_metrics.reflections_generated = full_reflexion_stats.get(
        "reflections_generated", 0
    )
    full_reflexion_metrics.reflections_reused = full_reflexion_stats.get(
        "reflections_reused", 0
    )

    # ExpeL (unified - same scenarios as all agents)
    expel_metrics = calculate_metrics(
        expel_results,
        expel_stats,
        "ExpeL",
        scenarios=filtered_test_scenarios,
        num_train=num_train,
    )
    expel_metrics.has_cross_episode_memory = True
    expel_metrics.insights_extracted = expel_stats.get("insights_extracted", 0)
    expel_metrics.insights_retrieved = expel_stats.get("insights_retrieved", 0)
    expel_metrics.insights_applied = expel_stats.get("insights_applied", 0)

    # PRECEPT (unified - same scenarios as all agents)
    precept_metrics = calculate_metrics(
        precept_results,
        precept_stats,
        "PRECEPT",
        scenarios=filtered_test_scenarios,
        num_train=num_train,
    )
    precept_metrics.has_cross_task_learning = True
    precept_metrics.has_deterministic_pruning = True
    precept_metrics.has_compass_evolution = precept_agent.get_prompt_stats().get(
        "has_evolved", False
    )
    precept_metrics.rules_learned = len(precept_agent.learning_events)

    # ═══════════════════════════════════════════════════════════════════════════
    # DISPLAY COMPREHENSIVE RESULTS
    # ═══════════════════════════════════════════════════════════════════════════

    logger.info("")
    logger.info("=" * 85)
    logger.info(
        f"📊 FAIR COMPARISON RESULTS - {domain.upper()} DOMAIN (PRECEPT vs Full Reflexion vs ExpeL)"
    )
    logger.info("=" * 85)

    # ═══════════════════════════════════════════════════════════════════════════
    # FAIR COMPARISON: PRECEPT vs Full Reflexion vs ExpeL (all train on T_train)
    # ═══════════════════════════════════════════════════════════════════════════

    logger.info("")
    logger.info("┌" + "─" * 83 + "┐")
    logger.info(
        f"│ {'Agent':<20} │ {'Success Rate':>12} │ {'Avg Steps':>10} │ {'First-Try':>10} │ {'Rules/Ref':>10} │"
    )
    logger.info("├" + "─" * 83 + "┤")
    logger.info(
        f"│ {'Full Reflexion':<20} │ {full_reflexion_metrics.success_rate * 100:>11.1f}% │ {full_reflexion_metrics.avg_steps:>10.2f} │ {full_reflexion_metrics.first_try_success_rate * 100:>9.1f}% │ {full_reflexion_metrics.reflections_generated:>10} │"
    )
    logger.info(
        f"│ {'ExpeL':<20} │ {expel_metrics.success_rate * 100:>11.1f}% │ {expel_metrics.avg_steps:>10.2f} │ {expel_metrics.first_try_success_rate * 100:>9.1f}% │ {expel_metrics.insights_extracted:>10} │"
    )
    logger.info(
        f"│ {'PRECEPT':<20} │ {precept_metrics.success_rate * 100:>11.1f}% │ {precept_metrics.avg_steps:>10.2f} │ {precept_metrics.first_try_success_rate * 100:>9.1f}% │ {precept_metrics.rules_learned:>10} │"
    )
    logger.info("└" + "─" * 83 + "┘")

    display.print_learning_capabilities(precept_metrics.has_compass_evolution)

    display.print_learning_statistics(
        rules_learned=len(precept_agent.learning_events),
        dumb_retries=pruning_stats.get("dumb_retries_prevented", 0),
        hard_constraints=pruning_stats.get("hard_constraints", 0),
        soft_constraints=pruning_stats.get("soft_constraints", 0),
        fr_reflections=full_reflexion_stats.get("reflections_generated", 0),
        fr_reused=full_reflexion_stats.get("reflections_reused", 0),
        fr_memory_size=sum(FullReflexionBaselineAgent.get_memory_stats().values()),
        ref_reflections=0,  # Adapted Reflexion not used
        ref_lessons=0,  # Adapted Reflexion not used
    )

    # ═══════════════════════════════════════════════════════════════════════════
    # KEY INSIGHTS (FAIR COMPARISON)
    # ═══════════════════════════════════════════════════════════════════════════

    # Calculate deltas (FAIR comparison only)
    precept_vs_fullref = (
        precept_metrics.success_rate - full_reflexion_metrics.success_rate
    )
    steps_vs_fullref = full_reflexion_metrics.avg_steps - precept_metrics.avg_steps
    first_try_advantage = (
        precept_metrics.first_try_success_rate
        - full_reflexion_metrics.first_try_success_rate
    )

    logger.info("")
    logger.info("=" * 85)
    logger.info("🎯 KEY INSIGHTS (FAIR COMPARISON: PRECEPT vs Full Reflexion)")
    logger.info("=" * 85)
    logger.info(
        f"  PRECEPT Success Rate Advantage: {precept_vs_fullref * 100:+.1f} pp over Full Reflexion"
    )
    logger.info(f"  PRECEPT Steps Saved: {steps_vs_fullref:+.1f} steps per task")
    logger.info(
        f"  PRECEPT First-Try Advantage: {first_try_advantage * 100:+.1f} pp over Full Reflexion"
    )
    logger.info("")
    logger.info("  Both agents trained on same T_train tasks (fair comparison)")
    logger.info(
        "  PRECEPT uses structured rules; Full Reflexion uses verbal reflections"
    )

    # Determine winner (FAIR comparison)
    all_metrics = [
        full_reflexion_metrics,
        precept_metrics,
    ]
    winner = max(all_metrics, key=lambda m: (m.success_rate, -m.avg_steps))

    display.print_winner(winner.name, winner.success_rate, winner.avg_steps)

    # Get server stats
    logger.info("📊 Server Statistics...")
    try:
        server_stats = await precept_agent.mcp_client.get_server_stats()
        logger.info(server_stats)
    except Exception:
        pass

    # Show evolved prompt
    logger.info("")
    logger.info("=" * 85)
    logger.info("🧬 PRECEPT EVOLVED PROMPT (Used in Testing Phase)")
    logger.info("=" * 85)
    try:
        evolved_prompt = await precept_agent.mcp_client.get_evolved_prompt(
            include_rules=True
        )
        if evolved_prompt:
            logger.info(evolved_prompt[:800])
            if len(evolved_prompt) > 800:
                logger.info(f"... [{len(evolved_prompt) - 800} more characters]")
    except Exception as e:
        logger.warning(f"Could not retrieve evolved prompt: {e}")

    # ═══════════════════════════════════════════════════════════════════════════
    # GET CONFLICT RESOLUTION STATS (PRECEPT's Unique Capability)
    # ═══════════════════════════════════════════════════════════════════════════
    conflict_stats = {}
    try:
        # Use the dedicated conflict resolution stats tool
        conflict_stats_raw = await precept_agent.mcp_client.call_tool(
            "get_conflict_resolution_stats", {}
        )
        if conflict_stats_raw:
            import json as json_module

            conflict_stats = json_module.loads(conflict_stats_raw)
            logger.info("")
            logger.info("=" * 85)
            logger.info(
                "🔀 CONFLICT RESOLUTION STATISTICS (PRECEPT's Unique Capability)"
            )
            logger.info("=" * 85)
            if conflict_stats.get("status") == "active":
                summary = conflict_stats.get("summary", {})
                outcomes = conflict_stats.get("resolution_outcomes", {})
                logger.info(
                    f"  Conflicts Detected:  {summary.get('conflicts_detected', 0)}"
                )
                logger.info(
                    f"  Conflicts Resolved:  {summary.get('conflicts_resolved', 0)}"
                )
                logger.info(f"  Static Wins:         {outcomes.get('static_wins', 0)}")
                logger.info(f"  Dynamic Wins:        {outcomes.get('dynamic_wins', 0)}")
                logger.info(f"  Merges:              {outcomes.get('merges', 0)}")
            else:
                logger.info(f"  Status: {conflict_stats.get('status', 'unknown')}")
    except Exception as e:
        logger.debug(f"Could not retrieve conflict stats: {e}")

    # ═══════════════════════════════════════════════════════════════════════════
    # SAVE EXPERIMENT RESULTS (FAIR COMPARISON)
    # ═══════════════════════════════════════════════════════════════════════════

    # Create placeholder metrics for unfair baselines (not run, but needed for save function)
    # These are marked with 0 values to indicate they were not tested
    dummy_llm_metrics = AgentMetrics(
        name="LLM Baseline (NOT RUN - unfair)",
        success_rate=0.0,
        total_successes=0,
        total_tasks=0,
        avg_steps=0.0,
        llm_calls=0,
        llm_calls_per_task=0.0,
        llm_accuracy=0.0,
    )
    dummy_reflexion_metrics = AgentMetrics(
        name="Reflexion (NOT RUN - unfair)",
        success_rate=0.0,
        total_successes=0,
        total_tasks=0,
        avg_steps=0.0,
        llm_calls=0,
        llm_calls_per_task=0.0,
        llm_accuracy=0.0,
    )

    results_file = save_experiment_results(
        domain=domain,
        train_scenarios=train_scenarios,
        test_scenarios=test_scenarios,
        llm_metrics=dummy_llm_metrics,  # Placeholder (not run)
        reflexion_metrics=dummy_reflexion_metrics,  # Placeholder (not run)
        full_reflexion_metrics=full_reflexion_metrics,
        expel_metrics=expel_metrics,  # ExpeL baseline (Zhao et al., 2023)
        precept_metrics=precept_metrics,
        precept_stats=precept_stats,
        pruning_stats=pruning_stats,
        train_elapsed=train_elapsed,
        test_elapsed=test_elapsed,
        concurrent_training=concurrent_training,
        concurrent_testing=concurrent_testing,
        seed=seed,
        enable_static_knowledge=enable_static_knowledge,
        conflict_stats=conflict_stats,
        max_retries=max_retries,
        test_mode=test_mode,
        dual_mode_results=dual_mode_results,  # Contains metrics for both modes when test_mode="both"
    )

    # ═══════════════════════════════════════════════════════════════════════════
    # SAVE DETAILED EXECUTION TRACES (if enabled)
    # ═══════════════════════════════════════════════════════════════════════════
    if detailed_logs:
        import datetime as dt

        if trace_file is None:
            timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
            trace_file = str(DATA_DIR / f"trace_{domain}_{timestamp}.json")

        # Add final experiment summary to tracer metadata
        tracer.log.metadata["experiment_results_file"] = results_file
        tracer.log.metadata["train_elapsed"] = train_elapsed
        tracer.log.metadata["test_elapsed"] = test_elapsed
        tracer.log.metadata["precept_success_rate"] = precept_metrics.success_rate
        tracer.log.metadata["full_reflexion_success_rate"] = (
            full_reflexion_metrics.success_rate
        )

        tracer.save_to_file(trace_file)
        logger.info(f"📝 Detailed execution traces saved to: {trace_file}")

    # Gracefully disconnect all agents (with timeout and error suppression)
    # Only PRECEPT and Full Reflexion were run
    await graceful_disconnect(
        [precept_agent, full_reflexion],  # Only fair comparison agents
        timeout=2.0,
    )

    # Save cumulative stats (FAIR COMPARISON only)
    stats["total_runs"] += 1
    stats["total_tasks"] += len(train_scenarios) + len(test_scenarios)
    stats["precept_successes"] += precept_metrics.total_successes
    # llm_baseline and reflexion not run - skip their stats
    # stats["llm_baseline_successes"] = (
    #     stats.get("llm_baseline_successes", 0) + llm_metrics.total_successes
    # )
    # stats["reflexion_successes"] = (
    #     stats.get("reflexion_successes", 0) + reflexion_metrics.total_successes
    # )
    stats["full_reflexion_successes"] = (
        stats.get("full_reflexion_successes", 0)
        + full_reflexion_metrics.total_successes
    )
    save_stats(stats)

    display.print_conclusion(domain)

    return results_file


def show_available_domains():
    """Display all available domains and their scenarios."""
    logger.info("")
    logger.info("╔" + "═" * 82 + "╗")
    logger.info("║" + " " * 25 + "AVAILABLE PRECEPT DOMAINS" + " " * 32 + "║")
    logger.info("╚" + "═" * 82 + "╝")

    for domain in list_available_domains():
        icon = DOMAIN_ICONS.get(domain, "🔬")
        strategy = get_domain_strategy(domain)
        baseline = get_baseline_strategy(domain)
        scenario_gen = SCENARIO_GENERATORS.get(domain)
        num_scenarios = len(scenario_gen()) if scenario_gen else 0

        display.print_domain_info(
            domain=domain,
            icon=icon,
            strategy_name=strategy.__class__.__name__,
            baseline_name=baseline.__class__.__name__,
            actions=strategy.get_available_actions(),
            options=baseline.get_available_options(),
            num_scenarios=num_scenarios,
        )

    logger.info("")
    logger.info("Usage:")
    logger.info(
        "    uv run examples/precept_autogen_mcp_full.py --domain <domain_name>"
    )
    logger.info("    uv run examples/precept_autogen_mcp_full.py --domain coding")
    logger.info(
        "    uv run examples/precept_autogen_mcp_full.py --domain devops --train 8 --test 4"
    )


def main():
    """Main entry point with argument parsing."""
    # Reset async client state to ensure clean event loop binding
    # Critical for parallel experiments that may reuse the same process
    reset_async_client()

    parser = argparse.ArgumentParser(
        description="PRECEPT vs 3 Baselines Multi-Domain Test Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
═══════════════════════════════════════════════════════════════════════════════
                              QUICK START EXAMPLES
═══════════════════════════════════════════════════════════════════════════════

  # Basic usage (sequential training, 6 train / 4 test)
  uv run examples/precept_autogen_mcp_full.py --domain logistics

  # Custom train/test split with reproducible seed
  uv run examples/precept_autogen_mcp_full.py -d logistics --train 8 --test 4 --seed 42

  # Concurrent training ("Tesla Fleet" mode) + concurrent testing
  uv run examples/precept_autogen_mcp_full.py -d logistics -ct -tw 4 -c -w 4 --seed 42

  # List all available domains
  uv run examples/precept_autogen_mcp_full.py --list

  # Verbose output with log file
  uv run examples/precept_autogen_mcp_full.py -d logistics --verbose --log-file run.log

═══════════════════════════════════════════════════════════════════════════════
                              AVAILABLE DOMAINS
═══════════════════════════════════════════════════════════════════════════════

  logistics   - Black swan shipping scenarios (port closures, customs)
  coding      - Code execution with runtime errors
  devops      - Infrastructure deployment failures
  finance     - Market volatility and compliance
  booking     - Reservation system edge cases
  integration - API integration failures

═══════════════════════════════════════════════════════════════════════════════
                            CONCURRENCY OPTIONS
═══════════════════════════════════════════════════════════════════════════════

  TESTING PHASE (--concurrent, -c):
    Sequential (default)  : Agents run one after another
    Concurrent (-c)       : Agents run in parallel per test task
      -w 1  Sequential (same as no -c)
      -w 2  2 agents at a time (balanced)
      -w 3  3 agents at a time
      -w 4  All 4 agents in parallel (fastest, default)

  TRAINING PHASE (--concurrent-training, -ct):
    Sequential (default)  : Tasks train one after another (best learning)
    Concurrent (-ct)      : "Tesla Fleet" mode - parallel training
      -tw 2  2 training tasks in parallel (default)
      -tw 4  4 training tasks in parallel (~4x speedup)
      -tw 6  6 training tasks in parallel (fastest)

  PER-AGENT INTERNAL (--agent-workers, -aw):
    Controls parallel operations WITHIN each agent (LLM calls, MCP tools)
      -aw 1  Sequential (slowest, safest for API rate limits)
      -aw 3  3 parallel ops (default, balanced)
      -aw 5  5 parallel ops (fastest, may hit rate limits)

═══════════════════════════════════════════════════════════════════════════════
                              BASELINE AGENTS
═══════════════════════════════════════════════════════════════════════════════

  1. LLMBaselineAgent       Adapted ReAct - error feedback only
  2. ReflexionBaselineAgent Adapted Reflexion - within-task reflection
  3. FullReflexionBaseline  Full Reflexion paper - cross-episode memory
  4. PRECEPT                Full learning stack (rules, COMPASS, pruning)

═══════════════════════════════════════════════════════════════════════════════
                            REPRODUCIBILITY
═══════════════════════════════════════════════════════════════════════════════

  Use --seed for reproducible experiments:
    uv run examples/precept_autogen_mcp_full.py --seed 42 --train 8 --test 4

  This ensures the same scenarios are generated across runs, enabling
  fair comparison between sequential and concurrent training modes.
        """,
    )

    parser.add_argument(
        "--domain",
        "-d",
        type=str,
        default="logistics",
        choices=["logistics", "coding", "devops", "finance", "booking", "integration"],
        help="Domain to test (default: logistics)",
    )

    parser.add_argument(
        "--train",
        type=int,
        default=6,
        help="Number of TRAINING tasks (PRECEPT + Full Reflexion learn). Default: 6",
    )

    parser.add_argument(
        "--test",
        type=int,
        default=4,
        help="Number of TEST tasks (4-way comparison). Default: 4",
    )

    parser.add_argument(
        "--tasks",
        "-t",
        type=int,
        default=None,
        help="[DEPRECATED] Total tasks. Use --train and --test instead.",
    )

    parser.add_argument(
        "--list", "-l", action="store_true", help="List all available domains and exit"
    )

    parser.add_argument(
        "--concurrent",
        "-c",
        action="store_true",
        help="Enable concurrent testing (faster). "
        "Runs agents in parallel per test task. "
        "Disables continuous learning during testing for fair snapshot comparison.",
    )

    parser.add_argument(
        "--workers",
        "-w",
        type=int,
        default=4,
        choices=[1, 2, 3, 4],
        help="Number of agents to run in parallel when --concurrent is enabled. "
        "1=sequential, 2=pairs, 3=three at a time, 4=all parallel. Default: 4",
    )

    parser.add_argument(
        "--agent-workers",
        "-aw",
        type=int,
        default=3,
        choices=[1, 2, 3, 4, 5],
        help="Max concurrent internal operations PER AGENT (LLM calls, MCP tools). "
        "Lower values help with API rate limits. Default: 3",
    )

    parser.add_argument(
        "--concurrent-training",
        "-ct",
        action="store_true",
        help="Enable concurrent training ('Tesla Fleet' mode). "
        "Multiple training tasks run in parallel, sharing Evo-Memory. "
        "GEPA consolidation runs as batch job after training. "
        "~Nx speedup where N = --training-workers.",
    )

    parser.add_argument(
        "--training-workers",
        "-tw",
        type=int,
        default=2,
        choices=[1, 2, 3, 4, 5, 6],
        help="Number of training tasks to run in parallel when --concurrent-training. "
        "Higher = faster but more API calls. Default: 2",
    )

    parser.add_argument(
        "--seed",
        "-s",
        type=int,
        default=None,
        help="Random seed for reproducible experiments. "
        "When set, the same scenarios are generated across runs, "
        "enabling fair comparison between sequential and concurrent training. "
        "Example: --seed 42",
    )

    # ═══════════════════════════════════════════════════════════════════════════
    # STATIC KNOWLEDGE CONFIGURATION (Simple on/off flag)
    # ═══════════════════════════════════════════════════════════════════════════
    parser.add_argument(
        "--static-knowledge",
        "-sk",
        action="store_true",
        dest="static_knowledge",
        default=True,
        help="Enable static knowledge for ALL agents (default: enabled). "
        "Both PRECEPT and baselines access the same static facts. "
        "PRECEPT's advantage: cutting-edge CONFLICT RESOLUTION.",
    )
    parser.add_argument(
        "--no-static-knowledge",
        "-nsk",
        action="store_false",
        dest="static_knowledge",
        help="Disable static knowledge for ALL agents. "
        "Pure dynamic learning comparison (no pre-loaded facts).",
    )

    # ═══════════════════════════════════════════════════════════════════════════
    # DATA RESET OPTIONS (for multi-phase experiments)
    # ═══════════════════════════════════════════════════════════════════════════
    parser.add_argument(
        "--preserve-learned-rules",
        action="store_true",
        help="Preserve PRECEPT learned rules + domain mappings on startup. "
        "Use for test-only runs that rely on prior training (e.g., rule drift).",
    )

    # ═══════════════════════════════════════════════════════════════════════════
    # MAX RETRIES CONFIGURATION (Fair comparison budget)
    # ═══════════════════════════════════════════════════════════════════════════
    parser.add_argument(
        "--max-retries",
        "-mr",
        type=int,
        default=None,
        help="Maximum number of retries allowed (same for all agents for fair comparison). "
        "Default: 2. Options: 1 (near first-try), 2 (balanced), 4 (lenient). "
        "Total attempts = 1 initial + max_retries.",
    )

    # ═══════════════════════════════════════════════════════════════════════════
    # MULTI-CONDITION CONFIGURATION (for ablation studies)
    # ═══════════════════════════════════════════════════════════════════════════
    parser.add_argument(
        "--num-conditions",
        "-nc",
        type=int,
        default=1,
        choices=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        help="Number of conditions per scenario (1-10). "
        "Default: 1 (single-condition, backward compatible). "
        "Higher values exponentially challenge baselines: "
        "N=3 → 8 states, N=5 → 32 states, N=10 → 1024 states. "
        "PRECEPT maintains ~85%% effectiveness, baselines degrade to ~50%%×0.75^(N-1).",
    )

    # COMPOSITIONAL GENERALIZATION: Separate train/test conditions
    parser.add_argument(
        "--train-num-conditions",
        "-tnc",
        type=int,
        default=None,
        choices=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        help="Number of conditions for TRAINING phase only. "
        "If set, overrides --num-conditions for training. "
        "Use with --test-num-conditions for compositional generalization: "
        "e.g., --train-num-conditions 1 --test-num-conditions 2 "
        "trains on atomic conditions (A, B) and tests on composites (A+B).",
    )

    parser.add_argument(
        "--test-num-conditions",
        "-xnc",
        type=int,
        default=None,
        choices=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        help="Number of conditions for TESTING phase only. "
        "If set, overrides --num-conditions for testing. "
        "Use with --train-num-conditions for compositional generalization.",
    )

    parser.add_argument(
        "--test-mode",
        "-tm",
        type=str,
        default="matched",
        choices=["matched", "random", "both"],
        help="Test scenario generation mode. "
        "matched: Test scenarios REUSE condition keys from training (O(1) lookup test). "
        "random: Test scenarios generate NEW random condition keys (generalization test). "
        "both: Run BOTH modes and report results for each. "
        "Default: matched.",
    )

    parser.add_argument(
        "--semantic-compositional",
        "-sc",
        action="store_true",
        help="Enable SEMANTIC compositional testing where composite solutions are "
        "DERIVABLE from atomic precepts. Unlike Black Swan CSPs where solutions "
        "are arbitrary, this mode creates scenarios where knowing A→X and B→Y "
        "allows the LLM to reason that A+B→X_Y. This enables P₁ > 0%% on novel "
        "combinations through compositional reasoning.",
    )

    parser.add_argument(
        "--beta",
        type=int,
        default=1,
        help="Repetitions per atomic condition during training (default: 1). "
        "With beta=3, each atomic condition is trained 3 times, giving ~95%% "
        "probability of learning all atomic precepts.",
    )

    parser.add_argument(
        "--filter-by-learned",
        "-fbl",
        action="store_true",
        help="PURE COMPOSITIONAL TEST: After training, only test on composite "
        "conditions where ALL constituent atomic precepts were LEARNED. "
        "This isolates the true compositional generalization capability by "
        "ensuring we only test A+B when we know PRECEPT learned both A and B.",
    )

    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose (DEBUG) logging output.",
    )

    parser.add_argument(
        "--log-file",
        type=str,
        default=None,
        help="Path to log file. When set, logs are also written to file in JSON format.",
    )

    parser.add_argument(
        "--detailed-logs",
        "-dl",
        action="store_true",
        help="Enable detailed execution tracing. Saves step-by-step logs for each task.",
    )

    parser.add_argument(
        "--trace-file",
        type=str,
        default=None,
        help="Path to save detailed execution traces (JSON). Default: data/trace_{domain}_{timestamp}.json",
    )

    parser.add_argument(
        "--condition-enhanced-retrieval",
        "-cer",
        action="store_true",
        help="ABLATION: Enable condition-enhanced retrieval for baselines. "
        "ExpeL: Includes condition codes in vector embeddings/search AND shows them in insights. "
        "Reflexion: Filters reflections by condition match AND shows them in reflections. "
        "Tests if 'soft' similarity can match PRECEPT's 'hard' hash lookup.",
    )

    parser.add_argument(
        "--hybrid-retrieval",
        "-hr",
        action="store_true",
        help="ABLATION: Enable hybrid BM25 + semantic retrieval for ALL agents. "
        "PRECEPT: Uses LangChain EnsembleRetriever for Tier 2 (BM25 + semantic). "
        "ExpeL: Combines BM25 keyword matching with vector similarity. "
        "Reflexion: Uses BM25 scoring on reflection text combined with condition matching. "
        "Tests if hybrid retrieval (RRF fusion) improves retrieval for condition codes.",
    )

    parser.add_argument(
        "--improved-baselines",
        "-ib",
        action="store_true",
        help="IMPROVED BASELINES: Enable metadata-based filtering for ExpeL and Reflexion. "
        "This stores conditions as structured metadata in the vector store and uses "
        "ChromaDB's 'where' clause to pre-filter by condition overlap BEFORE semantic search. "
        "This gives baselines O(1)-like lookup for matching conditions while still using "
        "vector similarity for ranking. More effective than embedding conditions in text. "
        "Not faithful to original papers but tests if structured metadata improves baselines.",
    )

    # ═══════════════════════════════════════════════════════════════════════════
    # MODEL AND EMBEDDING CONFIGURATION (for model ablation studies)
    # ═══════════════════════════════════════════════════════════════════════════
    parser.add_argument(
        "--model",
        "-m",
        type=str,
        default="gpt-4o-mini",
        choices=["gpt-4o-mini", "gpt-4o", "gpt-4-turbo", "gpt-3.5-turbo"],
        help="LLM model to use for ALL agents. "
        "Default: gpt-4o-mini (cost-effective). "
        "gpt-4o: More powerful, tests if stronger reasoning helps baselines. "
        "gpt-4-turbo: Larger context, more expensive. "
        "gpt-3.5-turbo: Fastest, cheapest baseline.",
    )

    parser.add_argument(
        "--embedding-model",
        "-em",
        type=str,
        default="text-embedding-3-small",
        choices=[
            "text-embedding-3-small",
            "text-embedding-3-large",
            "text-embedding-ada-002",
        ],
        help="Embedding model for vector search (ALL agents). "
        "Default: text-embedding-3-small (cost-effective, 1536 dims). "
        "text-embedding-3-large: More powerful (3072 dims), tests if better embeddings help baselines. "
        "text-embedding-ada-002: Legacy model.",
    )

    args = parser.parse_args()

    # ═══════════════════════════════════════════════════════════════════════════
    # LOGGING SETUP
    # ═══════════════════════════════════════════════════════════════════════════
    log_level = "DEBUG" if args.verbose else "INFO"
    setup_logging(level=log_level, log_file=args.log_file)

    # ═══════════════════════════════════════════════════════════════════════════
    # REPRODUCIBILITY: Set random seed if provided
    # ═══════════════════════════════════════════════════════════════════════════
    if args.seed is not None:
        import random

        random.seed(args.seed)
        logger.info(f"🎲 Random seed set to {args.seed} for reproducible experiments")

    if args.list:
        show_available_domains()
        return

    # Handle deprecated --tasks argument
    if args.tasks is not None:
        logger.warning("--tasks is deprecated. Use --train and --test instead.")
        logger.warning(
            f"   Splitting {args.tasks} tasks: {args.tasks // 2 + 2} train, {args.tasks // 2} test"
        )
        args.train = args.tasks // 2 + 2
        args.test = args.tasks // 2

    # Log static knowledge configuration
    if args.static_knowledge:
        logger.info("📚 Static knowledge: ENABLED for ALL agents")
    else:
        logger.info("📚 Static knowledge: DISABLED (--no-static-knowledge)")

    # Log max_retries setting
    if args.max_retries is not None:
        logger.info(f"🔄 Max retries: {args.max_retries} (from CLI)")

    # Determine train/test conditions (support compositional generalization)
    train_num_conditions = args.train_num_conditions or args.num_conditions
    test_num_conditions = args.test_num_conditions or args.num_conditions

    # Log multi-condition mode and test mode
    if train_num_conditions != test_num_conditions:
        # COMPOSITIONAL GENERALIZATION MODE
        logger.info("🧬 COMPOSITIONAL GENERALIZATION MODE:")
        logger.info(
            f"   Train: {train_num_conditions}-condition scenarios (learn atomic precepts)"
        )
        logger.info(
            f"   Test: {test_num_conditions}-condition scenarios (generalize to composites)"
        )
        logger.info(
            f"   Generalization: O({train_num_conditions}) → O(2^{test_num_conditions}) = {2**test_num_conditions} combinations"
        )
    elif args.num_conditions > 1:
        logger.info(
            f"🔢 Multi-condition mode: {args.num_conditions} conditions per scenario"
        )
        logger.info(
            f"   State space: 2^{args.num_conditions} = {2**args.num_conditions} possible states"
        )
        logger.info("   All scenarios (including fleet learning) use multi-conditions")

    # Log test mode
    test_mode_desc = {
        "matched": "MATCHED (O(1) exact lookup - test keys reuse training keys)",
        "random": "RANDOM (generalization - test keys are new random combinations)",
        "both": "BOTH (run matched and random modes separately)",
    }
    logger.info(f"🧪 Test mode: {test_mode_desc.get(args.test_mode, args.test_mode)}")

    # Run the test for the selected domain
    # Wrap in try-except to suppress MCP shutdown errors
    try:
        asyncio.run(
            run_domain_test(
                domain=args.domain,
                num_train=args.train,
                num_test=args.test,
                concurrent_testing=args.concurrent,
                concurrent_training=args.concurrent_training,
                max_workers=args.workers,
                training_workers=args.training_workers,
                agent_internal_workers=args.agent_workers,
                seed=args.seed,
                enable_static_knowledge=args.static_knowledge,
                max_retries=args.max_retries,
                # Enable detailed LLM logs when --verbose (-v) OR --detailed-logs is set
                detailed_logs=args.detailed_logs or args.verbose,
                trace_file=args.trace_file,
                num_conditions=args.num_conditions,
                train_num_conditions=train_num_conditions,
                test_num_conditions=test_num_conditions,
                test_mode=args.test_mode,
                args=args,  # Pass full args for other flags
            )
        )
    except (RuntimeError, asyncio.CancelledError) as e:
        # MCP library may raise these during shutdown - harmless
        if "cancel scope" not in str(e) and "Event loop is closed" not in str(e):
            raise


if __name__ == "__main__":
    # Suppress "unhandled exception during asyncio.run() shutdown" messages
    # These are printed by asyncio and are harmless cleanup errors from MCP
    import logging

    logging.getLogger("asyncio").setLevel(logging.CRITICAL)
    main()
