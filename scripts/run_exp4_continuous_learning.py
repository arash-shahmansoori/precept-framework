#!/usr/bin/env python3
"""
Experiment 4: Continuous Learning / Cross-Episode Learning

PURPOSE:
    Test PRECEPT's ability to learn from failures DURING TESTING.
    This experiment uses minimal training (β=1) so that learning happens
    during sequential test execution, demonstrating PRECEPT's unique
    cross-episode memory advantage over baselines.

KEY INSIGHT:
    - Training: β=1 (only 1 encounter per key) → partial learning
    - Testing: 4 encounters per key → observe learning curve
    - PRECEPT should show improving P₁ over repeated condition_keys
    - Baselines should show flat P₁ (no improvement)

EXPERIMENT DESIGN:
    - E unique condition_keys (domain-specific: finance=6, logistics=4, etc.)
    - --train E (β=1): Each key seen once during training
    - --test E×4: Each key seen 4 times during testing
    - --max-retries 2: Low retry budget to prevent brute-force success
    - -ct -tw 4: CONCURRENT training (4 workers for speed)
    - -w 1: SEQUENTIAL testing (critical for cross-episode learning)
    - --test-mode matched: Same keys in testing as training

METRICS TRACKED:
    1. P₁ (first-try success) by encounter number (1st, 2nd, 3rd, 4th)
    2. Pₜ (overall success) by encounter number
    3. Avg steps by encounter number
    4. Learning curve visualization data
    5. Per-condition_key progression
    6. Steps saved on subsequent encounters
    7. Rules learned during training vs testing
    8. Partial progress (failed options) tracking

OUTPUT:
    - data/publication_results/exp5_continuous_learning/
    - continuous_learning_results.json: All metrics and learning curves
    - learning_curve_analysis.md: Human-readable report
    - Per-seed logs and results

Usage:
    python scripts/run_exp5_continuous_learning.py [--quick] [--domain DOMAIN]

    --quick: Run with 3 seeds instead of 10
    --domain: Specific domain (default: finance)
"""

import argparse
import json
import re
import shutil
import subprocess
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    from scripts.utils.pct_bounds import bounded_pct_ci
except ImportError:
    def bounded_pct_ci(mean_pct, ci_pct, lower=0.0, upper=100.0):
        return max(0.0, min(ci_pct, mean_pct - lower, upper - mean_pct))

try:
    import numpy as np
    from scipy import stats

    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    np = None
    stats = None

try:
    from tqdm import tqdm

    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False

    def tqdm(iterable, desc=None, total=None, **kwargs):
        if desc:
            print(f"\n{desc}")
        for i, item in enumerate(iterable):
            if total:
                print(f"  Progress: {i + 1}/{total}", end="\r")
            yield item
        print()


# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Configuration
PUBLICATION_SEEDS = [42, 123, 456, 789, 999, 2024, 3141, 1337, 8888, 7777]
QUICK_SEEDS = [42, 123, 456]
VERY_QUICK_SEEDS = [42]  # Single seed for validation
PUBLICATION_DOMAINS = ["integration", "logistics"]

# ═══════════════════════════════════════════════════════════════════════════════
# DOMAIN E VALUES: Single-Condition vs Multi-Condition
# ═══════════════════════════════════════════════════════════════════════════════
#
# E (error types / condition keys) differs based on experiment mode:
#
# SINGLE-CONDITION (--num-conditions 1):
#   E = total distinct error codes across ALL categories in the domain config
#   Each error code is its own condition key (e.g., "FIN-058")
#
# MULTI-CONDITION (--num-conditions > 1):
#   E = number of unique BASE ENTITIES that generate composite condition keys
#   Multiple conditions are combined into one key (e.g., "FIN-058+R-482+M-LOW")
#   The base entity count determines how many unique composite keys exist
#
# For β=1: train = E (one encounter per key)
# For testing: test = E × encounters_per_key
# ═══════════════════════════════════════════════════════════════════════════════

# Single-condition E values: Total distinct error types per domain
# Single-condition E values: Count of unique error codes ACTUALLY USED by generators
# Note: Config files have more error types, but generators only use primary sources
SINGLE_CONDITION_E = {
    "finance": 6,  # VOLATILE_SYMBOLS only
    "logistics": 4,  # BLOCKED_PORTS only
    "coding": 5,  # BLOCKED_PACKAGES only
    "devops": 5,  # STUCK_STACKS only
    "booking": 17,  # BLOCKED_FLIGHTS (all 17)
    "integration": 6,  # OAUTH_SOURCES only
}

# Multi-condition E values: Unique base entities for composite key generation
# finance:     6 volatile symbols (GME, BTC-USD, MEME-COIN, AMC, ETH-USD, TSLA)
# Multi-condition E values: Same as single-condition (generators use same key pools)
MULTI_CONDITION_E = {
    "finance": 6,  # VOLATILE_SYMBOLS only
    "logistics": 4,  # BLOCKED_PORTS only
    "coding": 5,  # BLOCKED_PACKAGES only
    "devops": 5,  # STUCK_STACKS only
    "booking": 17,  # BLOCKED_FLIGHTS (all 17)
    "integration": 6,  # OAUTH_SOURCES only
}

# Fixed experiment parameters
# ═══════════════════════════════════════════════════════════════════════════════
# COMPOSITE-CONDITION MODE: Measure continuous learning under harder keys
# This experiment tests PRECEPT's ability to adapt during deployment (test time)
# when each scenario contains 5 conditions per key.
# For targeted compositional generalization analysis, see Experiment 6.
# ═══════════════════════════════════════════════════════════════════════════════
NUM_CONDITIONS = 5  # Composite-condition mode (5 conditions per scenario)
MAX_RETRIES = 4  # Sufficient retry budget for harder domains (e.g., integration)
ENCOUNTERS_PER_KEY = 4  # How many times each key is tested (learning curve)
DEFAULT_BETA = 1  # Default coverage factor: each key seen β times during training

# Select E values based on experiment mode
E_VALUES = MULTI_CONDITION_E if NUM_CONDITIONS > 1 else SINGLE_CONDITION_E

# Build domain configurations dynamically
DOMAIN_CONFIGS = {
    name: {"E": E_VALUES[name], "num_conditions": NUM_CONDITIONS}
    for name in E_VALUES.keys()
}


def get_experiment_params(domain: str, beta: int = DEFAULT_BETA) -> Dict[str, int]:
    """
    Get experiment parameters for a given domain.

    Args:
        domain: Domain name (finance, logistics, etc.)
        beta: Coverage factor - how many times each key is seen during training
              β=0: Zero training (cold start experiment)
              β=1: Each key seen once during training (default)
              β=2+: Multiple exposures per key during training

    Returns dict with:
        - num_keys: Number of unique condition keys (E)
        - train_size: Training tasks (β × E)
        - test_size: Test tasks (E × encounters_per_key)
        - num_conditions: Conditions per scenario
    """
    config = DOMAIN_CONFIGS.get(domain, DOMAIN_CONFIGS["finance"])
    num_keys = config["E"]

    return {
        "num_keys": num_keys,
        "train_size": beta
        * num_keys,  # β=0 → zero training, β=1 → one encounter per key
        "test_size": num_keys * ENCOUNTERS_PER_KEY,  # 4 encounters per key
        "num_conditions": config["num_conditions"],
        "max_retries": MAX_RETRIES,
        "beta": beta,
        "encounters_per_key": ENCOUNTERS_PER_KEY,
    }


def clean_data_directory():
    """Remove ALL persisted data to ensure clean experiment.

    This must remove every file that could leak state between seeds:
      - ChromaDB vector stores (chroma_*)
      - PRECEPT JSON state (precept_*.json)
      - Baseline memory files (full_reflexion_memory.json, expel_insights.json)
      - Stale experiment_results_*.json (prevents picking up old results)
    """
    data_dir = PROJECT_ROOT / "data"

    # 1. Remove ChromaDB vector stores
    paths_to_clean = [
        data_dir / "chroma_precept",
        data_dir / "chroma_static_knowledge",
        data_dir / "chroma_expel",
        data_dir / "chroma_full_reflexion",  # Full Reflexion's reflection store
    ]

    for path in paths_to_clean:
        if path.exists():
            shutil.rmtree(path)
            print(f"  🧹 Removed: {path.name}/")

    # 2. Remove all PRECEPT JSON state files
    for json_file in data_dir.glob("precept_*.json"):
        json_file.unlink()
        print(f"  🧹 Removed: {json_file.name}")

    # 3. Clean baseline persistence files (Full Reflexion + ExpeL JSON)
    # These must be cleaned between seed runs but are preserved between
    # train/test phases within the same run (via --preserve-learned-rules)
    baseline_files = [
        data_dir / "full_reflexion_memory.json",
        data_dir / "expel_insights.json",
    ]
    for bf in baseline_files:
        if bf.exists():
            bf.unlink()
            print(f"    🧹 Cleaned: {bf.name}")

    # 4. Remove stale experiment_results_*.json to prevent picking up old results
    stale_results = list(data_dir.glob("experiment_results_*.json"))
    if stale_results:
        for f in stale_results:
            f.unlink()
        print(f"  🧹 Removed {len(stale_results)} stale experiment_results_*.json files")



def parse_log_for_test_details(log_path: Path) -> Dict[str, List[Dict]]:
    """
    Parse log file to extract per-task test details for each agent.

    Parses the new detailed log format:
    📊 [MATCHED Test X/Y] key=... | PRECEPT: ✓ (P₁=Y, steps=2) | ExpeL: ✗ (P₁=N) | ...

    Returns dict: {agent_name: [{condition_key, success, first_try, steps, task_num}, ...]}
    """
    if not log_path.exists():
        return {}

    results = {
        "precept": [],
        "expel": [],
        "full_reflexion": [],
    }

    with open(log_path) as f:
        content = f.read()

    # ═══════════════════════════════════════════════════════════════════════
    # NEW DETAILED FORMAT PARSER
    # Format: 📊 [MATCHED Test X/Y] key=ABC... | PRECEPT: ✓ (P₁=Y, steps=2) | ...
    # ═══════════════════════════════════════════════════════════════════════
    pattern = re.compile(
        r"📊 \[(?:MATCHED|RANDOM) Test (\d+)/\d+\] "
        r"key=([A-Z0-9+\-]+)\.\.\. \| "
        r"PRECEPT: ([✓✗]) \(P₁=([YN]), steps=(\d+)\) \| "
        r"ExpeL: ([✓✗]) \(P₁=([YN])\) \| "
        r"FullRef: ([✓✗]) \(P₁=([YN])\)"
    )

    for match in pattern.finditer(content):
        task_num = int(match.group(1))
        condition_key = match.group(2)

        # PRECEPT
        precept_success = match.group(3) == "✓"
        precept_first_try = match.group(4) == "Y"
        precept_steps = int(match.group(5))
        results["precept"].append(
            {
                "task_num": task_num,
                "condition_key": condition_key,
                "success": precept_success,
                "first_try": precept_first_try,
                "steps": precept_steps,
            }
        )

        # ExpeL
        expel_success = match.group(6) == "✓"
        expel_first_try = match.group(7) == "Y"
        results["expel"].append(
            {
                "task_num": task_num,
                "condition_key": condition_key,
                "success": expel_success,
                "first_try": expel_first_try,
                "steps": 2 if expel_first_try else (4 if expel_success else 6),
            }
        )

        # Full Reflexion
        fr_success = match.group(8) == "✓"
        fr_first_try = match.group(9) == "Y"
        results["full_reflexion"].append(
            {
                "task_num": task_num,
                "condition_key": condition_key,
                "success": fr_success,
                "first_try": fr_first_try,
                "steps": 2 if fr_first_try else (4 if fr_success else 6),
            }
        )

    # ═══════════════════════════════════════════════════════════════════════
    # FALLBACK: Old format parser for backward compatibility
    # ═══════════════════════════════════════════════════════════════════════
    if not results["precept"]:
        # Track current test context
        current_condition_key = None
        current_task_num = 0
        in_test_phase = False

        lines = content.split("\n")
        for i, line in enumerate(lines):
            # Detect test phase start
            if "═══ TESTING PHASE" in line or "TEST PHASE" in line:
                in_test_phase = True
                continue

            if not in_test_phase:
                continue

            # Look for task markers (MATCHED Test or similar)
            task_match = re.search(r"\[(?:MATCHED|RANDOM) Test (\d+)/", line)
            if task_match:
                current_task_num = int(task_match.group(1))

            # Look for 3-TIER HYBRID FETCH condition_key (PRECEPT)
            if "3-TIER HYBRID FETCH: condition_key=" in line:
                match = re.search(r"condition_key=([A-Z0-9+\-\.]+)", line)
                if match:
                    current_condition_key = match.group(1).rstrip(".")

            # Look for condition_key in other formats
            if "condition_key" in line.lower() and current_condition_key is None:
                match = re.search(r"condition_key[=:]?\s*([A-Z0-9+\-]+)", line)
                if match:
                    current_condition_key = match.group(1)

            # Detect PRECEPT results
            if "PRECEPT" in line:
                if "first_try=True" in line or "first-try success" in line.lower():
                    results["precept"].append(
                        {
                            "task_num": current_task_num,
                            "condition_key": current_condition_key,
                            "success": True,
                            "first_try": True,
                            "steps": 2,
                        }
                    )
                    current_condition_key = None
                elif "✅" in line and "SUCCESS" in line:
                    first_try = "first_try=True" in "".join(
                        lines[max(0, i - 3) : i + 3]
                    )
                    results["precept"].append(
                        {
                            "task_num": current_task_num,
                            "condition_key": current_condition_key,
                            "success": True,
                            "first_try": first_try,
                            "steps": 2 if first_try else 4,
                        }
                    )
                    current_condition_key = None
                elif "❌" in line and "FAILED" in line:
                    results["precept"].append(
                        {
                            "task_num": current_task_num,
                            "condition_key": current_condition_key,
                            "success": False,
                            "first_try": False,
                            "steps": 6,
                        }
                    )
                    current_condition_key = None

            # Detect ExpeL results (same pattern)
            if "ExpeL" in line or "expel" in line.lower():
                if "✅" in line and "SUCCESS" in line:
                    first_try = "attempt 1" in line or "first" in line.lower()
                    results["expel"].append(
                        {
                            "task_num": current_task_num,
                            "condition_key": current_condition_key,
                            "success": True,
                            "first_try": first_try,
                            "steps": 2 if first_try else 4,
                        }
                    )
            elif "❌" in line and "FAILED" in line:
                results["expel"].append(
                    {
                        "task_num": current_task_num,
                        "condition_key": current_condition_key,
                        "success": False,
                        "first_try": False,
                        "steps": 6,
                    }
                )

        # Detect Full Reflexion results
        if "Full Reflexion" in line or "full_reflexion" in line.lower():
            if "✅" in line and "SUCCESS" in line:
                first_try = "attempt 1" in line or "first" in line.lower()
                results["full_reflexion"].append(
                    {
                        "task_num": current_task_num,
                        "condition_key": current_condition_key,
                        "success": True,
                        "first_try": first_try,
                        "steps": 2 if first_try else 4,
                    }
                )
            elif "❌" in line and "FAILED" in line:
                results["full_reflexion"].append(
                    {
                        "task_num": current_task_num,
                        "condition_key": current_condition_key,
                        "success": False,
                        "first_try": False,
                        "steps": 6,
                    }
                )

    return results


def parse_results_file(results_path: Path) -> Dict[str, Any]:
    """Parse experiment results JSON file for aggregated metrics."""
    if not results_path.exists():
        return {}

    with open(results_path) as f:
        return json.load(f)


def copy_learning_artifacts(seed: int, output_dir: Path) -> Dict[str, Any]:
    """
    Copy and parse PRECEPT learning artifacts (rules, procedures, partial progress).

    Returns dict with artifact contents for analysis.
    """
    data_dir = PROJECT_ROOT / "data"
    artifacts = {}

    artifact_files = [
        ("learned_rules", "precept_learned_rules.json"),
        ("procedures", "precept_procedures.json"),
        ("partial_progress", "precept_partial_progress.json"),
        ("experiences", "precept_experiences.json"),
    ]

    for name, filename in artifact_files:
        src = data_dir / filename
        if src.exists():
            dest = output_dir / f"seed{seed}_{filename}"
            shutil.copy(src, dest)

            with open(src) as f:
                try:
                    artifacts[name] = json.load(f)
                except json.JSONDecodeError:
                    artifacts[name] = {}

    return artifacts


def compute_encounter_metrics(
    test_results: List[Dict], num_keys: int, encounters_per_key: int
) -> Dict[str, Any]:
    """
    Compute metrics grouped by encounter number.

    For sequential testing with repeated condition_keys, groups results
    by which encounter (1st, 2nd, 3rd, 4th) of each key and computes:
    - P₁ (first-try success rate)
    - Pₜ (overall success rate)
    - Avg steps
    """
    # Group by condition_key
    by_key = defaultdict(list)
    for result in test_results:
        key = result.get("condition_key", "unknown")
        by_key[key].append(result)

    # Compute metrics by encounter number
    metrics_by_encounter = defaultdict(
        lambda: {
            "first_try_successes": 0,
            "successes": 0,
            "total": 0,
            "steps": [],
        }
    )

    for key, encounters in by_key.items():
        for enc_num, result in enumerate(encounters):
            if enc_num >= encounters_per_key:
                continue  # Only count up to expected encounters

            enc_key = enc_num + 1  # 1-indexed
            metrics_by_encounter[enc_key]["total"] += 1
            metrics_by_encounter[enc_key]["steps"].append(result.get("steps", 0))

            if result.get("success", False):
                metrics_by_encounter[enc_key]["successes"] += 1
            if result.get("first_try", False):
                metrics_by_encounter[enc_key]["first_try_successes"] += 1

    # Convert to final metrics
    encounter_stats = {}
    for enc in range(1, encounters_per_key + 1):
        m = metrics_by_encounter.get(enc, {})
        total = m.get("total", 0)
        steps = m.get("steps", [])

        if total > 0:
            encounter_stats[f"encounter_{enc}"] = {
                "p1": m.get("first_try_successes", 0) / total,
                "pt": m.get("successes", 0) / total,
                "avg_steps": sum(steps) / len(steps) if steps else 0,
                "first_try_successes": m.get("first_try_successes", 0),
                "successes": m.get("successes", 0),
                "total": total,
            }
        else:
            encounter_stats[f"encounter_{enc}"] = {
                "p1": 0,
                "pt": 0,
                "avg_steps": 0,
                "first_try_successes": 0,
                "successes": 0,
                "total": 0,
            }

    return {
        "by_encounter": encounter_stats,
        "by_key": {k: len(v) for k, v in by_key.items()},
        "total_tasks": len(test_results),
    }


def aggregate_learning_curves(
    all_runs: List[Dict], num_encounters: int = 4
) -> Dict[str, Any]:
    """
    Aggregate learning curves across all runs with statistical analysis.

    Returns metrics per agent per encounter with mean, std, and 95% CI.
    """
    if not SCIPY_AVAILABLE:
        print("⚠️ scipy not available, statistical analysis will be limited")
        import statistics as pystats

    agents = ["precept", "expel", "full_reflexion"]
    learning_curves = {}

    for agent in agents:
        # Collect per-encounter values across runs
        p1_by_enc = defaultdict(list)
        pt_by_enc = defaultdict(list)
        steps_by_enc = defaultdict(list)

        for run in all_runs:
            if run is None:
                continue

            agent_metrics = run.get(agent, {})
            by_encounter = agent_metrics.get("by_encounter", {})

            for enc in range(1, num_encounters + 1):
                enc_key = f"encounter_{enc}"
                enc_data = by_encounter.get(enc_key, {})

                if enc_data.get("total", 0) > 0:
                    p1_by_enc[enc].append(enc_data.get("p1", 0))
                    pt_by_enc[enc].append(enc_data.get("pt", 0))
                    steps_by_enc[enc].append(enc_data.get("avg_steps", 0))

        # Compute statistics
        curve = {}
        for enc in range(1, num_encounters + 1):
            p1_vals = p1_by_enc.get(enc, [])
            pt_vals = pt_by_enc.get(enc, [])
            steps_vals = steps_by_enc.get(enc, [])

            if not p1_vals:
                continue

            n = len(p1_vals)

            if SCIPY_AVAILABLE:
                t_critical = stats.t.ppf(0.975, df=n - 1) if n > 1 else 0

                def compute_ci(vals):
                    arr = np.array(vals)
                    std = np.std(arr, ddof=1) if len(arr) > 1 else 0
                    return t_critical * std / np.sqrt(n) if n > 1 else 0

                curve[f"encounter_{enc}"] = {
                    "p1_mean": float(np.mean(p1_vals)),
                    "p1_std": float(np.std(p1_vals, ddof=1)) if n > 1 else 0,
                    "p1_ci_95": compute_ci(p1_vals),
                    "pt_mean": float(np.mean(pt_vals)),
                    "pt_std": float(np.std(pt_vals, ddof=1)) if n > 1 else 0,
                    "pt_ci_95": compute_ci(pt_vals),
                    "steps_mean": float(np.mean(steps_vals)) if steps_vals else 0,
                    "steps_std": float(np.std(steps_vals, ddof=1))
                    if len(steps_vals) > 1
                    else 0,
                    "steps_ci_95": compute_ci(steps_vals) if steps_vals else 0,
                    "n": n,
                }
            else:
                import statistics as pystats

                curve[f"encounter_{enc}"] = {
                    "p1_mean": pystats.mean(p1_vals),
                    "p1_std": pystats.stdev(p1_vals) if n > 1 else 0,
                    "p1_ci_95": 0,
                    "pt_mean": pystats.mean(pt_vals),
                    "pt_std": pystats.stdev(pt_vals) if n > 1 else 0,
                    "pt_ci_95": 0,
                    "steps_mean": pystats.mean(steps_vals) if steps_vals else 0,
                    "steps_std": pystats.stdev(steps_vals)
                    if len(steps_vals) > 1
                    else 0,
                    "steps_ci_95": 0,
                    "n": n,
                }

        learning_curves[agent] = curve

    return learning_curves


def compute_learning_improvement(curves: Dict[str, Any]) -> Dict[str, Any]:
    """
    Compute learning improvement metrics for each agent.

    Returns improvement from 1st to 4th encounter.
    """
    improvements = {}

    for agent in ["precept", "expel", "full_reflexion"]:
        agent_curve = curves.get(agent, {})
        enc1 = agent_curve.get("encounter_1", {})
        enc4 = agent_curve.get("encounter_4", {})

        improvements[agent] = {
            "p1_improvement_pp": (enc4.get("p1_mean", 0) - enc1.get("p1_mean", 0))
            * 100,
            "pt_improvement_pp": (enc4.get("pt_mean", 0) - enc1.get("pt_mean", 0))
            * 100,
            "steps_saved": enc1.get("steps_mean", 0) - enc4.get("steps_mean", 0),
            "enc1_p1": enc1.get("p1_mean", 0),
            "enc4_p1": enc4.get("p1_mean", 0),
        }

    return improvements


def compute_statistical_tests(
    all_runs: List[Dict], num_encounters: int
) -> Dict[str, Any]:
    """Compute statistical significance tests for publication quality.

    Returns:
        Dictionary with:
        - per_encounter_tests: PRECEPT vs baselines at each encounter
        - improvement_tests: significance of learning improvement (enc1 → enc4)
        - overall_advantage: effect sizes and p-values
    """
    if not SCIPY_AVAILABLE:
        return {"error": "scipy not available for statistical tests"}

    agents = ["precept", "expel", "full_reflexion"]

    # Collect raw P₁ values per encounter per agent
    p1_by_agent_enc: Dict[str, Dict[int, List[float]]] = {
        agent: defaultdict(list) for agent in agents
    }

    for run in all_runs:
        if run is None:
            continue
        for agent in agents:
            agent_metrics = run.get(agent, {})
            by_encounter = agent_metrics.get("by_encounter", {})
            for enc in range(1, num_encounters + 1):
                enc_key = f"encounter_{enc}"
                enc_data = by_encounter.get(enc_key, {})
                if enc_data.get("total", 0) > 0:
                    p1_by_agent_enc[agent][enc].append(enc_data.get("p1", 0))

    def paired_ttest(v1: List[float], v2: List[float]) -> Dict[str, float]:
        """Compute paired t-test with Cohen's d effect size."""
        if len(v1) < 2 or len(v2) < 2 or len(v1) != len(v2):
            return {"t_stat": 0.0, "p_value": 1.0, "cohens_d": 0.0, "n": 0}
        arr1, arr2 = np.array(v1), np.array(v2)
        t_stat, p_value = stats.ttest_rel(arr1, arr2)
        diff = arr1 - arr2
        std_diff = np.std(diff, ddof=1)
        cohens_d = float(np.mean(diff) / std_diff) if std_diff > 0 else 0.0
        return {
            "t_stat": float(t_stat),
            "p_value": float(p_value),
            "cohens_d": cohens_d,
            "n": len(v1),
        }

    def one_sample_ttest(values: List[float]) -> Dict[str, float]:
        """Test if improvement is significantly > 0."""
        if len(values) < 2:
            return {"t_stat": 0.0, "p_value": 1.0, "n": 0}
        arr = np.array(values)
        t_stat, p_value = stats.ttest_1samp(arr, 0)
        # One-tailed: is improvement > 0?
        p_one_tailed = p_value / 2 if t_stat > 0 else 1 - p_value / 2
        return {
            "t_stat": float(t_stat),
            "p_value_two_tailed": float(p_value),
            "p_value_one_tailed": float(p_one_tailed),
            "n": len(values),
        }

    def apply_bonferroni(test: Dict[str, Any], p_key: str, n_tests: int) -> None:
        """Annotate a test dict with Bonferroni-corrected p-values."""
        if p_key not in test:
            return
        raw_p = float(test.get(p_key, 1.0))
        corrected_p = min(1.0, raw_p * max(1, n_tests))
        test[f"{p_key}_bonferroni"] = corrected_p
        test["bonferroni_n_tests"] = int(max(1, n_tests))
        test["significant_raw"] = raw_p < 0.05
        test["significant_bonferroni"] = corrected_p < 0.05

    # 1. Per-encounter comparisons: PRECEPT vs baselines
    per_encounter_tests = {}
    for enc in range(1, num_encounters + 1):
        precept_vals = p1_by_agent_enc["precept"].get(enc, [])
        expel_vals = p1_by_agent_enc["expel"].get(enc, [])
        fr_vals = p1_by_agent_enc["full_reflexion"].get(enc, [])

        per_encounter_tests[f"encounter_{enc}"] = {
            "precept_vs_expel": paired_ttest(precept_vals, expel_vals),
            "precept_vs_full_reflexion": paired_ttest(precept_vals, fr_vals),
        }
    per_encounter_m = num_encounters * 2
    for enc_tests in per_encounter_tests.values():
        apply_bonferroni(enc_tests["precept_vs_expel"], "p_value", per_encounter_m)
        apply_bonferroni(
            enc_tests["precept_vs_full_reflexion"], "p_value", per_encounter_m
        )

    # 2. Learning improvement tests (enc1 → enc4 significant?)
    improvement_tests = {}
    for agent in agents:
        enc1_vals = p1_by_agent_enc[agent].get(1, [])
        enc4_vals = p1_by_agent_enc[agent].get(num_encounters, [])

        if len(enc1_vals) == len(enc4_vals) and len(enc1_vals) > 1:
            # Paired comparison: did P₁ improve from enc1 to enc4?
            improvement = [e4 - e1 for e1, e4 in zip(enc1_vals, enc4_vals)]
            improvement_tests[agent] = {
                "paired_test": paired_ttest(enc4_vals, enc1_vals),
                "improvement_significance": one_sample_ttest(improvement),
                "mean_improvement_pp": float(np.mean(improvement)) * 100,
            }
            apply_bonferroni(improvement_tests[agent]["paired_test"], "p_value", len(agents))
            apply_bonferroni(
                improvement_tests[agent]["improvement_significance"],
                "p_value_one_tailed",
                len(agents),
            )
        else:
            improvement_tests[agent] = {"error": "insufficient paired data"}

    # 3. Final encounter advantage (PRECEPT vs baselines at last encounter)
    final_enc = num_encounters
    precept_final = p1_by_agent_enc["precept"].get(final_enc, [])
    expel_final = p1_by_agent_enc["expel"].get(final_enc, [])
    fr_final = p1_by_agent_enc["full_reflexion"].get(final_enc, [])

    final_advantage = {
        "precept_vs_expel": paired_ttest(precept_final, expel_final),
        "precept_vs_full_reflexion": paired_ttest(precept_final, fr_final),
        "precept_mean_p1": float(np.mean(precept_final)) if precept_final else 0,
        "expel_mean_p1": float(np.mean(expel_final)) if expel_final else 0,
        "full_reflexion_mean_p1": float(np.mean(fr_final)) if fr_final else 0,
        "advantage_vs_expel_pp": (
            (np.mean(precept_final) - np.mean(expel_final)) * 100
            if precept_final and expel_final
            else 0
        ),
        "advantage_vs_fr_pp": (
            (np.mean(precept_final) - np.mean(fr_final)) * 100
            if precept_final and fr_final
            else 0
        ),
    }
    apply_bonferroni(final_advantage["precept_vs_expel"], "p_value", 2)
    apply_bonferroni(final_advantage["precept_vs_full_reflexion"], "p_value", 2)

    return {
        "per_encounter_tests": per_encounter_tests,
        "improvement_tests": improvement_tests,
        "final_encounter_advantage": final_advantage,
        "multiple_comparison_correction": {
            "method": "bonferroni",
            "per_encounter_n_tests": per_encounter_m,
            "improvement_n_tests": len(agents),
            "final_encounter_n_tests": 2,
        },
    }


def run_single_experiment(
    domain: str, seed: int, output_dir: Path, params: Dict[str, int]
) -> Optional[Dict[str, Any]]:
    """
    Run a single sequential experiment and parse detailed results.

    Args:
        domain: Domain name
        seed: Random seed
        output_dir: Output directory
        params: Experiment parameters from get_experiment_params()

    Returns dict with per-agent metrics organized by encounter number.
    """
    clean_data_directory()

    cmd = [
        "uv",
        "run",
        "examples/precept_autogen_mcp_full.py",
        "--domain",
        domain,
        "--train",
        str(params["train_size"]),
        "--test",
        str(params["test_size"]),
        "--max-retries",
        str(params["max_retries"]),
        "--num-conditions",
        str(params["num_conditions"]),
        "--seed",
        str(seed),
        "--test-mode",
        "both",  # Run BOTH matched (learned keys) and random (new keys) test modes
        # Ablation flags for fair comparison
        "--no-static-knowledge",  # Enable static knowledge for all agents
        "--hybrid-retrieval",  # Enable BM25 + semantic retrieval for all agents
        "--improved-baselines",  # Enable metadata-based filtering for baselines
        # Concurrent training for speed
        "-ct",  # Enable concurrent training
        "-tw",
        "4",  # 4 training workers
        # SEQUENTIAL testing (critical for cross-episode learning!)
        "-w",
        "1",  # 1 test worker = sequential execution
        "-v",
    ]

    print(
        f"\n    🔄 Running: {domain} seed={seed} (concurrent train, sequential test)..."
    )

    log_file = output_dir / f"{domain}_seed{seed}.log"

    try:
        # Use Popen for real-time output with progress tracking
        process = subprocess.Popen(
            cmd,
            cwd=PROJECT_ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,  # Line buffered
        )

        stdout_lines = []
        train_total = params["train_size"]
        test_total = params["test_size"]
        train_progress = 0
        test_progress = 0
        current_phase = "init"

        # Create progress bars (skip training bar if β=0)
        train_bar = None
        test_bar = None

        if TQDM_AVAILABLE:
            if train_total > 0:
                train_bar = tqdm(
                    total=train_total,
                    desc=f"       📚 Training (seed={seed})",
                    leave=False,
                    position=1,
                    bar_format="{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt}",
                )
            else:
                print("       📚 Training: SKIPPED (β=0, cold start)")
            test_bar = tqdm(
                total=test_total,
                desc=f"       🧪 Testing (seed={seed}) ",
                leave=False,
                position=1 if train_total == 0 else 2,  # Adjust position if no training
                bar_format="{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt}",
            )

        # Stream output and update progress
        for line in iter(process.stdout.readline, ""):
            stdout_lines.append(line)

            # Detect phase transitions
            if "TRAINING PHASE" in line or "Training:" in line:
                current_phase = "training"
            elif "TESTING PHASE" in line or "Testing:" in line:
                current_phase = "testing"
                if train_bar:
                    train_bar.n = train_total
                    train_bar.refresh()

            # Track training progress
            if current_phase == "training":
                if re.search(r"Train(?:ing)?\s*\[?\s*(\d+)/", line):
                    match = re.search(r"Train(?:ing)?\s*\[?\s*(\d+)/", line)
                    if match:
                        train_progress = int(match.group(1))
                        if train_bar:
                            train_bar.n = train_progress
                            train_bar.refresh()
                # Also detect completion patterns
                elif "Training complete" in line or "training complete" in line:
                    if train_bar:
                        train_bar.n = train_total
                        train_bar.refresh()

            # Track testing progress
            if current_phase == "testing":
                if re.search(r"(?:MATCHED|RANDOM)\s+Test\s+(\d+)/", line):
                    match = re.search(r"(?:MATCHED|RANDOM)\s+Test\s+(\d+)/", line)
                    if match:
                        test_progress = int(match.group(1))
                        if test_bar:
                            test_bar.n = test_progress
                            test_bar.refresh()
                # Also detect task completion in testing
                elif re.search(r"Test\s+(\d+):", line):
                    match = re.search(r"Test\s+(\d+):", line)
                    if match:
                        test_progress = int(match.group(1))
                        if test_bar:
                            test_bar.n = min(test_progress, test_total)
                            test_bar.refresh()

        process.wait()

        # Close progress bars
        if train_bar:
            train_bar.n = train_total
            train_bar.refresh()
            train_bar.close()
        if test_bar:
            test_bar.n = test_total
            test_bar.refresh()
            test_bar.close()

        stdout_content = "".join(stdout_lines)

        # Save log
        with open(log_file, "w") as f:
            f.write(f"Command: {' '.join(cmd)}\n")
            f.write(f"Exit code: {process.returncode}\n")
            f.write("=" * 80 + "\n")
            f.write("STDOUT:\n")
            f.write(stdout_content)

        if process.returncode == 0:
            print(f"    ✅ Completed: {domain} seed={seed}")

            # Find and copy results file
            data_dir = PROJECT_ROOT / "data"
            result_files = list(data_dir.glob(f"experiment_results_{domain}_*.json"))

            run_metrics = {"seed": seed}
            agg_results = {}

            if result_files:
                latest = max(result_files, key=lambda p: p.stat().st_mtime)
                dest = output_dir / f"{domain}_seed{seed}_results.json"
                shutil.copy(latest, dest)

                # Parse aggregated results
                agg_results = parse_results_file(latest)
                run_metrics["aggregated"] = agg_results

            # Copy learning artifacts
            artifacts = copy_learning_artifacts(seed, output_dir)
            run_metrics["artifacts"] = {
                "rules_learned": len(artifacts.get("learned_rules", {})),
                "procedures": len(artifacts.get("procedures", {})),
                "partial_progress_keys": len(artifacts.get("partial_progress", {})),
            }

            # Parse detailed test results from log
            test_details = parse_log_for_test_details(log_file)

            # Compute per-encounter metrics for each agent
            for agent in ["precept", "expel", "full_reflexion"]:
                agent_tests = test_details.get(agent, [])
                agg_agent = (
                    agg_results.get("agents", {}).get(agent, {}) if agg_results else {}
                )

                if agent_tests and len(agent_tests) > 0:
                    run_metrics[agent] = compute_encounter_metrics(
                        agent_tests, params["num_keys"], params["encounters_per_key"]
                    )
                else:
                    # Fallback to aggregated metrics - synthesize per-encounter data
                    # Assume uniform distribution across encounters for aggregated data
                    total_tasks = agg_agent.get("total_tasks", 0)
                    if total_tasks <= 0:
                        print(
                            f"    ⚠️ Invalid run detected: {agent} has total_tasks=0 "
                            f"(domain={domain}, seed={seed}). Skipping run."
                        )
                        return None
                    p1 = agg_agent.get("first_try_success_rate", 0)
                    pt = agg_agent.get("success_rate", 0)
                    avg_steps = agg_agent.get("avg_steps", 0)

                    # Create synthetic by_encounter data
                    by_encounter = {}
                    for enc in range(1, params["encounters_per_key"] + 1):
                        by_encounter[f"encounter_{enc}"] = {
                            "p1": p1,
                            "pt": pt,
                            "avg_steps": avg_steps,
                            "first_try_successes": int(p1 * params["num_keys"]),
                            "successes": int(pt * params["num_keys"]),
                            "total": params["num_keys"],
                        }

                    run_metrics[agent] = {
                        "by_encounter": by_encounter,
                        "total_tasks": total_tasks,
                        "aggregated_p1": p1,
                        "aggregated_pt": pt,
                        "aggregated_steps": avg_steps,
                        "note": "Synthetic data from aggregated results (uniform assumption)",
                    }

            for agent in ["precept", "expel", "full_reflexion"]:
                if run_metrics.get(agent, {}).get("total_tasks", 0) <= 0:
                    print(
                        f"    ⚠️ Invalid run detected: {agent} has no test tasks "
                        f"(domain={domain}, seed={seed}). Skipping run."
                    )
                    return None

            return run_metrics

        else:
            print(f"    ❌ Failed: {domain} seed={seed}")
            print(f"    Check log: {log_file}")

    except Exception as e:
        # Handle timeout or other errors
        if "timeout" in str(e).lower():
            print(f"    ⏰ Timeout: {domain} seed={seed}")
        else:
            print(f"    ❌ Error: {domain} seed={seed}: {e}")
            import traceback

            traceback.print_exc()

    return None


def print_learning_curve_table(curves: Dict[str, Any], encounters_per_key: int):
    """Print learning curve results as a formatted table."""
    print("\n" + "=" * 80)
    print("📈 LEARNING CURVE: P₁ (First-Try Success) BY ENCOUNTER")
    print("=" * 80)

    # Header
    print(
        f"{'Encounter':<12} | {'PRECEPT P₁':>14} | {'ExpeL P₁':>14} | {'Full Ref P₁':>14}"
    )
    print("-" * 65)

    for enc in range(1, encounters_per_key + 1):
        key = f"encounter_{enc}"
        precept = curves.get("precept", {}).get(key, {})
        expel = curves.get("expel", {}).get(key, {})
        fr = curves.get("full_reflexion", {}).get(key, {})

        precept_p1 = precept.get("p1_mean", 0) * 100
        precept_ci = bounded_pct_ci(precept_p1, precept.get("p1_ci_95", 0) * 100)
        expel_p1 = expel.get("p1_mean", 0) * 100
        expel_ci = bounded_pct_ci(expel_p1, expel.get("p1_ci_95", 0) * 100)
        fr_p1 = fr.get("p1_mean", 0) * 100
        fr_ci = bounded_pct_ci(fr_p1, fr.get("p1_ci_95", 0) * 100)

        suffix = (
            "st" if enc == 1 else ("nd" if enc == 2 else ("rd" if enc == 3 else "th"))
        )

        # Format with CI if available
        if precept_ci > 0:
            print(
                f"{enc}{suffix:<11} | {precept_p1:>6.1f}%±{precept_ci:>4.1f} | "
                f"{expel_p1:>6.1f}%±{expel_ci:>4.1f} | "
                f"{fr_p1:>6.1f}%±{fr_ci:>4.1f}"
            )
        else:
            print(
                f"{enc}{suffix:<11} | {precept_p1:>13.1f}% | "
                f"{expel_p1:>13.1f}% | "
                f"{fr_p1:>13.1f}%"
            )

    print("-" * 65)


def print_overall_success_table(curves: Dict[str, Any], encounters_per_key: int):
    """Print overall success rate (Pₜ) by encounter."""
    print("\n" + "=" * 80)
    print("📊 OVERALL SUCCESS RATE (Pₜ) BY ENCOUNTER")
    print("=" * 80)

    print(
        f"{'Encounter':<12} | {'PRECEPT Pₜ':>14} | {'ExpeL Pₜ':>14} | {'Full Ref Pₜ':>14}"
    )
    print("-" * 65)

    for enc in range(1, encounters_per_key + 1):
        key = f"encounter_{enc}"
        precept = curves.get("precept", {}).get(key, {})
        expel = curves.get("expel", {}).get(key, {})
        fr = curves.get("full_reflexion", {}).get(key, {})

        precept_pt = precept.get("pt_mean", 0) * 100
        expel_pt = expel.get("pt_mean", 0) * 100
        fr_pt = fr.get("pt_mean", 0) * 100

        suffix = (
            "st" if enc == 1 else ("nd" if enc == 2 else ("rd" if enc == 3 else "th"))
        )
        print(
            f"{enc}{suffix:<11} | {precept_pt:>13.1f}% | "
            f"{expel_pt:>13.1f}% | "
            f"{fr_pt:>13.1f}%"
        )

    print("-" * 65)


def print_steps_table(curves: Dict[str, Any], encounters_per_key: int):
    """Print average steps by encounter."""
    print("\n" + "=" * 80)
    print("⚡ AVERAGE STEPS BY ENCOUNTER")
    print("=" * 80)

    print(f"{'Encounter':<12} | {'PRECEPT':>14} | {'ExpeL':>14} | {'Full Ref':>14}")
    print("-" * 65)

    for enc in range(1, encounters_per_key + 1):
        key = f"encounter_{enc}"
        precept = curves.get("precept", {}).get(key, {})
        expel = curves.get("expel", {}).get(key, {})
        fr = curves.get("full_reflexion", {}).get(key, {})

        suffix = (
            "st" if enc == 1 else ("nd" if enc == 2 else ("rd" if enc == 3 else "th"))
        )
        print(
            f"{enc}{suffix:<11} | {precept.get('steps_mean', 0):>14.2f} | "
            f"{expel.get('steps_mean', 0):>14.2f} | "
            f"{fr.get('steps_mean', 0):>14.2f}"
        )

    print("-" * 65)


def print_improvement_summary(improvements: Dict[str, Any]):
    """Print learning improvement summary."""
    print("\n" + "=" * 80)
    print("🎯 LEARNING IMPROVEMENT SUMMARY (1st → 4th encounter)")
    print("=" * 80)

    for agent in ["precept", "expel", "full_reflexion"]:
        imp = improvements.get(agent, {})
        agent_name = agent.upper().replace("_", " ")

        print(f"\n{agent_name}:")
        print(
            f"  P₁: {imp.get('enc1_p1', 0) * 100:.1f}% → {imp.get('enc4_p1', 0) * 100:.1f}% ({imp.get('p1_improvement_pp', 0):+.1f} pp)"
        )
        print(f"  Steps saved: {imp.get('steps_saved', 0):.2f}")


def generate_report(
    output_dir: Path,
    domain: str,
    curves: Dict[str, Any],
    improvements: Dict[str, Any],
    statistical_tests: Dict[str, Any],
    n_runs: int,
    seeds: List[int],
    params: Dict[str, int],
) -> None:
    """Generate a markdown report of the experiment."""
    report_path = output_dir / "learning_curve_analysis.md"

    with open(report_path, "w") as f:
        f.write("# Experiment 4: Continuous Learning Analysis\n\n")
        f.write(f"**Domain:** {domain}\n")
        f.write(f"**Seeds:** {seeds}\n")
        f.write(f"**Successful Runs:** {n_runs}\n\n")

        f.write("## Parameters\n\n")
        f.write(f"- Training: {params['train_size']} tasks (β={params['beta']})\n")
        f.write(
            f"- Testing: {params['test_size']} tasks ({params['encounters_per_key']} encounters per key)\n"
        )
        f.write(f"- Max Retries: {params['max_retries']}\n")
        f.write(f"- Num Conditions: {params['num_conditions']}\n")
        f.write(f"- Unique Keys: {params['num_keys']}\n\n")

        f.write("## Learning Curve: P₁ by Encounter\n\n")
        f.write("| Encounter | PRECEPT | ExpeL | Full Reflexion |\n")
        f.write("|-----------|---------|-------|----------------|\n")

        for enc in range(1, params["encounters_per_key"] + 1):
            key = f"encounter_{enc}"
            p = curves.get("precept", {}).get(key, {})
            e = curves.get("expel", {}).get(key, {})
            fr = curves.get("full_reflexion", {}).get(key, {})

            f.write(
                f"| {enc} | {p.get('p1_mean', 0) * 100:.1f}% | "
                f"{e.get('p1_mean', 0) * 100:.1f}% | "
                f"{fr.get('p1_mean', 0) * 100:.1f}% |\n"
            )

        f.write("\n## Learning Improvement (1st → 4th)\n\n")
        for agent in ["precept", "expel", "full_reflexion"]:
            imp = improvements.get(agent, {})
            imp_test = statistical_tests.get("improvement_tests", {}).get(agent, {})
            imp_sig = imp_test.get("improvement_significance", {})
            p_val_raw = imp_sig.get("p_value_one_tailed", 1.0)
            p_val = imp_sig.get("p_value_one_tailed_bonferroni", p_val_raw)
            sig = (
                "***"
                if p_val < 0.001
                else "**"
                if p_val < 0.01
                else "*"
                if p_val < 0.05
                else ""
            )
            f.write(
                f"**{agent.upper()}:** {imp.get('p1_improvement_pp', 0):+.1f} pp "
                f"(p={p_val:.4f}{sig})\n\n"
            )

        # Statistical significance section
        f.write("## Statistical Significance\n\n")
        f.write("### Final Encounter Comparison (PRECEPT vs Baselines)\n\n")
        f.write("| Comparison | p-value | Cohen's d | Significance |\n")
        f.write("|------------|---------|-----------|-------------|\n")

        final_adv = statistical_tests.get("final_encounter_advantage", {})
        for comparison, label in [
            ("precept_vs_expel", "PRECEPT vs ExpeL"),
            ("precept_vs_full_reflexion", "PRECEPT vs Full Reflexion"),
        ]:
            test = final_adv.get(comparison, {})
            p_val = test.get("p_value_bonferroni", test.get("p_value", 1.0))
            d = test.get("cohens_d", 0)
            sig = (
                "***"
                if p_val < 0.001
                else "**"
                if p_val < 0.01
                else "*"
                if p_val < 0.05
                else "n.s."
            )
            f.write(f"| {label} | {p_val:.4f} | {d:.2f} | {sig} |\n")

        f.write("\n### Per-Encounter Statistical Tests\n\n")
        f.write("| Encounter | PRECEPT vs ExpeL (p) | PRECEPT vs FR (p) |\n")
        f.write("|-----------|----------------------|-------------------|\n")
        per_enc = statistical_tests.get("per_encounter_tests", {})
        for enc in range(1, params["encounters_per_key"] + 1):
            key = f"encounter_{enc}"
            enc_tests = per_enc.get(key, {})
            p_expel = enc_tests.get("precept_vs_expel", {}).get(
                "p_value_bonferroni",
                enc_tests.get("precept_vs_expel", {}).get("p_value", 1.0),
            )
            p_fr = enc_tests.get("precept_vs_full_reflexion", {}).get(
                "p_value_bonferroni",
                enc_tests.get("precept_vs_full_reflexion", {}).get("p_value", 1.0),
            )
            sig_e = (
                "***"
                if p_expel < 0.001
                else "**"
                if p_expel < 0.01
                else "*"
                if p_expel < 0.05
                else ""
            )
            sig_f = (
                "***"
                if p_fr < 0.001
                else "**"
                if p_fr < 0.01
                else "*"
                if p_fr < 0.05
                else ""
            )
            f.write(f"| {enc} | {p_expel:.4f}{sig_e} | {p_fr:.4f}{sig_f} |\n")

        f.write(
            "\n*Significance: \\* p<0.05, \\*\\* p<0.01, \\*\\*\\* p<0.001 "
            "(Bonferroni-corrected p-values)*\n"
        )

        # Effect sizes interpretation
        f.write("\n### Effect Size Interpretation (Cohen's d)\n\n")
        f.write("- Small: d = 0.2\n")
        f.write("- Medium: d = 0.5\n")
        f.write("- Large: d = 0.8\n")

        f.write("## Key Findings\n\n")
        precept_imp = improvements.get("precept", {}).get("p1_improvement_pp", 0)
        expel_imp = improvements.get("expel", {}).get("p1_improvement_pp", 0)

        if precept_imp > expel_imp + 10:
            f.write(
                "✅ **PRECEPT shows significant cross-episode learning advantage**\n\n"
            )
        elif precept_imp > expel_imp:
            f.write("✅ **PRECEPT shows modest cross-episode learning advantage**\n\n")
        else:
            f.write(
                "⚠️ **Cross-episode learning advantage not clearly demonstrated**\n\n"
            )

    print(f"\n📄 Report saved to: {report_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Run Experiment 4: Continuous Learning"
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Run with 3 seeds instead of 10",
    )
    parser.add_argument(
        "--very-quick",
        action="store_true",
        help="Very quick validation (1 seed, logistics domain) for testing scripts work",
    )
    parser.add_argument(
        "--domain",
        type=str,
        default=None,
        choices=["finance", "logistics", "coding", "devops", "booking", "integration"],
        help="Domain to test (single domain). In publication mode, defaults to all 3 publication domains.",
    )
    parser.add_argument(
        "--publication",
        action="store_true",
        help="Full publication mode (10 seeds, 3 domains: integration, booking, logistics)",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        help="Specific seeds to use (overrides --quick)",
    )
    parser.add_argument(
        "--beta",
        type=int,
        default=DEFAULT_BETA,
        help="Training coverage factor. β=0: zero training (cold start), β=1: each key once (default)",
    )
    args = parser.parse_args()

    # Determine seeds and domains based on mode
    if args.very_quick:
        seeds = VERY_QUICK_SEEDS
        domains_to_run = ["logistics"]
        print("🚀 VERY QUICK MODE: 1 seed, logistics domain (validation only)")
    elif args.seeds:
        seeds = args.seeds
        domains_to_run = [args.domain] if args.domain else PUBLICATION_DOMAINS
    elif args.quick:
        seeds = QUICK_SEEDS
        domains_to_run = [args.domain] if args.domain else ["logistics"]
    elif args.publication or not args.domain:
        seeds = PUBLICATION_SEEDS
        domains_to_run = [args.domain] if args.domain else PUBLICATION_DOMAINS
        print(f"📊 PUBLICATION MODE: {len(seeds)} seeds, domains: {domains_to_run}")
    else:
        seeds = PUBLICATION_SEEDS
        domains_to_run = [args.domain]

    # Loop over all domains
    for domain in domains_to_run:
        # Get dynamic experiment parameters for domain
        params = get_experiment_params(domain, beta=args.beta)

        # Create output directory (include beta and domain in name)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        beta_suffix = f"_beta{args.beta}" if args.beta != DEFAULT_BETA else ""
        output_dir = (
            PROJECT_ROOT
            / "data"
            / "publication_results"
            / f"exp4_continuous_learning{beta_suffix}_{domain}_{timestamp}"
        )
        output_dir.mkdir(parents=True, exist_ok=True)

        print("=" * 80)
        print("🧪 EXPERIMENT 4: CONTINUOUS LEARNING / CROSS-EPISODE MEMORY")
        print("=" * 80)
        print(f"Domain: {domain}")
        print(f"Seeds: {seeds}")
        print(
            f"Beta (training coverage): {args.beta}"
            + (" (COLD START - zero training!)" if args.beta == 0 else "")
        )
        print(f"Output: {output_dir}")
        print()
        print("📋 Experiment Parameters (dynamic for domain):")
        print(
            f"  --train {params['train_size']} (β={params['beta']}: {params['num_keys']} keys × {params['beta']})"
        )
        print(
            f"  --test {params['test_size']} ({params['num_keys']} keys × {params['encounters_per_key']} encounters)"
        )
        print(f"  --max-retries {params['max_retries']}")
        print(f"  --num-conditions {params['num_conditions']}")
        print(f"  --num-keys {params['num_keys']} (unique condition keys for {domain})")
        print("  -ct -tw 4 (CONCURRENT training)")
        print("  -w 1 (SEQUENTIAL testing - critical for cross-episode learning)")
        print("=" * 80)

        # Run experiments for this domain
        all_runs = []
        successful_seeds: List[int] = []
        total_seeds = len(seeds)

        print(f"\n{'─' * 70}")
        print(f"🚀 Starting {total_seeds} experiment runs")
        print(
            f"   Domain: {domain} | Keys: {params['num_keys']} | Encounters: {params['encounters_per_key']}"
        )
        print(
            f"   Train: {params['train_size']} tasks (β={params['beta']}) | Test: {params['test_size']} tasks"
        )
        print(f"{'─' * 70}\n")

        for i, seed in enumerate(
            tqdm(
                seeds,
                desc=f"🧪 Exp4: {domain}",
                unit="seed",
                ncols=80,
                bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]",
            )
        ):
            # Status header for this seed
            print(f"\n  ┌{'─' * 50}")
            print(f"  │ Seed {seed} ({i + 1}/{total_seeds})")
            print(f"  └{'─' * 50}")

            result = run_single_experiment(domain, seed, output_dir, params)
            all_runs.append(result)
            if result is not None:
                successful_seeds.append(seed)

            # Summary for this seed
            if result:
                precept_enc = result.get("precept", {}).get("by_encounter", {})
                enc4 = precept_enc.get(f"encounter_{params['encounters_per_key']}", {})
                p1_final = enc4.get("p1", 0) * 100 if enc4 else 0
                enc1 = precept_enc.get("encounter_1", {})
                p1_start = enc1.get("p1", 0) * 100 if enc1 else 0
                improvement = p1_final - p1_start
                print(
                    f"  ✅ Seed {seed}: P₁ {p1_start:.0f}%→{p1_final:.0f}% ({improvement:+.0f}pp)"
                )

        # Filter successful runs
        successful_runs = [r for r in all_runs if r is not None]

        if not successful_runs:
            print(f"\n❌ No successful runs for {domain}. Check logs for errors.")
            continue

        print(f"\n{'─' * 70}")
        print(f"✅ Completed {len(successful_runs)}/{len(seeds)} runs for {domain}")
        print(f"{'─' * 70}")

        # Aggregate learning curves
        learning_curves = aggregate_learning_curves(
            successful_runs, params["encounters_per_key"]
        )

        # Compute improvements
        improvements = compute_learning_improvement(learning_curves)

        # Compute statistical tests
        statistical_tests = compute_statistical_tests(
            successful_runs, params["encounters_per_key"]
        )

        # Save comprehensive results
        results = {
            "experiment": "exp4_continuous_learning",
            "domain": domain,
            "timestamp": timestamp,
            "parameters": params,
            "n_runs": len(successful_runs),
            "seeds_used": successful_seeds,
            "learning_curves": learning_curves,
            "improvements": improvements,
            "statistical_tests": statistical_tests,
            "raw_runs": successful_runs,
        }

        results_file = output_dir / "continuous_learning_results.json"
        with open(results_file, "w") as f:
            json.dump(results, f, indent=2)

        print(f"\n📁 Results saved to: {results_file}")

        # Print analysis tables
        print_learning_curve_table(learning_curves, params["encounters_per_key"])
        print_overall_success_table(learning_curves, params["encounters_per_key"])
        print_steps_table(learning_curves, params["encounters_per_key"])
        print_improvement_summary(improvements)

        # Generate report
        generate_report(
            output_dir,
            domain,
            learning_curves,
            improvements,
            statistical_tests,
            len(successful_runs),
            successful_seeds,
            params,
        )

        # Final summary for this domain
        print("\n" + "=" * 80)
        print(f"📊 SUMMARY: CROSS-EPISODE LEARNING ADVANTAGE ({domain.upper()})")
        print("=" * 80)

        precept_imp = improvements.get("precept", {})
        expel_imp = improvements.get("expel", {})
        fr_imp = improvements.get("full_reflexion", {})

        print("\n🎯 P₁ Improvement (1st → 4th encounter):")
        for agent in ["precept", "expel", "full_reflexion"]:
            imp = improvements.get(agent, {})
            imp_test = statistical_tests.get("improvement_tests", {}).get(agent, {})
            imp_sig = imp_test.get("improvement_significance", {})
            p_val_raw = imp_sig.get("p_value_one_tailed", 1.0)
            p_val = imp_sig.get("p_value_one_tailed_bonferroni", p_val_raw)
            sig = (
                "***"
                if p_val < 0.001
                else "**"
                if p_val < 0.01
                else "*"
                if p_val < 0.05
                else ""
            )
            label = agent.upper().replace("_", " ")
            print(f"   {label:<15}: {imp.get('p1_improvement_pp', 0):+.1f} pp {sig}")

        # Print statistical significance summary
        print("\n📊 STATISTICAL SIGNIFICANCE (Final Encounter):")
        final_adv = statistical_tests.get("final_encounter_advantage", {})
        for comparison in ["precept_vs_expel", "precept_vs_full_reflexion"]:
            test = final_adv.get(comparison, {})
            p_val = test.get("p_value_bonferroni", test.get("p_value", 1.0))
            d = test.get("cohens_d", 0)
            sig = (
                "***"
                if p_val < 0.001
                else "**"
                if p_val < 0.01
                else "*"
                if p_val < 0.05
                else "n.s."
            )
            label = "vs ExpeL" if "expel" in comparison else "vs Full Reflexion"
            print(f"   PRECEPT {label:<18}: p={p_val:.4f} {sig}, Cohen's d={d:.2f}")

        print("\n   Significance: * p<0.05, ** p<0.01, *** p<0.001")

    print("\n" + "=" * 80)
    print("✅ Experiment 4 Complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()
