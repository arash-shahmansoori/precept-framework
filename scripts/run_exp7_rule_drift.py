#!/usr/bin/env python3
"""
Rule Drift & Rule Persistence Experiment Runner
================================================

This script supports TWO related but distinct experiments in the paper:

EXPERIMENT A — Rule Drift Adaptation (Paper: "Experiment 7: Rule Drift Adaptation")
    Tests whether agents can adapt when previously learned rules become STALE.

    Mechanism: The PRECEPT_DRIFT_SALT environment variable salts the MD5 hash
    used by get_valid_solution_for_conditions(). Training uses salt="0" and
    testing uses salt="1", so ~65-85% of condition keys map to DIFFERENT
    solutions at test time. Agents must detect stale rules and re-learn.

    Settings:
        --train-hash-seed 0   (default)
        --test-hash-seed 1    (default — different salt → genuine drift)

    Expected results:
        - PRECEPT: low P₁ on encounter 1 (stale rules), rapid recovery by
          encounter 2-4 as rules are updated via exploration.
        - Baselines: limited or no recovery due to approximate retrieval and
          inability to overwrite stale verbal memories with precision.

EXPERIMENT B — Rule Persistence and Retrieval Fidelity (Paper: "Experiment 5")
    Tests whether agents retain rules accurately across session boundaries
    when NO drift occurs (solutions are identical between training and testing).

    Settings:
        --train-hash-seed 0
        --test-hash-seed 0    (SAME salt → no drift, rules should be valid)

    Expected results:
        - PRECEPT: 100% P₁ across all encounters (exact hash-table lookup).
        - Baselines: degraded P₁ due to lossy approximate retrieval, even
          though solutions have not changed.

    To run Experiment 5 (rule persistence):
        python scripts/run_exp7_rule_drift.py --publication \\
            --train-hash-seed 0 --test-hash-seed 0

IMPLEMENTATION NOTE:
    The drift mechanism uses PRECEPT_DRIFT_SALT (not PYTHONHASHSEED alone).
    Both logistics.py and integration.py read os.environ["PRECEPT_DRIFT_SALT"]
    and salt the MD5 hash: md5(f"{salt}:{condition_key}"). When the env var is
    unset or empty, the hash is unsalted (backward compatible with all other
    experiments). PYTHONHASHSEED is also set for consistency but has no effect
    on hashlib.md5.

MEASURES (both experiments):
    - P₁ by encounter (1st → 4th)
    - Pₜ by encounter
    - Steps by encounter
    - Rule updates (rules_before vs rules_after)

Usage:
    # Rule Drift (Experiment 7) — default settings
    python scripts/run_exp7_rule_drift.py --publication

    # Rule Persistence (Experiment 5) — same hash seed for both phases
    python scripts/run_exp7_rule_drift.py --publication \\
        --train-hash-seed 0 --test-hash-seed 0

    # Quick validation run
    python scripts/run_exp7_rule_drift.py --quick --domain logistics
"""

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

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
VERY_QUICK_SEEDS = [42]
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
#
# ═══════════════════════════════════════════════════════════════════════════════

# Single-condition E values: Total distinct error types per domain
# Note: These are the CONFIG totals. Generators only use primary sources (see below).
# Single-condition E values: Count of unique error codes ACTUALLY USED by generators
# These values match what each domain's generator produces (verified empirically)
# Note: Config files have more error types, but generators only use primary sources
SINGLE_CONDITION_E = {
    "finance": 6,       # VOLATILE_SYMBOLS only (GME, BTC-USD, MEME-COIN, AMC, ETH-USD, TSLA)
    "logistics": 4,     # BLOCKED_PORTS only (rotterdam, hamburg, shanghai, los_angeles)
    "coding": 5,        # BLOCKED_PACKAGES only (fast_xml, auth_lib_v1, numpy_mkl, legacy_orm, gpu_compute)
    "devops": 5,        # STUCK_STACKS only (prod-api, data-pipeline, auth-service, monitoring, vpc-network)
    "booking": 17,      # BLOCKED_FLIGHTS (all 17 flights used)
    "integration": 6,   # OAUTH_SOURCES only (salesforce, hubspot, zendesk, stripe, google_workspace, microsoft_graph)
}

# Multi-condition E values: Unique base entities for composite key generation
# Multi-condition E values: Same as single-condition (generators use same key pools)
MULTI_CONDITION_E = {
    "finance": 6,       # VOLATILE_SYMBOLS only
    "logistics": 4,     # BLOCKED_PORTS only
    "coding": 5,        # BLOCKED_PACKAGES only
    "devops": 5,        # STUCK_STACKS only
    "booking": 17,      # BLOCKED_FLIGHTS (all 17)
    "integration": 6,   # OAUTH_SOURCES only
}

# Fixed experiment parameters
DEFAULT_NUM_CONDITIONS = 5  # Default: multi-condition CSPs
MAX_RETRIES = 3
ENCOUNTERS_PER_KEY = 4
DEFAULT_BETA = 3  # train coverage factor

# ═══════════════════════════════════════════════════════════════════════════════
# HASH SEED DEFAULTS — Controls whether drift occurs
# ═══════════════════════════════════════════════════════════════════════════════
# These values are passed as PRECEPT_DRIFT_SALT to the subprocess environment.
# The salt is incorporated into MD5 hashing in get_valid_solution_for_conditions().
#
# RULE DRIFT (Experiment 7):  train="0", test="1"  → different solutions → drift
# RULE PERSISTENCE (Exp 5):   train="0", test="0"  → same solutions → no drift
#
# Override via CLI: --train-hash-seed X --test-hash-seed Y
# ═══════════════════════════════════════════════════════════════════════════════
DEFAULT_TRAIN_HASH_SEED = "0"
DEFAULT_TEST_HASH_SEED = "1"  # Different from train → genuine rule drift


def clean_data_directory() -> None:
    """Remove ALL persisted data to ensure clean experiment."""
    data_dir = PROJECT_ROOT / "data"
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

    for json_file in data_dir.glob("precept_*.json"):
        json_file.unlink()
        print(f"  🧹 Removed: {json_file.name}")

    # Clean baseline persistence files (Full Reflexion + ExpeL JSON)
    # These must be cleaned between beta runs but are preserved between
    # train/test phases within the same run (via --preserve-learned-rules)
    baseline_files = [
        data_dir / "full_reflexion_memory.json",
        data_dir / "expel_insights.json",
    ]
    for bf in baseline_files:
        if bf.exists():
            bf.unlink()
            print(f"    🧹 Cleaned: {bf.name}")


def get_experiment_params(
    domain: str, beta: int, num_conditions: int
) -> Dict[str, int]:
    """Get training/test sizes for a given domain.

    Args:
        domain: Domain name (finance, logistics, etc.)
        beta: Training coverage factor
        num_conditions: Number of conditions per scenario (1 = single, >1 = multi)
    """
    # Select E values based on single vs multi condition mode
    e_values = SINGLE_CONDITION_E if num_conditions == 1 else MULTI_CONDITION_E
    num_keys = e_values[domain]
    return {
        "num_keys": num_keys,
        "train_size": beta * num_keys,
        "test_size": num_keys * ENCOUNTERS_PER_KEY,
        "num_conditions": num_conditions,
        "max_retries": MAX_RETRIES,
        "beta": beta,
        "encounters_per_key": ENCOUNTERS_PER_KEY,
    }


def _run_phase(
    *,
    domain: str,
    seed: int,
    train: int,
    test: int,
    num_conditions: int,
    max_retries: int,
    hash_seed: str,
    output_log: Path,
    test_mode: Optional[str] = None,
    sequential_test: bool = False,
    preserve_learned_rules: bool = False,
    phase_name: str = "Phase",
    show_progress: bool = True,
) -> int:
    """Run a training or testing phase with an explicit hash seed and live progress."""
    cmd = [
        "uv",
        "run",
        "examples/precept_autogen_mcp_full.py",
        "--domain",
        domain,
        "--train",
        str(train),
        "--test",
        str(test),
        "--max-retries",
        str(max_retries),
        "--num-conditions",
        str(num_conditions),
        "--seed",
        str(seed),
        "--no-static-knowledge",
        "--hybrid-retrieval",
        "--improved-baselines",
        "-v",
    ]

    if test_mode:
        cmd.extend(["--test-mode", test_mode])

    if preserve_learned_rules:
        cmd.append("--preserve-learned-rules")

    if train > 0:
        cmd.extend(["-ct", "-tw", "4"])

    if test > 0:
        if sequential_test:
            cmd.extend(["-w", "1"])
        else:
            cmd.extend(["-c", "-w", "4"])

    env = os.environ.copy()
    env["PYTHONHASHSEED"] = hash_seed
    env["PRECEPT_DRIFT_SALT"] = hash_seed  # Salt MD5 hash for genuine rule drift

    # Determine total tasks for progress tracking
    total_tasks = train if train > 0 else test
    mode_str = "sequential" if sequential_test else "concurrent"
    
    # Run subprocess and show real progress from output
    process = subprocess.Popen(
        cmd,
        cwd=PROJECT_ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        env=env,
        bufsize=1,
    )
    
    output_lines = []
    start_time = datetime.now()
    rules_learned = 0
    tasks_done = 0
    precept_success = 0
    precept_fail = 0
    
    # Create tqdm progress bar
    phase_type = "Train" if train > 0 else "Test"
    pbar = None
    if TQDM_AVAILABLE:
        pbar = tqdm(
            total=total_tasks,
            desc=f"      {phase_type}",
            unit="task",
            ncols=80,
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]",
            leave=True,
        )
    
    try:
        # Read output line by line and track progress
        for line in process.stdout:
            output_lines.append(line)
            
            # Count rules being learned
            if "Rule persisted:" in line:
                rules_learned += 1
            
            # Track PRECEPT task completions (both success and failure count as task done)
            # Look for the summary lines that indicate a task batch completed
            if "PRECEPT:" in line and ("✓" in line or "✗" in line):
                # This is a batch summary line like "PRECEPT:16✓"
                pass  # Handled by completion markers below
            
            # Count individual task completions for PRECEPT
            # These appear as "✅ PIVOT SUCCESS" or task results
            if "scenarios" in line and ("Training complete" in line or "Testing complete" in line):
                # Final completion - set to total
                if pbar:
                    pbar.n = total_tasks
                    pbar.refresh()
            
            # Track by counting SUCCESS/FAIL result lines
            if "SUCCESS: True" in line or "SUCCESS: False" in line:
                tasks_done += 1
                if pbar and tasks_done <= total_tasks * 4:  # Account for 4 agents
                    # Update every 4 agent results (1 task = 4 agent runs)
                    task_num = tasks_done // 4
                    if task_num > pbar.n:
                        pbar.n = min(task_num, total_tasks)
                        pbar.refresh()
            
            # Also track by PRECEPT-specific completions
            if "PIVOT SUCCESS" in line:
                precept_success += 1
            if "PIVOT" in line and "FAIL" in line:
                precept_fail += 1
        
        process.wait(timeout=1800)
        
    except subprocess.TimeoutExpired:
        process.kill()
        if pbar:
            pbar.close()
        print("\n      ⚠️ Phase timed out!", flush=True)
        return 1
    finally:
        if pbar:
            pbar.n = total_tasks  # Ensure it shows 100% at end
            pbar.refresh()
            pbar.close()
    
    elapsed = (datetime.now() - start_time).total_seconds()
    print(f"      ✓ {elapsed:.0f}s | {rules_learned} rules learned", flush=True)
    
    # Write log file
    with open(output_log, "w") as f:
        f.write(f"Command: {' '.join(cmd)}\n")
        f.write(f"PYTHONHASHSEED={hash_seed}\n")
        f.write(f"Exit code: {process.returncode}\n")
        f.write("=" * 80 + "\n")
        f.write("".join(output_lines))

    return process.returncode


def _load_rules(path: Path) -> Dict[str, str]:
    if not path.exists():
        return {}
    with open(path) as f:
        try:
            return json.load(f)
        except json.JSONDecodeError:
            return {}


def parse_log_for_test_details(log_path: Path) -> Dict[str, List[Dict]]:
    """Parse test log for per-task results (from exp4 parser)."""
    if not log_path.exists():
        return {}

    results = {
        "precept": [],
        "expel": [],
        "full_reflexion": [],
    }

    with open(log_path) as f:
        content = f.read()

    # Pattern accounts for log prefix (timestamp, level, module)
    pattern = re.compile(
        r"📊 \[(?:MATCHED|RANDOM) Test (\d+)/\d+\] "
        r"key=([^|]+?) \| "
        r"PRECEPT: ([✓✗]) \(P₁=([YN]), steps=(\d+)\) \| "
        r"ExpeL: ([✓✗]) \(P₁=([YN])\) \| "
        r"FullRef: ([✓✗]) \(P₁=([YN])\)",
        re.MULTILINE
    )

    for match in pattern.finditer(content):
        task_num = int(match.group(1))
        condition_key = match.group(2).strip()
        if condition_key.endswith("..."):
            condition_key = condition_key[:-3].strip()
        if not condition_key or condition_key == "...":
            condition_key = "unknown"

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

    return results


def compute_encounter_metrics(
    test_results: List[Dict], encounters_per_key: int
) -> Dict[str, Any]:
    """Compute metrics grouped by encounter number."""
    by_key = defaultdict(list)
    for result in test_results:
        key = result.get("condition_key", "unknown")
        by_key[key].append(result)

    metrics_by_encounter = defaultdict(
        lambda: {
            "first_try_successes": 0,
            "successes": 0,
            "total": 0,
            "steps": [],
        }
    )

    for _, encounters in by_key.items():
        for enc_num, result in enumerate(encounters):
            if enc_num >= encounters_per_key:
                continue
            enc_key = enc_num + 1
            metrics_by_encounter[enc_key]["total"] += 1
            metrics_by_encounter[enc_key]["steps"].append(result.get("steps", 0))
            if result.get("success", False):
                metrics_by_encounter[enc_key]["successes"] += 1
            if result.get("first_try", False):
                metrics_by_encounter[enc_key]["first_try_successes"] += 1

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
    all_runs: List[Dict], num_encounters: int
) -> Dict[str, Any]:
    """Aggregate learning curves across runs."""
    agents = ["precept", "expel", "full_reflexion"]
    learning_curves = {}

    for agent in agents:
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
    """Compute improvement from encounter 1 → encounter N."""
    improvements = {}
    for agent in ["precept", "expel", "full_reflexion"]:
        agent_curve = curves.get(agent, {})
        enc1 = agent_curve.get("encounter_1", {})
        enc4 = agent_curve.get(f"encounter_{ENCOUNTERS_PER_KEY}", {})
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
    *,
    domain: str,
    seed: int,
    output_dir: Path,
    params: Dict[str, int],
    train_hash_seed: str,
    test_hash_seed: str,
    seed_idx: int = 0,
    total_seeds: int = 1,
) -> Optional[Dict[str, Any]]:
    """Run training with one hash seed, then testing with a different hash seed."""
    clean_data_directory()

    data_dir = PROJECT_ROOT / "data"
    rules_path = data_dir / "precept_learned_rules.json"
    
    train_tasks = params["train_size"]
    test_tasks = params["test_size"]

    # Training phase
    print(f"    🎓 Training: {train_tasks} tasks (concurrent, PYTHONHASHSEED={train_hash_seed})", flush=True)
    train_log = output_dir / f"{domain}_seed{seed}_train.log"
    train_code = _run_phase(
        domain=domain,
        seed=seed,
        train=train_tasks,
        test=0,
        num_conditions=params["num_conditions"],
        max_retries=params["max_retries"],
        hash_seed=train_hash_seed,
        output_log=train_log,
        test_mode=None,
        sequential_test=False,
        phase_name=f"Training ({train_tasks} tasks)",
        show_progress=True,
    )

    if train_code != 0:
        print(f"    ❌ Training failed: {domain} seed={seed}", flush=True)
        return None

    # Wait for rules file to be fully written (with retry)
    time.sleep(1.0)
    rules_before = _load_rules(rules_path)
    
    # Retry if empty (file might still be flushing)
    if not rules_before:
        time.sleep(1.0)
        rules_before = _load_rules(rules_path)
    
    print(f"    ✅ Training done: {len(rules_before)} rules learned", flush=True)
    if rules_before:
        with open(output_dir / f"{domain}_seed{seed}_rules_before.json", "w") as f:
            json.dump(rules_before, f, indent=2)

    # Testing phase (hash seed changed → rules are now STALE)
    print(f"    🧪 Testing: {test_tasks} tasks (sequential, PYTHONHASHSEED={test_hash_seed} → drift!)", flush=True)
    test_log = output_dir / f"{domain}_seed{seed}_test.log"
    test_code = _run_phase(
        domain=domain,
        seed=seed,
        train=0,
        test=test_tasks,
        num_conditions=params["num_conditions"],
        max_retries=params["max_retries"],
        hash_seed=test_hash_seed,
        output_log=test_log,
        test_mode="matched",
        sequential_test=True,
        preserve_learned_rules=True,
        phase_name=f"Testing ({test_tasks} tasks)",
        show_progress=True,
    )

    if test_code != 0:
        print(f"    ❌ Testing failed: {domain} seed={seed}", flush=True)
        return None

    # Wait for rules file to be fully written (with retry)
    time.sleep(1.0)
    rules_after = _load_rules(rules_path)
    
    # Retry if seems incomplete
    if len(rules_after) < len(rules_before):
        time.sleep(1.0)
        rules_after = _load_rules(rules_path)
    changed_keys = len(
        {
            k
            for k in set(rules_before.keys()) | set(rules_after.keys())
            if rules_before.get(k) != rules_after.get(k)
        }
    )
    print(f"    ✅ Testing done: {changed_keys} rules updated (drift recovery)", flush=True)
    if rules_after:
        with open(output_dir / f"{domain}_seed{seed}_rules_after.json", "w") as f:
            json.dump(rules_after, f, indent=2)

    # Copy latest results file
    result_files = list(data_dir.glob(f"experiment_results_{domain}_*.json"))
    aggregated = {}
    if result_files:
        latest = max(result_files, key=lambda p: p.stat().st_mtime)
        dest = output_dir / f"{domain}_seed{seed}_results.json"
        shutil.copy(latest, dest)
        with open(latest) as f:
            aggregated = json.load(f)

    # Parse detailed test results from log
    test_details = parse_log_for_test_details(test_log)
    agg_agents = aggregated.get("agents", {}) if aggregated else {}
    for agent in ["precept", "expel", "full_reflexion"]:
        if agg_agents and agg_agents.get(agent, {}).get("total_tasks", 0) <= 0:
            print(
                f"    ⚠️ Invalid run detected: {agent} has total_tasks=0 "
                f"(domain={domain}, seed={seed}). Skipping run.",
                flush=True,
            )
            return None

    run_metrics = {
        "seed": seed,
        "aggregated": aggregated,
        "rules": {
            "before_count": len(rules_before),
            "after_count": len(rules_after),
            "changed_keys": len(
                {
                    k
                    for k in set(rules_before.keys()) | set(rules_after.keys())
                    if rules_before.get(k) != rules_after.get(k)
                }
            ),
        },
        "train_hash_seed": train_hash_seed,
        "test_hash_seed": test_hash_seed,
    }

    for agent in ["precept", "expel", "full_reflexion"]:
        agent_tests = test_details.get(agent, [])
        if agent_tests:
            run_metrics[agent] = compute_encounter_metrics(
                agent_tests, params["encounters_per_key"]
            )
        else:
            print(
                f"    ⚠️ Invalid run detected: missing parsed test details for {agent} "
                f"(domain={domain}, seed={seed}). Skipping run.",
                flush=True,
            )
            return None

    return run_metrics


def generate_report(
    output_dir: Path,
    domain: str,
    curves: Dict[str, Any],
    improvements: Dict[str, Any],
    statistical_tests: Dict[str, Any],
    n_runs: int,
    seeds: List[int],
    params: Dict[str, int],
    train_hash_seed: str,
    test_hash_seed: str,
    experiment_id: str,
) -> None:
    """Generate a markdown report of the experiment."""
    is_persistence = train_hash_seed == test_hash_seed
    report_path = (
        output_dir / "rule_persistence_analysis.md"
        if is_persistence
        else output_dir / "rule_drift_analysis.md"
    )
    title = (
        "Experiment 5: Rule Persistence / Retrieval Fidelity"
        if is_persistence
        else "Experiment 7: Rule Drift / Non-Stationary CSPs"
    )
    with open(report_path, "w") as f:
        f.write(f"# {title}\n\n")
        f.write(f"**Domain:** {domain}\n")
        f.write(f"**Experiment ID:** {experiment_id}\n")
        f.write(f"**Seeds:** {seeds}\n")
        f.write(f"**Successful Runs:** {n_runs}\n\n")

        f.write("## Hash Setup\n")
        f.write(f"- Train hash seed: `{train_hash_seed}`\n")
        f.write(f"- Test hash seed: `{test_hash_seed}`\n\n")

        f.write("## Parameters\n")
        f.write(f"- Training: {params['train_size']} tasks (β={params['beta']})\n")
        f.write(
            f"- Testing: {params['test_size']} tasks ({params['encounters_per_key']} encounters per key)\n"
        )
        f.write(f"- Num Conditions: {params['num_conditions']}\n")
        f.write(f"- Max Retries: {params['max_retries']}\n")
        f.write(f"- Unique Keys: {params['num_keys']}\n\n")

        f.write("## P₁ by Encounter\n\n")
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

        f.write("\n## Learning Improvement (1st → last encounter)\n\n")
        for agent in ["precept", "expel", "full_reflexion"]:
            imp = improvements.get(agent, {})
            imp_test = statistical_tests.get("improvement_tests", {}).get(agent, {})
            imp_sig = imp_test.get("improvement_significance", {})
            p_val_raw = imp_sig.get("p_value_one_tailed", 1.0)
            p_val = imp_sig.get("p_value_one_tailed_bonferroni", p_val_raw)
            sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else ""
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
            sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else "n.s."
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
            sig_e = "***" if p_expel < 0.001 else "**" if p_expel < 0.01 else "*" if p_expel < 0.05 else ""
            sig_f = "***" if p_fr < 0.001 else "**" if p_fr < 0.01 else "*" if p_fr < 0.05 else ""
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

    print(f"\n📄 Report saved to: {report_path}")


def main() -> None:
    # Ensure unbuffered output for real-time progress display
    import sys
    sys.stdout.reconfigure(line_buffering=True) if hasattr(sys.stdout, 'reconfigure') else None
    
    parser = argparse.ArgumentParser(
        description="Run Experiment 7: Rule Drift / Non-Stationary CSPs"
    )
    parser.add_argument("--quick", action="store_true", help="Run with 3 seeds")
    parser.add_argument(
        "--very-quick",
        action="store_true",
        help="Very quick validation (1 seed, logistics domain)",
    )
    parser.add_argument(
        "--domain",
        type=str,
        default=None,
        choices=["finance", "logistics", "coding", "devops", "booking", "integration"],
        help="Single domain to test (use --domains for multiple)",
    )
    parser.add_argument(
        "--domains",
        type=str,
        nargs="+",
        default=None,
        choices=["finance", "logistics", "coding", "devops", "booking", "integration"],
        help="Multiple domains to test (e.g., --domains logistics booking finance)",
    )
    parser.add_argument(
        "--num-conditions",
        type=int,
        default=DEFAULT_NUM_CONDITIONS,
        help="Number of conditions per scenario (1=single, >1=multi). Default: 5",
    )
    parser.add_argument(
        "--beta",
        type=int,
        default=DEFAULT_BETA,
        help="Training coverage factor (default: 3)",
    )
    parser.add_argument(
        "--train-hash-seed",
        type=str,
        default=DEFAULT_TRAIN_HASH_SEED,
        help="Drift salt used during training (sets PRECEPT_DRIFT_SALT env var). "
             "Default: '0'",
    )
    parser.add_argument(
        "--test-hash-seed",
        type=str,
        default=DEFAULT_TEST_HASH_SEED,
        help="Drift salt used during testing. Set DIFFERENT from train-hash-seed "
             "for rule drift (Exp 7), or SAME for rule persistence (Exp 5). "
             "Default: '1' (drift)",
    )
    parser.add_argument(
        "--publication",
        action="store_true",
        help="Full publication mode (10 seeds, all specified domains)",
    )
    args = parser.parse_args()

    # Determine domains to run
    if args.domains:
        domains = args.domains
    elif args.domain:
        domains = [args.domain]
    else:
        domains = None  # Will be set by mode below

    if args.very_quick:
        seeds = VERY_QUICK_SEEDS
        if domains is None:
            domains = ["logistics"]
            print("🚀 VERY QUICK MODE: 1 seed, logistics domain (validation only)")
        else:
            print(f"🚀 VERY QUICK MODE: 1 seed, domains: {domains} (validation only)")
        num_conditions = args.num_conditions
    elif args.quick:
        seeds = QUICK_SEEDS
        if domains is None:
            domains = ["logistics"]
        num_conditions = args.num_conditions
        print(f"🚀 QUICK MODE: {len(seeds)} seeds")
    elif args.publication:
        seeds = PUBLICATION_SEEDS
        if domains is None:
            domains = PUBLICATION_DOMAINS
        num_conditions = args.num_conditions
        print(f"📊 PUBLICATION MODE: {len(seeds)} seeds, domains: {domains}")
    else:
        seeds = PUBLICATION_SEEDS  # Default to full seeds
        if domains is None:
            domains = PUBLICATION_DOMAINS
        num_conditions = args.num_conditions

    # Run experiment for each domain
    all_domain_results = {}
    
    for domain in domains:
        params = get_experiment_params(
            domain, beta=args.beta, num_conditions=num_conditions
        )
        
        domain_results = run_domain_experiment(
            domain=domain,
            seeds=seeds,
            params=params,
            num_conditions=num_conditions,
            args=args,
        )
        
        if domain_results:
            all_domain_results[domain] = domain_results
    
    # Print final summary across all domains
    if len(domains) > 1 and all_domain_results:
        print_multi_domain_summary(all_domain_results, num_conditions)


def run_domain_experiment(
    domain: str,
    seeds: List[int],
    params: Dict[str, int],
    num_conditions: int,
    args,
) -> Optional[Dict[str, Any]]:
    """Run experiment for a single domain."""

    is_persistence = args.train_hash_seed == args.test_hash_seed
    experiment_id = "exp5_rule_persistence" if is_persistence else "exp7_rule_drift"
    experiment_title = (
        "EXPERIMENT 5: RULE PERSISTENCE"
        if is_persistence
        else "EXPERIMENT 7: RULE DRIFT"
    )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = (
        PROJECT_ROOT
        / "data"
        / "publication_results"
        / f"{experiment_id}_{domain}_{timestamp}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    condition_mode = "SINGLE" if num_conditions == 1 else "MULTI"
    print("\n" + "=" * 80, flush=True)
    print(f"🧪 {experiment_title} - {domain.upper()}", flush=True)
    print("=" * 80, flush=True)
    print(f"Domain: {domain}", flush=True)
    print(f"Seeds: {seeds}", flush=True)
    print(
        f"Condition Mode: {condition_mode} ({num_conditions} conditions per scenario)", flush=True
    )
    print(f"Train hash seed: {args.train_hash_seed}", flush=True)
    print(f"Test hash seed:  {args.test_hash_seed}", flush=True)
    print(f"Output: {output_dir}", flush=True)
    print(flush=True)
    print("📋 Experiment Parameters:", flush=True)
    print(
        f"  --train {params['train_size']} (β={params['beta']}: {params['num_keys']} keys × {params['beta']})", flush=True
    )
    print(
        f"  --test {params['test_size']} ({params['num_keys']} keys × {params['encounters_per_key']} encounters)", flush=True
    )
    print(f"  --max-retries {params['max_retries']}", flush=True)
    print(
        f"  --num-conditions {params['num_conditions']} ({condition_mode}-condition CSPs)", flush=True
    )
    print("  -ct -tw 4 (CONCURRENT training)", flush=True)
    print("  -w 1 (SEQUENTIAL testing - critical for encounter tracking)", flush=True)
    print("=" * 80, flush=True)

    # Save experiment config
    config = {
        "experiment": experiment_id,
        "experiment_variant": "persistence_no_drift" if is_persistence else "drift",
        "domain": domain,
        "seeds": seeds,
        "num_conditions": num_conditions,
        "condition_mode": condition_mode,
        "parameters": params,
        "train_hash_seed": args.train_hash_seed,
        "test_hash_seed": args.test_hash_seed,
        "timestamp": timestamp,
        "mode": "very_quick"
        if args.very_quick
        else ("quick" if args.quick else "full"),
    }
    with open(output_dir / "experiment_config.json", "w") as f:
        json.dump(config, f, indent=2)

    all_runs = []
    successful_seeds: List[int] = []
    total_seeds = len(seeds)
    
    print(f"\n{'─' * 70}", flush=True)
    print(f"🚀 Starting {total_seeds} experiment runs for {domain}", flush=True)
    print(f"   Keys: {params['num_keys']} | Encounters: {params['encounters_per_key']}", flush=True)
    print(f"   Train: {params['train_size']} tasks | Test: {params['test_size']} tasks", flush=True)
    print(f"{'─' * 70}\n", flush=True)
    
    # Track timing for estimates
    seed_times = []
    
    for i, seed in enumerate(seeds):
        seed_start = datetime.now()
        
        # Status header for this seed
        print(f"\n  ╔{'═' * 60}", flush=True)
        print(f"  ║ 🔄 SEED {seed} ({i + 1}/{total_seeds}) - {domain.upper()}", flush=True)
        if seed_times:
            avg_time = sum(seed_times) / len(seed_times)
            remaining = (total_seeds - i) * avg_time
            print(f"  ║ ⏱️  Avg: {avg_time:.1f}s/seed | Est. remaining: {remaining/60:.1f} min", flush=True)
        print(f"  ╚{'═' * 60}", flush=True)
        
        result = run_single_experiment(
            domain=domain,
            seed=seed,
            output_dir=output_dir,
            params=params,
            train_hash_seed=args.train_hash_seed,
            test_hash_seed=args.test_hash_seed,
            seed_idx=i,
            total_seeds=total_seeds,
        )
        all_runs.append(result)
        if result is not None:
            successful_seeds.append(seed)
        
        seed_elapsed = (datetime.now() - seed_start).total_seconds()
        seed_times.append(seed_elapsed)
        
        # Summary for this seed
        if result:
            rules_changed = result.get("rules", {}).get("changed_keys", 0)
            precept_enc = result.get("precept", {}).get("by_encounter", {})
            enc4 = precept_enc.get("encounter_4", {})
            p1_final = enc4.get("p1", 0) * 100 if enc4 else 0
            print(f"  ✅ Seed {seed} complete in {seed_elapsed:.1f}s: P₁@enc4={p1_final:.0f}%, rules_updated={rules_changed}")
        else:
            print(f"  ❌ Seed {seed} failed after {seed_elapsed:.1f}s")
        
        # Overall progress
        progress_pct = (i + 1) / total_seeds * 100
        print(f"  📊 Overall: {i + 1}/{total_seeds} seeds ({progress_pct:.0f}%)")

    successful_runs = [r for r in all_runs if r is not None]
    if not successful_runs:
        print(f"\n❌ No successful runs for {domain}. Check logs for errors.")
        return None

    learning_curves = aggregate_learning_curves(
        successful_runs, params["encounters_per_key"]
    )
    improvements = compute_learning_improvement(learning_curves)
    statistical_tests = compute_statistical_tests(
        successful_runs, params["encounters_per_key"]
    )

    results = {
        "experiment": experiment_id,
        "experiment_variant": "persistence_no_drift" if is_persistence else "drift",
        "domain": domain,
        "timestamp": timestamp,
        "parameters": params,
        "n_runs": len(successful_runs),
        "seeds_used": successful_seeds,
        "train_hash_seed": args.train_hash_seed,
        "test_hash_seed": args.test_hash_seed,
        "learning_curves": learning_curves,
        "improvements": improvements,
        "statistical_tests": statistical_tests,
        "raw_runs": successful_runs,
    }

    results_file = (
        output_dir / "rule_persistence_results.json"
        if is_persistence
        else output_dir / "rule_drift_results.json"
    )
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)

    generate_report(
        output_dir,
        domain,
        learning_curves,
        improvements,
        statistical_tests,
        len(successful_runs),
        successful_seeds,
        params,
        args.train_hash_seed,
        args.test_hash_seed,
        experiment_id,
    )

    # Print summary table
    print("\n" + "=" * 70)
    print(f"📈 [{domain.upper()}] RECOVERY CURVE: P₁ BY ENCOUNTER")
    print("=" * 70)
    print(f"{'Encounter':<12} | {'PRECEPT':>12} | {'ExpeL':>12} | {'Full Ref':>12}")
    print("-" * 70)
    for enc in range(1, params["encounters_per_key"] + 1):
        key = f"encounter_{enc}"
        p = learning_curves.get("precept", {}).get(key, {})
        e = learning_curves.get("expel", {}).get(key, {})
        fr = learning_curves.get("full_reflexion", {}).get(key, {})
        suffix = (
            "st" if enc == 1 else ("nd" if enc == 2 else ("rd" if enc == 3 else "th"))
        )
        print(
            f"{enc}{suffix:<11} | {p.get('p1_mean', 0) * 100:>11.1f}% | "
            f"{e.get('p1_mean', 0) * 100:>11.1f}% | "
            f"{fr.get('p1_mean', 0) * 100:>11.1f}%"
        )
    print("-" * 70)

    # Print improvement summary
    trend_label = "PERSISTENCE TREND" if is_persistence else "DRIFT RECOVERY"
    print(f"\n🎯 [{domain.upper()}] {trend_label} (1st → last encounter):")
    for agent in ["precept", "expel", "full_reflexion"]:
        imp = improvements.get(agent, {})
        imp_test = statistical_tests.get("improvement_tests", {}).get(agent, {})
        imp_sig = imp_test.get("improvement_significance", {})
        p_val_raw = imp_sig.get("p_value_one_tailed", 1.0)
        p_val = imp_sig.get("p_value_one_tailed_bonferroni", p_val_raw)
        sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else ""
        print(f"   {agent.upper():<15}: {imp.get('p1_improvement_pp', 0):+.1f} pp {sig}")

    # Print statistical significance summary
    print(f"\n📊 [{domain.upper()}] STATISTICAL SIGNIFICANCE (Final Encounter):")
    final_adv = statistical_tests.get("final_encounter_advantage", {})
    for comparison in ["precept_vs_expel", "precept_vs_full_reflexion"]:
        test = final_adv.get(comparison, {})
        p_val = test.get("p_value_bonferroni", test.get("p_value", 1.0))
        d = test.get("cohens_d", 0)
        sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else "n.s."
        label = "vs ExpeL" if "expel" in comparison else "vs Full Reflexion"
        print(f"   PRECEPT {label:<18}: p={p_val:.4f} {sig}, Cohen's d={d:.2f}")

    print(f"\n📁 Results saved to: {output_dir}")
    
    return results


def print_multi_domain_summary(all_results: Dict[str, Any], num_conditions: int) -> None:
    """Print summary across all domains."""
    condition_mode = "SINGLE" if num_conditions == 1 else "MULTI"
    
    first_result = next(iter(all_results.values()), {})
    experiment_variant = first_result.get("experiment_variant", "drift")
    exp_label = (
        "EXPERIMENT 5 SUMMARY"
        if experiment_variant == "persistence_no_drift"
        else "EXPERIMENT 7 SUMMARY"
    )

    print("\n" + "=" * 80)
    print(f"📊 {exp_label}: ALL DOMAINS ({condition_mode}-CONDITION)")
    print("=" * 80)
    
    print(f"\n{'Domain':<15} | {'PRECEPT Δ':>12} | {'ExpeL Δ':>12} | {'FR Δ':>12} | {'PRECEPT @Enc4':>14}")
    print("-" * 80)
    
    for domain, results in all_results.items():
        improvements = results.get("improvements", {})
        curves = results.get("learning_curves", {})
        
        p_imp = improvements.get("precept", {}).get("p1_improvement_pp", 0)
        e_imp = improvements.get("expel", {}).get("p1_improvement_pp", 0)
        fr_imp = improvements.get("full_reflexion", {}).get("p1_improvement_pp", 0)
        
        p_enc4 = curves.get("precept", {}).get("encounter_4", {}).get("p1_mean", 0) * 100
        
        print(f"{domain:<15} | {p_imp:>+11.1f}% | {e_imp:>+11.1f}% | {fr_imp:>+11.1f}% | {p_enc4:>13.1f}%")
    
    print("-" * 80)
    print("\nΔ = P₁ improvement from encounter 1 → encounter 4")
    print("Significance: * p<0.05, ** p<0.01, *** p<0.001")
    completion_label = (
        "Experiment 5 Complete (All Domains)!"
        if experiment_variant == "persistence_no_drift"
        else "Experiment 7 Complete (All Domains)!"
    )
    print("\n" + "=" * 80)
    print(f"✅ {completion_label}")
    print("=" * 80)


if __name__ == "__main__":
    main()
