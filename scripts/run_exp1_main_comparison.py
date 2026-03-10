#!/usr/bin/env python3
"""
Experiment 1: Main Domain Comparison (PRECEPT vs Full Reflexion)

PURPOSE:
    Generate Table 1 and Figure 1 for publication - the main results comparing
    PRECEPT against Full Reflexion across all 6 domains.

STATISTICAL REQUIREMENTS:
    - N = 10 independent runs per domain (seeds: 42-7777)
    - Reports mean ± 95% CI, p-values, Cohen's d effect sizes
    - Bonferroni correction for multiple comparisons (6 domains)

OUTPUT:
    - data/publication_results/exp1_main_comparison/
    - Results: P₁ (first-try success), Pₜ (overall success), Steps, Cost

EXPECTED RUNTIME: ~2-3 hours (6 domains × 10 seeds × training + testing)

Usage:
    python scripts/run_exp1_main_comparison.py [--quick]

    --quick: Run with 3 seeds instead of 10 (for validation)
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

try:
    from tqdm import tqdm

    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False

    # Fallback: simple progress indicator
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

try:
    from scripts.utils.pct_bounds import bounded_pct_from_frac, fmt_pct, fmt_pct_bold
except ImportError:
    def bounded_pct_from_frac(metric):
        mean_pct = metric.get("mean", 0) * 100
        ci_pct = metric.get("ci_95", 0) * 100
        ci_pct = max(0.0, min(ci_pct, mean_pct, 100.0 - mean_pct))
        return mean_pct, ci_pct
    def fmt_pct(metric):
        m, c = bounded_pct_from_frac(metric)
        return f"{m:.1f}% ± {c:.1f}%"
    def fmt_pct_bold(metric):
        m, c = bounded_pct_from_frac(metric)
        return f"**{m:.1f}%** ± {c:.1f}%"

# Configuration
PUBLICATION_SEEDS = [42, 123, 456, 789, 999, 2024, 3141, 1337, 8888, 7777]
QUICK_SEEDS = [42, 123, 456]
VERY_QUICK_SEEDS = [42]  # Single seed for validation
VERY_QUICK_DOMAIN = "logistics"  # Single domain for validation (E=4, smallest)

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
# ═══════════════════════════════════════════════════════════════════════════════

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

# Multi-condition E values: Same as single-condition (generators use same key pools)
MULTI_CONDITION_E = {
    "finance": 6,  # VOLATILE_SYMBOLS only
    "logistics": 4,  # BLOCKED_PORTS only
    "coding": 5,  # BLOCKED_PACKAGES only
    "devops": 5,  # STUCK_STACKS only
    "booking": 17,  # BLOCKED_FLIGHTS (all 17)
    "integration": 6,  # OAUTH_SOURCES only
}

# Fixed parameters
# ═══════════════════════════════════════════════════════════════════════════════
# COMPOSITE-CONDITION MODE: Stress retrieval under overlapping condition keys
# This experiment evaluates core performance in the manuscript's multi-condition
# setting, where each scenario includes 5 conditions per key.
# For targeted compositional generalization analysis, see Experiment 6.
# ═══════════════════════════════════════════════════════════════════════════════
NUM_CONDITIONS = 5  # Multi-condition mode: 5 conditions per scenario (2^5=32 states)
MAX_RETRIES = 4
BETA = 3  # Coverage factor: T_train = β * E

# Select E values based on experiment mode
E_VALUES = MULTI_CONDITION_E if NUM_CONDITIONS > 1 else SINGLE_CONDITION_E

# Build domain configurations: T_train = β * E, T_test = E
DOMAINS = {
    name: {"E": E_VALUES[name], "train": BETA * E_VALUES[name], "test": E_VALUES[name]}
    for name in E_VALUES.keys()
}


def clean_data_directory():
    """Remove ALL persisted data to ensure clean experiment.

    This must remove every file that could leak state between seeds:
      - ChromaDB vector stores (chroma_*)
      - PRECEPT JSON state (precept_*.json)
      - Baseline memory files (full_reflexion_memory.json, expel_insights.json)
      - Stale experiment_results_*.json (prevents picking up old results)

    IMPORTANT: This must be called at the start of each experiment run
    to ensure no learned data leaks between experiments.
    """
    data_dir = PROJECT_ROOT / "data"

    # 1. Remove ChromaDB vector stores
    paths_to_clean = [
        data_dir / "chroma_precept",  # PRECEPT vector store
        data_dir / "chroma_static_knowledge",  # Static knowledge store
        data_dir / "chroma_expel",  # ExpeL vector store (isolated)
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
    # Without this, a failed run could pick up results from a previous seed via
    # max(result_files, key=lambda p: p.stat().st_mtime)
    stale_results = list(data_dir.glob("experiment_results_*.json"))
    if stale_results:
        for f in stale_results:
            f.unlink()
        print(f"  🧹 Removed {len(stale_results)} stale experiment_results_*.json files")


def run_single_experiment(
    domain: str, config: dict, seed: int, output_dir: Path
) -> dict:
    """Run a single experiment with live progress bars for training and testing."""

    # Clean data before each run
    clean_data_directory()

    train_tasks = config["train"]
    test_tasks = config["test"]

    # Build command with concurrent mode for faster execution
    cmd = [
        "uv",
        "run",
        "examples/precept_autogen_mcp_full.py",
        "--domain",
        domain,
        "--train",
        str(train_tasks),
        "--test",
        str(test_tasks),
        "--test-mode",
        "both",
        "--max-retries",
        str(MAX_RETRIES),
        "--num-conditions",
        str(NUM_CONDITIONS),
        "--no-static-knowledge",
        "--hybrid-retrieval",
        "--improved-baselines",
        "--seed",
        str(seed),
        # Concurrent mode flags for faster execution
        "-ct",  # Enable concurrent training ("Tesla Fleet" mode)
        "-tw",
        "4",  # 4 training workers
        "-c",  # Enable concurrent testing
        "-w",
        "4",  # 4 test workers
        "-v",
    ]

    print(
        f"    🔄 Running: {domain} seed={seed} (train={train_tasks}, test={test_tasks})..."
    )

    # Run experiment with live progress
    log_file = output_dir / f"{domain}_seed{seed}.log"

    # Use Popen for real-time progress tracking
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"

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
    rules_learned = 0
    tasks_done = 0
    current_phase = "training"

    # Create progress bars for training and testing phases
    train_bar = None
    test_bar = None

    if TQDM_AVAILABLE and train_tasks > 0:
        train_bar = tqdm(
            total=train_tasks,
            desc="      🎓 Train",
            unit="task",
            ncols=80,
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]",
            leave=True,
        )

    try:
        for line in process.stdout:
            output_lines.append(line)

            # Count rules being learned
            if "Rule persisted:" in line:
                rules_learned += 1

            # Detect phase transitions
            if "TRAINING PHASE" in line or "Training:" in line:
                current_phase = "training"
            elif "TESTING PHASE" in line or "Testing:" in line:
                current_phase = "testing"
                # Close training bar and open testing bar
                if train_bar:
                    train_bar.n = train_tasks
                    train_bar.refresh()
                    train_bar.close()
                    train_bar = None
                    print(
                        f"      ✓ Training done | {rules_learned} rules learned",
                        flush=True,
                    )

                if TQDM_AVAILABLE and test_tasks > 0 and test_bar is None:
                    test_bar = tqdm(
                        total=test_tasks,
                        desc="      🧪 Test ",
                        unit="task",
                        ncols=80,
                        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]",
                        leave=True,
                    )

            # Track task completions
            if "SUCCESS: True" in line or "SUCCESS: False" in line:
                tasks_done += 1
                if current_phase == "training" and train_bar:
                    task_num = tasks_done // 4  # 4 agents per task
                    if task_num > train_bar.n:
                        train_bar.n = min(task_num, train_tasks)
                        train_bar.refresh()
                elif current_phase == "testing" and test_bar:
                    # Reset counter for testing phase
                    test_task_num = (tasks_done - train_tasks * 4) // 4
                    if test_task_num > 0 and test_task_num > test_bar.n:
                        test_bar.n = min(test_task_num, test_tasks)
                        test_bar.refresh()

            # Track completion markers
            if "Training complete" in line:
                if train_bar:
                    train_bar.n = train_tasks
                    train_bar.refresh()
            if "Testing complete" in line:
                if test_bar:
                    test_bar.n = test_tasks
                    test_bar.refresh()

        process.wait(timeout=1800)

    except subprocess.TimeoutExpired:
        process.kill()
        if train_bar:
            train_bar.close()
        if test_bar:
            test_bar.close()
        print(f"    ⏰ Timeout: {domain} seed={seed}")
        return None
    except Exception as e:
        if train_bar:
            train_bar.close()
        if test_bar:
            test_bar.close()
        print(f"    ❌ Error: {domain} seed={seed}: {e}")
        return None
    finally:
        if train_bar:
            train_bar.n = train_tasks
            train_bar.refresh()
            train_bar.close()
        if test_bar:
            test_bar.n = test_tasks
            test_bar.refresh()
            test_bar.close()

    # Save log
    with open(log_file, "w") as f:
        f.write(f"Command: {' '.join(cmd)}\n")
        f.write(f"Exit code: {process.returncode}\n")
        f.write("=" * 80 + "\n")
        f.writelines(output_lines)

    if process.returncode == 0:
        print(f"    ✅ Completed: {domain} seed={seed} | {rules_learned} rules")

        # Find and copy the results file
        data_dir = PROJECT_ROOT / "data"
        result_files = list(data_dir.glob(f"experiment_results_{domain}_*.json"))
        if result_files:
            latest = max(result_files, key=lambda p: p.stat().st_mtime)
            dest = output_dir / f"{domain}_seed{seed}_results.json"
            shutil.copy(latest, dest)

            with open(latest) as f:
                return json.load(f)
    else:
        print(f"    ❌ Failed: {domain} seed={seed}")

    return None


def aggregate_results(all_results: dict, output_dir: Path):
    """Aggregate results across all runs with statistical analysis."""
    import numpy as np
    from scipy import stats

    aggregated = {}

    for domain, runs in all_results.items():
        if not runs:
            continue

        # Extract metrics
        precept_p1, precept_pt, precept_steps = [], [], []
        fr_p1, fr_pt, fr_steps = [], [], []
        expel_p1, expel_pt, expel_steps = [], [], []

        for run in runs:
            if run is None:
                continue
            agents = run.get("agents", {})

            # PRECEPT metrics
            p = agents.get("precept", {})
            precept_p1.append(p.get("first_try_success_rate", 0))
            precept_pt.append(p.get("success_rate", 0))
            precept_steps.append(p.get("avg_steps", 0))

            # Full Reflexion metrics
            fr = agents.get("full_reflexion", {})
            fr_p1.append(fr.get("first_try_success_rate", 0))
            fr_pt.append(fr.get("success_rate", 0))
            fr_steps.append(fr.get("avg_steps", 0))

            # ExpeL metrics (Zhao et al., 2023)
            expel = agents.get("expel", {})
            expel_p1.append(expel.get("first_try_success_rate", 0))
            expel_pt.append(expel.get("success_rate", 0))
            expel_steps.append(expel.get("avg_steps", 0))

        if not precept_p1:
            continue

        n = len(precept_p1)
        t_critical = stats.t.ppf(0.975, df=n - 1) if n > 1 else 0

        def compute_stats(values):
            arr = np.array(values)
            mean = np.mean(arr)
            std = np.std(arr, ddof=1) if len(arr) > 1 else 0
            ci = t_critical * std / np.sqrt(n) if n > 1 else 0
            return {"mean": mean, "std": std, "ci_95": ci, "n": n}

        # Statistical tests (paired t-test)
        def paired_test(v1, v2):
            if len(v1) < 2 or len(v2) < 2:
                return {"t_stat": 0, "p_value": 1, "cohens_d": 0}

            arr1, arr2 = np.array(v1), np.array(v2)
            t_stat, p_value = stats.ttest_rel(arr1, arr2)
            diff = arr1 - arr2
            cohens_d = (
                np.mean(diff) / np.std(diff, ddof=1) if np.std(diff, ddof=1) > 0 else 0
            )
            return {
                "t_stat": float(t_stat),
                "p_value": float(p_value),
                "cohens_d": float(cohens_d),
            }

        aggregated[domain] = {
            "n_runs": n,
            "precept": {
                "first_try_success": compute_stats(precept_p1),
                "success_rate": compute_stats(precept_pt),
                "avg_steps": compute_stats(precept_steps),
            },
            "full_reflexion": {
                "first_try_success": compute_stats(fr_p1),
                "success_rate": compute_stats(fr_pt),
                "avg_steps": compute_stats(fr_steps),
            },
            "expel": {
                "first_try_success": compute_stats(expel_p1),
                "success_rate": compute_stats(expel_pt),
                "avg_steps": compute_stats(expel_steps),
            },
            "statistical_tests": {
                "precept_vs_fr": {
                    "first_try_success": paired_test(precept_p1, fr_p1),
                    "success_rate": paired_test(precept_pt, fr_pt),
                    "avg_steps": paired_test(precept_steps, fr_steps),
                },
                "precept_vs_expel": {
                    "first_try_success": paired_test(precept_p1, expel_p1),
                    "success_rate": paired_test(precept_pt, expel_pt),
                    "avg_steps": paired_test(precept_steps, expel_steps),
                },
                # Legacy keys for backwards compatibility
                "first_try_success": paired_test(precept_p1, fr_p1),
                "success_rate": paired_test(precept_pt, fr_pt),
                "avg_steps": paired_test(precept_steps, fr_steps),
            },
            "advantage": {
                "vs_fr_first_try_pp": (np.mean(precept_p1) - np.mean(fr_p1)) * 100,
                "vs_fr_success_pp": (np.mean(precept_pt) - np.mean(fr_pt)) * 100,
                "vs_fr_steps_saved": np.mean(fr_steps) - np.mean(precept_steps),
                "vs_expel_first_try_pp": (np.mean(precept_p1) - np.mean(expel_p1))
                * 100,
                "vs_expel_success_pp": (np.mean(precept_pt) - np.mean(expel_pt)) * 100,
                "vs_expel_steps_saved": np.mean(expel_steps) - np.mean(precept_steps),
                # Legacy keys for backwards compatibility
                "first_try_success_pp": (np.mean(precept_p1) - np.mean(fr_p1)) * 100,
                "success_rate_pp": (np.mean(precept_pt) - np.mean(fr_pt)) * 100,
                "steps_saved": np.mean(fr_steps) - np.mean(precept_steps),
            },
        }

    # Bonferroni correction across domains (family-wise control per metric/pair)
    if aggregated:
        n_domains = len(aggregated)
        comparisons = ["precept_vs_fr", "precept_vs_expel"]
        metrics = ["first_try_success", "success_rate", "avg_steps"]

        for domain_data in aggregated.values():
            tests = domain_data.get("statistical_tests", {})
            for comp in comparisons:
                for metric in metrics:
                    test = tests.get(comp, {}).get(metric, {})
                    if not isinstance(test, dict) or "p_value" not in test:
                        continue
                    raw_p = float(test.get("p_value", 1.0))
                    corrected_p = min(1.0, raw_p * n_domains)
                    test["p_value_bonferroni"] = corrected_p
                    test["bonferroni_n_tests"] = n_domains
                    test["significant_raw"] = raw_p < 0.05
                    test["significant_bonferroni"] = corrected_p < 0.05

            # Keep legacy keys aligned with PRECEPT-vs-FR corrected values
            for metric in metrics:
                legacy = tests.get(metric, {})
                canonical = tests.get("precept_vs_fr", {}).get(metric, {})
                if isinstance(legacy, dict) and isinstance(canonical, dict):
                    if "p_value_bonferroni" in canonical:
                        legacy["p_value_bonferroni"] = canonical["p_value_bonferroni"]
                    if "bonferroni_n_tests" in canonical:
                        legacy["bonferroni_n_tests"] = canonical["bonferroni_n_tests"]
                    if "significant_bonferroni" in canonical:
                        legacy["significant_bonferroni"] = canonical[
                            "significant_bonferroni"
                        ]

    # Save aggregated results
    output_path = output_dir / "aggregated_results.json"
    with open(output_path, "w") as f:
        json.dump(aggregated, f, indent=2)

    # Generate summary report
    generate_summary_report(aggregated, output_dir)

    return aggregated


def generate_summary_report(aggregated: dict, output_dir: Path):
    """Generate a markdown summary report."""
    lines = []
    lines.append("# Experiment 1: Main Domain Comparison Results\n")
    lines.append(f"Generated: {datetime.now().isoformat()}\n")
    lines.append("## Summary: PRECEPT vs Baselines\n")
    lines.append(
        "| Domain | PRECEPT P₁ | FR P₁ | ExpeL P₁ | Δ vs FR | Δ vs ExpeL | p-value (FR) |"
    )
    lines.append(
        "|--------|-----------|-------|----------|---------|------------|--------------|"
    )

    for domain, data in aggregated.items():
        p = data["precept"]["first_try_success"]
        fr = data["full_reflexion"]["first_try_success"]
        expel = data.get("expel", {}).get("first_try_success", {"mean": 0, "ci_95": 0})
        test = (
            data["statistical_tests"]
            .get("precept_vs_fr", {})
            .get(
                "first_try_success",
                data["statistical_tests"].get(
                    "first_try_success", {"p_value": 1, "cohens_d": 0}
                ),
            )
        )
        adv_fr = data["advantage"].get(
            "vs_fr_first_try_pp", data["advantage"].get("first_try_success_pp", 0)
        )
        adv_expel = data["advantage"].get("vs_expel_first_try_pp", 0)

        p_for_sig = test.get("p_value_bonferroni", test.get("p_value", 1.0))
        sig = (
            "***"
            if p_for_sig < 0.001
            else "**"
            if p_for_sig < 0.01
            else "*"
            if p_for_sig < 0.05
            else ""
        )

        lines.append(
            f"| {domain.capitalize()} | "
            f"{fmt_pct_bold(p)} | "
            f"{fmt_pct(fr)} | "
            f"{fmt_pct(expel)} | "
            f"**+{adv_fr:.1f} pp** | "
            f"**+{adv_expel:.1f} pp** | "
            f"{p_for_sig:.4f}{sig} |"
        )

    lines.append(
        "\n*Significance: \\* p<0.05, \\*\\* p<0.01, \\*\\*\\* p<0.001 "
        "(Bonferroni-corrected p-values)*"
    )
    lines.append(
        f"\n*N = {aggregated[list(aggregated.keys())[0]]['n_runs']} independent runs per domain*"
    )

    # Add detailed tables
    lines.append("\n## Detailed Results\n")

    for domain, data in aggregated.items():
        lines.append(f"\n### {domain.capitalize()}\n")
        lines.append(
            "| Metric | PRECEPT | Full Reflexion | ExpeL | Δ vs FR | Δ vs ExpeL |"
        )
        lines.append(
            "|--------|---------|----------------|-------|---------|------------|"
        )

        for metric, label in [
            ("first_try_success", "P₁ (First-Try)"),
            ("success_rate", "Pₜ (Overall)"),
            ("avg_steps", "Avg Steps"),
        ]:
            p = data["precept"][metric]
            fr = data["full_reflexion"][metric]
            expel = data.get("expel", {}).get(metric, {"mean": 0, "ci_95": 0})

            if metric == "avg_steps":
                adv_fr = data["advantage"].get(
                    "vs_fr_steps_saved", data["advantage"].get("steps_saved", 0)
                )
                adv_expel = data["advantage"].get("vs_expel_steps_saved", 0)
                lines.append(
                    f"| {label} | {p['mean']:.2f} ± {p['ci_95']:.2f} | "
                    f"{fr['mean']:.2f} ± {fr['ci_95']:.2f} | "
                    f"{expel['mean']:.2f} ± {expel['ci_95']:.2f} | "
                    f"{adv_fr:.2f} saved | {adv_expel:.2f} saved |"
                )
            else:
                adv_fr = data["advantage"].get(
                    f"vs_fr_{metric}_pp", data["advantage"].get(f"{metric}_pp", 0)
                )
                adv_expel = data["advantage"].get(f"vs_expel_{metric}_pp", 0)
                lines.append(
                    f"| {label} | {fmt_pct_bold(p)} | "
                    f"{fmt_pct(fr)} | "
                    f"{fmt_pct(expel)} | "
                    f"+{adv_fr:.1f} pp | +{adv_expel:.1f} pp |"
                )

    with open(output_dir / "experiment_report.md", "w") as f:
        f.write("\n".join(lines))

    print(f"\n📊 Report saved: {output_dir / 'experiment_report.md'}")


def main():
    parser = argparse.ArgumentParser(
        description="Run Experiment 1: Main Domain Comparison"
    )
    parser.add_argument(
        "--quick", action="store_true", help="Quick validation (3 seeds)"
    )
    parser.add_argument(
        "--very-quick",
        action="store_true",
        help="Very quick validation (1 seed, 1 domain) for testing scripts work",
    )
    parser.add_argument(
        "--domain",
        type=str,
        default=None,
        choices=list(DOMAINS.keys()),
        help="Run only a specific domain (e.g., --domain booking)",
    )
    args = parser.parse_args()

    # Determine seeds and domains based on mode
    if args.very_quick:
        seeds = VERY_QUICK_SEEDS
        domains_to_run = {VERY_QUICK_DOMAIN: DOMAINS[VERY_QUICK_DOMAIN]}
        print("🚀 VERY QUICK MODE: 1 seed, 1 domain (validation only)")
    elif args.quick:
        seeds = QUICK_SEEDS
        domains_to_run = DOMAINS
    else:
        seeds = PUBLICATION_SEEDS
        domains_to_run = DOMAINS

    # Apply domain filter if specified
    if args.domain:
        if args.domain in domains_to_run:
            domains_to_run = {args.domain: domains_to_run[args.domain]}
            print(f"🎯 SINGLE DOMAIN MODE: Running only '{args.domain}'")
        else:
            print(f"❌ Domain '{args.domain}' not found in {list(DOMAINS.keys())}")
            sys.exit(1)

    print("=" * 80)
    print("EXPERIMENT 1: MAIN DOMAIN COMPARISON")
    print("PRECEPT vs Full Reflexion across 6 domains")
    print("=" * 80)
    print("\nConfiguration:")
    print(f"  Seeds: {seeds}")
    print(f"  Domains: {list(domains_to_run.keys())}")
    print(f"  num_conditions: {NUM_CONDITIONS}")
    print(f"  max_retries: {MAX_RETRIES}")

    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = (
        PROJECT_ROOT
        / "data"
        / "publication_results"
        / f"exp1_main_comparison_{timestamp}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nOutput directory: {output_dir}")

    # Save experiment config
    config = {
        "experiment": "exp1_main_comparison",
        "seeds": seeds,
        "domains": domains_to_run,
        "num_conditions": NUM_CONDITIONS,
        "max_retries": MAX_RETRIES,
        "timestamp": timestamp,
        "mode": "very_quick"
        if args.very_quick
        else ("quick" if args.quick else "full"),
    }
    with open(output_dir / "experiment_config.json", "w") as f:
        json.dump(config, f, indent=2)

    # Run experiments with progress bars
    all_results = {domain: [] for domain in domains_to_run}
    total_runs = len(domains_to_run) * len(seeds)
    completed = 0

    # Overall progress bar
    print(f"\n{'─' * 70}")
    print(f"🚀 Starting {total_runs} experiment runs")
    print(f"   Domains: {len(domains_to_run)} | Seeds: {len(seeds)} per domain")
    print("   Composite-condition mode (num_conditions=5)")
    print(f"{'─' * 70}\n")

    domain_list = list(domains_to_run.items())
    for i, (domain, domain_config) in enumerate(
        tqdm(
            domain_list,
            desc="🧪 Exp1: Main Comparison",
            unit="domain",
            ncols=80,
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]",
        )
    ):
        print(f"\n  ┌{'─' * 55}")
        print(f"  │ {domain.upper()} ({i + 1}/{len(domain_list)})")
        print(
            f"  │ E={domain_config['E']}, train={domain_config['train']}, test={domain_config['test']}"
        )
        print(f"  └{'─' * 55}")

        # Seed progress bar for this domain
        for j, seed in enumerate(
            tqdm(
                seeds,
                desc=f"    🌱 {domain.capitalize()}",
                unit="seed",
                leave=False,
                ncols=70,
                bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]",
            )
        ):
            result = run_single_experiment(domain, domain_config, seed, output_dir)
            all_results[domain].append(result)
            completed += 1

            # Brief status
            if result:
                p1 = result.get("precept", {}).get("first_try_success_rate", 0) * 100
                print(f"    ✓ seed={seed}: P₁={p1:.0f}%", end="\r")

        # Domain summary
        successful = [r for r in all_results[domain] if r]
        if successful:
            avg_p1 = (
                sum(
                    r.get("precept", {}).get("first_try_success_rate", 0)
                    for r in successful
                )
                / len(successful)
                * 100
            )
            print(
                f"  ✅ {domain.capitalize()}: {len(successful)}/{len(seeds)} seeds, avg P₁={avg_p1:.1f}%"
            )

    # Aggregate results
    print(f"\n{'=' * 60}")
    print("AGGREGATING RESULTS")
    print(f"{'=' * 60}")

    aggregated = aggregate_results(all_results, output_dir)

    # Print summary
    print("\n" + "=" * 80)
    print("EXPERIMENT 1 COMPLETE!")
    print("=" * 80)
    print(f"\nResults saved to: {output_dir}")
    print("\nKey files:")
    print("  • aggregated_results.json - Statistical summary")
    print("  • experiment_report.md - Publication-ready report")
    print("  • *_results.json - Individual run results")

    # Quick summary
    print("\n📊 QUICK SUMMARY (First-Try Success P₁):")
    print("-" * 60)
    for domain, data in aggregated.items():
        if data:
            p1_precept = data["precept"]["first_try_success"]["mean"] * 100
            p1_fr = data["full_reflexion"]["first_try_success"]["mean"] * 100
            adv = data["advantage"]["first_try_success_pp"]
            print(
                f"  {domain.capitalize():12s}: PRECEPT {p1_precept:.1f}% vs FR {p1_fr:.1f}% (Δ +{adv:.1f} pp)"
            )


if __name__ == "__main__":
    main()
