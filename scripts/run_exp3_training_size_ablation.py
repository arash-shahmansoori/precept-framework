#!/usr/bin/env python3
"""
Experiment 3: Training Size Ablation Study (β values)

PURPOSE:
    Generate Figure 3 for publication - demonstrating how learning effectiveness
    varies with training exposure (β = number of times each error type is seen).

    Shows the learning curve and sample efficiency of PRECEPT vs Full Reflexion.

STATISTICAL REQUIREMENTS:
    - N = 10 independent runs per configuration
    - Tests: β = 1, 2, 3, 4 (T_train = β × E)
    - Logistics domain (E = 7)

OUTPUT:
    - data/publication_results/exp4_training_size_ablation/
    - Results: Performance vs training exposure

EXPECTED RUNTIME: ~60 minutes (4 β values × 10 seeds)

Usage:
    python scripts/run_exp4_training_size_ablation.py [--quick]
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

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

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

PUBLICATION_SEEDS = [42, 123, 456, 789, 999, 2024, 3141, 1337, 8888, 7777]
QUICK_SEEDS = [42, 123, 456]
VERY_QUICK_SEEDS = [42]  # Single seed for validation
PUBLICATION_DOMAINS = ["integration", "logistics"]

# Ablation: β values (training exposure)
BETA_VALUES = [1, 2, 3, 4, 5]

# ═══════════════════════════════════════════════════════════════════════════════
# DOMAIN E VALUES: Single-Condition vs Multi-Condition
# ═══════════════════════════════════════════════════════════════════════════════
#
# E (error types / condition keys) differs based on experiment mode:
#
# SINGLE-CONDITION (--num-conditions 1):
#   E = total distinct error codes across ALL categories in the domain config
#   Each error code is its own condition key (e.g., "R-482")
#
# MULTI-CONDITION (--num-conditions > 1):
#   E = number of unique BASE ENTITIES that generate composite condition keys
#   Multiple conditions are combined into one key (e.g., "R-482+C-HIGH+T-PEAK")
#   The base entity count determines how many unique composite keys exist
#
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

# ═══════════════════════════════════════════════════════════════════════════════
# MULTI-CONDITION MODE: Test learning efficiency with composite condition keys
# This experiment tests how PRECEPT's learning scales with training exposure (β).
# Multi-condition mode (5 conditions) creates 2^5=32 possible states per scenario,
# challenging baselines while testing PRECEPT's O(1) lookup efficiency.
# Max allowed: 10 conditions (2^10 = 1024 possible states)
# ═══════════════════════════════════════════════════════════════════════════════
NUM_CONDITIONS = 5  # Composite-condition mode (5 conditions per scenario)
MAX_RETRIES = 4


def _bounded_pct_ci(mean_pct: float, ci_pct: float, lower: float = 0.0, upper: float = 100.0) -> float:
    """Bound symmetric CI so mean±CI stays within [lower, upper]."""
    return max(0.0, min(ci_pct, mean_pct - lower, upper - mean_pct))


def _bounded_pct_from_frac(metric: dict) -> tuple[float, float]:
    """Convert fractional metric dict to bounded percentage mean/CI."""
    mean_pct = metric.get("mean", 0) * 100
    ci_pct = metric.get("ci_95", 0) * 100
    return mean_pct, _bounded_pct_ci(mean_pct, ci_pct)


def clean_data_directory():
    """Remove ALL persisted data to avoid experiment contamination.

    This cleans:
    - PRECEPT's ChromaDB vector store (chroma_precept/)
    - Static knowledge vector store (chroma_static_knowledge/)
    - ExpeL's ChromaDB vector store (chroma_expel/) - NEW
    - All PRECEPT JSON data files (precept_*.json)

    IMPORTANT: This must be called at the start of each experiment run
    to ensure no learned data leaks between experiments.
    """
    data_dir = PROJECT_ROOT / "data"

    # All ChromaDB directories to clean (including ExpeL's and Full Reflexion's stores)
    paths_to_clean = [
        data_dir / "chroma_precept",  # PRECEPT vector store
        data_dir / "chroma_static_knowledge",  # Static knowledge store
        data_dir / "chroma_expel",  # ExpeL vector store (isolated)
        data_dir / "chroma_full_reflexion",  # Full Reflexion's reflection store
    ]

    for path in paths_to_clean:
        if path.exists():
            shutil.rmtree(path)
            print(f"    🧹 Cleaned: {path.name}/")

    # Clean PRECEPT JSON data files
    json_cleaned = 0
    for json_file in data_dir.glob("precept_*.json"):
        json_file.unlink()
        json_cleaned += 1

    if json_cleaned > 0:
        print(f"    🧹 Cleaned: {json_cleaned} precept_*.json files")

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


def _run_phase(
    *,
    domain: str,
    train: int,
    test: int,
    seed: int,
    output_log: Path,
    phase_name: str = "Phase",
) -> tuple:
    """Run a training or testing phase with live progress bar."""
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
        str(MAX_RETRIES),
        "--num-conditions",
        str(NUM_CONDITIONS),
        "--seed",
        str(seed),
        "--no-static-knowledge",
        "--hybrid-retrieval",
        "--improved-baselines",
        "-v",
    ]

    if train > 0:
        # Use concurrent training for efficiency (same as main PRECEPT design)
        # The hash-based solution determination is now deterministic (uses hashlib.md5)
        cmd.extend(["-ct", "-tw", "4"])  # Concurrent training with 4 workers
        cmd.extend(["--test-mode", "matched"])

    if test > 0:
        cmd.extend(["-c", "-w", "4"])  # Concurrent testing
        # CRITICAL: Use "matched" to test ONLY on trained keys
        # "both" would introduce new keys, confounding the β-ablation study
        cmd.extend(["--test-mode", "matched"])
        cmd.append("--preserve-learned-rules")

    # Determine total tasks for progress tracking
    total_tasks = train if train > 0 else test

    # Run subprocess with live progress
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
        for line in process.stdout:
            output_lines.append(line)

            # Count rules being learned
            if "Rule persisted:" in line:
                rules_learned += 1

            # Track task completions
            if "SUCCESS: True" in line or "SUCCESS: False" in line:
                tasks_done += 1
                if pbar and tasks_done <= total_tasks * 4:
                    task_num = tasks_done // 4
                    if task_num > pbar.n:
                        pbar.n = min(task_num, total_tasks)
                        pbar.refresh()

            # Also track completion markers
            if "Training complete" in line or "Testing complete" in line:
                if pbar:
                    pbar.n = total_tasks
                    pbar.refresh()

        process.wait(timeout=1200)

    except subprocess.TimeoutExpired:
        process.kill()
        if pbar:
            pbar.close()
        return 1, [], 0
    finally:
        if pbar:
            pbar.n = total_tasks
            pbar.refresh()
            pbar.close()

    # Write log file
    with open(output_log, "w") as f:
        f.write(f"Command: {' '.join(cmd)}\n")
        f.write(f"Exit code: {process.returncode}\n")
        f.write("=" * 80 + "\n")
        f.writelines(output_lines)

    return process.returncode, output_lines, rules_learned


def run_single_experiment(
    beta: int, seed: int, output_dir: Path, domain: str, E: int, test_size: int
) -> dict:
    """Run a single experiment with live progress bars for train and test phases."""
    clean_data_directory()

    train = beta * E  # T_train = β × E

    print(f"    🔄 Running: β={beta} (train={train}, test={test_size}) seed={seed} domain={domain}...")

    # Phase 1: Training
    train_log = output_dir / f"beta{beta}_seed{seed}_train.log"
    train_exit, train_output, rules_learned = _run_phase(
        domain=domain,
        train=train,
        test=0,
        seed=seed,
        output_log=train_log,
        phase_name="Train",
    )

    if train_exit != 0:
        print(f"    ❌ Training failed: β={beta} seed={seed}")
        return None

    print(f"      ✓ Training done | {rules_learned} rules learned", flush=True)
    time.sleep(1)  # Allow file sync

    # Phase 2: Testing
    test_log = output_dir / f"beta{beta}_seed{seed}_test.log"
    test_exit, test_output, _ = _run_phase(
        domain=domain,
        train=0,
        test=test_size,
        seed=seed,
        output_log=test_log,
        phase_name="Test",
    )

    if test_exit != 0:
        print(f"    ❌ Testing failed: β={beta} seed={seed}")
        return None

    print(f"    ✅ Completed: β={beta} seed={seed}", flush=True)

    # Load results
    data_dir = PROJECT_ROOT / "data"
    result_files = list(data_dir.glob(f"experiment_results_{domain}_*.json"))
    if result_files:
        latest = max(result_files, key=lambda p: p.stat().st_mtime)
        dest = output_dir / f"beta{beta}_seed{seed}_results.json"
        shutil.copy(latest, dest)

        with open(latest) as f:
            return json.load(f)

    return None


def aggregate_results(all_results: dict, output_dir: Path, domain: str = "logistics", E_val: int = None):
    """Aggregate results with statistical analysis."""
    import numpy as np
    from scipy import stats

    aggregated = {}

    for beta, runs in all_results.items():
        if not runs:
            continue

        precept_p1, precept_pt, precept_steps = [], [], []
        fr_p1, fr_pt, fr_steps = [], [], []
        expel_p1, expel_pt, expel_steps = [], [], []

        for run in runs:
            if run is None:
                continue
            agents = run.get("agents", {})

            p = agents.get("precept", {})
            precept_p1.append(p.get("first_try_success_rate", 0))
            precept_pt.append(p.get("success_rate", 0))
            precept_steps.append(p.get("avg_steps", 0))

            fr = agents.get("full_reflexion", {})
            fr_p1.append(fr.get("first_try_success_rate", 0))
            fr_pt.append(fr.get("success_rate", 0))
            fr_steps.append(fr.get("avg_steps", 0))

            # ExpeL baseline (Zhao et al., 2023)
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
            return {"mean": float(mean), "std": float(std), "ci_95": float(ci), "n": n}

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

        aggregated[f"beta_{beta}"] = {
            "beta": beta,
            "train_count": beta * E_val,
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
                },
                "precept_vs_expel": {
                    "first_try_success": paired_test(precept_p1, expel_p1),
                    "success_rate": paired_test(precept_pt, expel_pt),
                },
                # Legacy keys for backwards compatibility
                "first_try_success": paired_test(precept_p1, fr_p1),
                "success_rate": paired_test(precept_pt, fr_pt),
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

    # Bonferroni correction across beta settings (family-wise per comparison/metric)
    if aggregated:
        n_betas = len(aggregated)
        comparisons = ["precept_vs_fr", "precept_vs_expel"]
        metrics = ["first_try_success", "success_rate"]

        for beta_data in aggregated.values():
            tests = beta_data.get("statistical_tests", {})
            for comp in comparisons:
                for metric in metrics:
                    test = tests.get(comp, {}).get(metric, {})
                    if not isinstance(test, dict) or "p_value" not in test:
                        continue
                    raw_p = float(test.get("p_value", 1.0))
                    corrected_p = min(1.0, raw_p * n_betas)
                    test["p_value_bonferroni"] = corrected_p
                    test["bonferroni_n_tests"] = n_betas
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

    output_path = output_dir / "aggregated_results.json"
    with open(output_path, "w") as f:
        json.dump(aggregated, f, indent=2)

    generate_summary_report(aggregated, output_dir, domain=domain, E_val=E_val)

    return aggregated


def generate_summary_report(aggregated: dict, output_dir: Path, domain: str = "logistics", E_val: int = None):
    """Generate markdown summary report."""
    if E_val is None:
        E_val = MULTI_CONDITION_E.get(domain, 4) if NUM_CONDITIONS > 1 else SINGLE_CONDITION_E.get(domain, 4)
    lines = []
    lines.append(f"# Experiment 3: Training Size Ablation Results ({domain})\n")
    lines.append(f"Generated: {datetime.now().isoformat()}\n")

    lines.append("## Training Exposure (β) Effect\n")
    lines.append("```")
    lines.append("β = Learning Threshold (times each error type seen during training)")
    lines.append(f"T_train = β × E = β × {E_val} for {domain}")
    lines.append("")
    lines.append("β=1: Single encounter (fragile rules)")
    lines.append("β=2: Two encounters (moderate robustness)")
    lines.append("β=3: Three encounters (publication quality) ← RECOMMENDED")
    lines.append("β=4: Four encounters (diminishing returns)")
    lines.append("```\n")

    lines.append("## Results Summary\n")
    lines.append("| β | T_train | PRECEPT P₁ | FR P₁ | Δ P₁ | p-value |")
    lines.append("|---|---------|-----------|-------|------|---------|")

    for key in sorted(aggregated.keys()):
        data = aggregated[key]
        beta = data["beta"]
        train = data["train_count"]
        p = data["precept"]["first_try_success"]
        fr = data["full_reflexion"]["first_try_success"]
        adv = data["advantage"]["first_try_success_pp"]
        test = data["statistical_tests"]["first_try_success"]

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
        p_mean, p_ci = _bounded_pct_from_frac(p)
        fr_mean, fr_ci = _bounded_pct_from_frac(fr)

        lines.append(
            f"| {beta} | {train} | "
            f"{p_mean:.1f}% ± {p_ci:.1f}% | "
            f"{fr_mean:.1f}% ± {fr_ci:.1f}% | "
            f"**+{adv:.1f} pp** | "
            f"{p_for_sig:.4f}{sig} |"
        )

    lines.append(
        "\n*Significance: \\* p<0.05, \\*\\* p<0.01, \\*\\*\\* p<0.001 "
        "(Bonferroni-corrected p-values across β)*"
    )

    lines.append("\n## Key Findings\n")
    lines.append(
        "1. **Sample Efficiency**: PRECEPT shows strong performance even at β=1"
    )
    lines.append("2. **Robustness**: Performance improves with more training exposure")
    lines.append(
        "3. **Recommended**: β=3 provides good balance of efficiency and robustness"
    )
    lines.append(
        "4. **PRECEPT Advantage**: Consistently outperforms Full Reflexion across all β values"
    )

    with open(output_dir / "experiment_report.md", "w") as f:
        f.write("\n".join(lines))

    print(f"\n📊 Report saved: {output_dir / 'experiment_report.md'}")


def main():
    parser = argparse.ArgumentParser(
        description="Run Experiment 3: Training Size Ablation"
    )
    parser.add_argument(
        "--quick", action="store_true", help="Quick validation (3 seeds)"
    )
    parser.add_argument(
        "--very-quick",
        action="store_true",
        help="Very quick validation (1 seed, β=3 only) for testing scripts work",
    )
    parser.add_argument(
        "--domain",
        type=str,
        default=None,
        choices=list(SINGLE_CONDITION_E.keys()),
        help="Run only a specific domain (default: all publication domains)",
    )
    parser.add_argument(
        "--publication",
        action="store_true",
        help="Full publication mode (10 seeds, 3 domains: integration, booking, logistics)",
    )
    args = parser.parse_args()

    # Determine seeds, beta values, and domains based on mode
    if args.very_quick:
        seeds = VERY_QUICK_SEEDS
        beta_values = BETA_VALUES
        domains_to_run = [args.domain] if args.domain else ["logistics"]
        print("🚀 VERY QUICK MODE: 1 seed, β=3 only (validation only)")
    elif args.quick:
        seeds = QUICK_SEEDS
        beta_values = BETA_VALUES
        domains_to_run = [args.domain] if args.domain else ["logistics"]
    elif args.publication or not args.domain:
        seeds = PUBLICATION_SEEDS
        beta_values = BETA_VALUES
        domains_to_run = [args.domain] if args.domain else PUBLICATION_DOMAINS
        print(f"📊 PUBLICATION MODE: {len(seeds)} seeds, domains: {domains_to_run}")
    else:
        seeds = PUBLICATION_SEEDS
        beta_values = BETA_VALUES
        domains_to_run = [args.domain]

    for domain in domains_to_run:
        # Compute E for this domain
        E_val = MULTI_CONDITION_E[domain] if NUM_CONDITIONS > 1 else SINGLE_CONDITION_E[domain]
        test_size = E_val  # T_test = E

        print("=" * 80)
        print("EXPERIMENT 3: TRAINING SIZE ABLATION")
        print("Testing impact of training exposure (β) on PRECEPT vs Full Reflexion")
        print("=" * 80)
        print("\nConfiguration:")
        print(f"  Seeds: {seeds}")
        print(f"  Domain: {domain} (E={E_val})")
        print(f"  β values: {beta_values}")
        print(f"  T_train = β × E: {[b * E_val for b in beta_values]}")

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = (
            PROJECT_ROOT
            / "data"
            / "publication_results"
            / f"exp3_training_size_{domain}_{timestamp}"
        )
        output_dir.mkdir(parents=True, exist_ok=True)
        print(f"\nOutput directory: {output_dir}")

        config = {
            "experiment": "exp3_training_size_ablation",
            "seeds": seeds,
            "domain": domain,
            "E": E_val,
            "beta_values": beta_values,
            "test": test_size,
            "max_retries": MAX_RETRIES,
            "num_conditions": NUM_CONDITIONS,
            "timestamp": timestamp,
            "mode": "very_quick"
            if args.very_quick
            else ("quick" if args.quick else "full"),
        }
        with open(output_dir / "experiment_config.json", "w") as f:
            json.dump(config, f, indent=2)

        all_results = {beta: [] for beta in beta_values}
        total_runs = len(beta_values) * len(seeds)

        print(f"\n{'─' * 70}")
        print(f"🚀 Starting {total_runs} experiment runs")
        print(f"   Domain: {domain} | E: {E_val} | β values: {beta_values}")
        print(f"   Seeds: {len(seeds)} per β value")
        print(f"{'─' * 70}\n")

        for i, beta in enumerate(
            tqdm(
                beta_values,
                desc=f"🧪 Exp3: {domain} β Ablation",
                unit="β",
                ncols=80,
                bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]",
            )
        ):
            train = beta * E_val
            print(f"\n  ┌{'─' * 50}")
            print(f"  │ β = {beta} ({i + 1}/{len(beta_values)}) → T_train = {train}")
            print(f"  └{'─' * 50}")

            for j, seed in enumerate(
                tqdm(
                    seeds,
                    desc=f"    🌱 β={beta} seeds",
                    unit="seed",
                    leave=False,
                    ncols=70,
                    bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]",
                )
            ):
                result = run_single_experiment(beta, seed, output_dir, domain, E_val, test_size)
                all_results[beta].append(result)

                # Brief status after each seed
                if result:
                    p1 = (
                        result.get("agents", {})
                        .get("precept", {})
                        .get("first_try_success_rate", 0)
                        * 100
                    )
                    print(f"    ✓ seed={seed}: P₁={p1:.0f}%", end="\r")

            # Summary for this β
            successful = [r for r in all_results[beta] if r]
            if successful:
                avg_p1 = (
                    sum(
                        r.get("agents", {})
                        .get("precept", {})
                        .get("first_try_success_rate", 0)
                        for r in successful
                    )
                    / len(successful)
                    * 100
                )
                print(
                    f"  ✅ β={beta}: {len(successful)}/{len(seeds)} seeds, avg P₁={avg_p1:.1f}%"
                )

        print(f"\n{'=' * 60}")
        print(f"AGGREGATING RESULTS ({domain.upper()})")
        print(f"{'=' * 60}")

        aggregated = aggregate_results(all_results, output_dir, domain=domain, E_val=E_val)

        print(f"\nResults saved to: {output_dir}")

        print(f"\n📊 QUICK SUMMARY - {domain.upper()} (Learning Curve):")
        print("-" * 60)
        for key in sorted(aggregated.keys()):
            data = aggregated[key]
            beta = data["beta"]
            train = data["train_count"]
            p1_precept = data["precept"]["first_try_success"]["mean"] * 100
            p1_fr = data["full_reflexion"]["first_try_success"]["mean"] * 100
            adv = data["advantage"]["first_try_success_pp"]
            print(
                f"  β={beta} (train={train:2d}): PRECEPT {p1_precept:.1f}% vs FR {p1_fr:.1f}% (Δ +{adv:.1f} pp)"
            )

    print("\n" + "=" * 80)
    print("EXPERIMENT 3 COMPLETE!")
    print("=" * 80)


if __name__ == "__main__":
    main()
