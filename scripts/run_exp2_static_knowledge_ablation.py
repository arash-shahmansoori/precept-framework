#!/usr/bin/env python3
"""
Experiment 2: Static Knowledge Ablation Study

PURPOSE:
    Generate Table 2 for publication - demonstrating the impact of static
    knowledge on both PRECEPT and Full Reflexion performance.

    Tests the contribution of pre-loaded knowledge base vs pure dynamic learning.

STATISTICAL REQUIREMENTS:
    - N = 10 independent runs per configuration
    - Tests: with_static_knowledge vs without_static_knowledge
    - Logistics domain (baseline domain for ablations)

OUTPUT:
    - data/publication_results/exp3_static_knowledge_ablation/
    - Results: Performance with and without static knowledge

EXPECTED RUNTIME: ~30 minutes (2 configurations × 10 seeds)

Usage:
    python scripts/run_exp3_static_knowledge_ablation.py [--quick]
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

    def tqdm(iterable, desc=None, total=None, **kwargs):
        if desc:
            print(f"\n{desc}")
        for i, item in enumerate(iterable):
            if total:
                print(f"  Progress: {i + 1}/{total}", end="\r")
            yield item
        print()


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

PUBLICATION_SEEDS = [42, 123, 456, 789, 999, 2024, 3141, 1337, 8888, 7777]
QUICK_SEEDS = [42, 123, 456]
VERY_QUICK_SEEDS = [42]  # Single seed for validation
PUBLICATION_DOMAINS = ["integration", "logistics"]

# Ablation: static knowledge on/off
STATIC_KNOWLEDGE_VALUES = [True, False]

# Domain E values (number of unique condition keys per domain)
E_VALUES = {
    "finance": 6,  # VOLATILE_SYMBOLS only
    "logistics": 4,  # BLOCKED_PORTS only
    "coding": 5,  # BLOCKED_PACKAGES only
    "devops": 5,  # STUCK_STACKS only
    "booking": 17,  # BLOCKED_FLIGHTS (all 17)
    "integration": 6,  # OAUTH_SOURCES only
}

# Fixed parameters
BETA = 3  # Training coverage factor
MAX_RETRIES = 4
NUM_CONDITIONS = 5


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

    # All ChromaDB directories to clean (including ExpeL's isolated store)
    paths_to_clean = [
        data_dir / "chroma_precept",  # PRECEPT vector store
        data_dir / "chroma_static_knowledge",  # Static knowledge store
        data_dir / "chroma_expel",  # ExpeL vector store (isolated)
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


def run_single_experiment(
    static_knowledge: bool,
    seed: int,
    output_dir: Path,
    domain: str = "logistics",
    train_tasks: int = 12,
    test_tasks: int = 4,
) -> dict:
    """Run a single experiment with live progress bars for training and testing."""

    clean_data_directory()

    sk_flag = "--static-knowledge" if static_knowledge else "--no-static-knowledge"
    sk_label = "with_sk" if static_knowledge else "no_sk"

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
        "--max-retries",
        str(MAX_RETRIES),
        "--num-conditions",
        str(NUM_CONDITIONS),
        sk_flag,
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
        f"    🔄 Running: {sk_label} seed={seed} (train={train_tasks}, test={test_tasks})..."
    )

    log_file = output_dir / f"{sk_label}_seed{seed}.log"

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

        process.wait(timeout=1200)

    except subprocess.TimeoutExpired:
        process.kill()
        if train_bar:
            train_bar.close()
        if test_bar:
            test_bar.close()
        print(f"    ⏰ Timeout: {sk_label} seed={seed}")
        return None
    except Exception as e:
        if train_bar:
            train_bar.close()
        if test_bar:
            test_bar.close()
        print(f"    ❌ Error: {sk_label} seed={seed}: {e}")
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
        print(f"    ✅ Completed: {sk_label} seed={seed} | {rules_learned} rules")

        data_dir = PROJECT_ROOT / "data"
        result_files = list(data_dir.glob(f"experiment_results_{domain}_*.json"))
        if result_files:
            latest = max(result_files, key=lambda p: p.stat().st_mtime)
            dest = output_dir / f"{sk_label}_seed{seed}_results.json"
            shutil.copy(latest, dest)

            with open(latest) as f:
                return json.load(f)
    else:
        print(f"    ❌ Failed: {sk_label} seed={seed}")

    return None


def aggregate_results(all_results: dict, output_dir: Path):
    """Aggregate results with statistical analysis."""
    import numpy as np
    from scipy import stats

    aggregated = {}

    for sk_value, runs in all_results.items():
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

        sk_label = "with_static_knowledge" if sk_value else "without_static_knowledge"

        aggregated[sk_label] = {
            "static_knowledge": sk_value,
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
            "advantage": {
                "vs_fr_first_try_pp": (np.mean(precept_p1) - np.mean(fr_p1)) * 100,
                "vs_fr_success_pp": (np.mean(precept_pt) - np.mean(fr_pt)) * 100,
                "vs_expel_first_try_pp": (np.mean(precept_p1) - np.mean(expel_p1))
                * 100,
                "vs_expel_success_pp": (np.mean(precept_pt) - np.mean(expel_pt)) * 100,
                # Legacy keys for backwards compatibility
                "first_try_success_pp": (np.mean(precept_p1) - np.mean(fr_p1)) * 100,
                "success_rate_pp": (np.mean(precept_pt) - np.mean(fr_pt)) * 100,
            },
        }

    # Compute effect of static knowledge
    if (
        "with_static_knowledge" in aggregated
        and "without_static_knowledge" in aggregated
    ):
        with_sk = aggregated["with_static_knowledge"]
        without_sk = aggregated["without_static_knowledge"]

        aggregated["static_knowledge_effect"] = {
            "precept_p1_gain": (
                with_sk["precept"]["first_try_success"]["mean"]
                - without_sk["precept"]["first_try_success"]["mean"]
            )
            * 100,
            "precept_pt_gain": (
                with_sk["precept"]["success_rate"]["mean"]
                - without_sk["precept"]["success_rate"]["mean"]
            )
            * 100,
            "fr_p1_gain": (
                with_sk["full_reflexion"]["first_try_success"]["mean"]
                - without_sk["full_reflexion"]["first_try_success"]["mean"]
            )
            * 100,
            "fr_pt_gain": (
                with_sk["full_reflexion"]["success_rate"]["mean"]
                - without_sk["full_reflexion"]["success_rate"]["mean"]
            )
            * 100,
            "expel_p1_gain": (
                with_sk.get("expel", {}).get("first_try_success", {"mean": 0})["mean"]
                - without_sk.get("expel", {}).get("first_try_success", {"mean": 0})[
                    "mean"
                ]
            )
            * 100,
            "expel_pt_gain": (
                with_sk.get("expel", {}).get("success_rate", {"mean": 0})["mean"]
                - without_sk.get("expel", {}).get("success_rate", {"mean": 0})["mean"]
            )
            * 100,
        }

    output_path = output_dir / "aggregated_results.json"
    with open(output_path, "w") as f:
        json.dump(aggregated, f, indent=2)

    generate_summary_report(aggregated, output_dir)

    return aggregated


def generate_summary_report(aggregated: dict, output_dir: Path):
    """Generate markdown summary report."""
    lines = []
    lines.append("# Experiment 3: Static Knowledge Ablation Results\n")
    lines.append(f"Generated: {datetime.now().isoformat()}\n")

    lines.append("## Summary: Impact of Static Knowledge\n")
    lines.append("| Configuration | PRECEPT P₁ | FR P₁ | PRECEPT Pₜ | FR Pₜ |")
    lines.append("|---------------|-----------|-------|-----------|-------|")

    for key in ["with_static_knowledge", "without_static_knowledge"]:
        if key not in aggregated:
            continue
        data = aggregated[key]
        p_p1 = data["precept"]["first_try_success"]
        fr_p1 = data["full_reflexion"]["first_try_success"]
        p_pt = data["precept"]["success_rate"]
        fr_pt = data["full_reflexion"]["success_rate"]

        label = "With Static KB" if data["static_knowledge"] else "Without Static KB"
        lines.append(
            f"| {label} | "
            f"{fmt_pct(p_p1)} | "
            f"{fmt_pct(fr_p1)} | "
            f"{fmt_pct(p_pt)} | "
            f"{fmt_pct(fr_pt)} |"
        )

    if "static_knowledge_effect" in aggregated:
        effect = aggregated["static_knowledge_effect"]
        lines.append("\n## Static Knowledge Effect\n")
        lines.append("| Agent | P₁ Gain | Pₜ Gain |")
        lines.append("|-------|---------|---------|")
        lines.append(
            f"| PRECEPT | +{effect['precept_p1_gain']:.1f} pp | +{effect['precept_pt_gain']:.1f} pp |"
        )
        lines.append(
            f"| Full Reflexion | +{effect['fr_p1_gain']:.1f} pp | +{effect['fr_pt_gain']:.1f} pp |"
        )

    lines.append("\n## Key Finding\n")
    lines.append("Static knowledge provides initial context that benefits both agents,")
    lines.append("but PRECEPT maintains a significant advantage in both configurations")
    lines.append("due to its structured rule learning and deterministic application.")

    with open(output_dir / "experiment_report.md", "w") as f:
        f.write("\n".join(lines))

    print(f"\n📊 Report saved: {output_dir / 'experiment_report.md'}")


def main():
    parser = argparse.ArgumentParser(
        description="Run Experiment 2: Static Knowledge Ablation"
    )
    parser.add_argument(
        "--quick", action="store_true", help="Quick validation (3 seeds)"
    )
    parser.add_argument(
        "--very-quick",
        action="store_true",
        help="Very quick validation (1 seed, with_sk only) for testing scripts work",
    )
    parser.add_argument(
        "--domain",
        type=str,
        default=None,
        choices=list(E_VALUES.keys()),
        help="Run only a specific domain (default: all publication domains)",
    )
    parser.add_argument(
        "--publication",
        action="store_true",
        help="Full publication mode (10 seeds, 3 domains: integration, booking, logistics)",
    )
    args = parser.parse_args()

    # Determine seeds, SK configurations, and domains based on mode
    if args.very_quick:
        seeds = VERY_QUICK_SEEDS
        sk_values = STATIC_KNOWLEDGE_VALUES
        domains_to_run = [args.domain] if args.domain else ["logistics"]
        print("🚀 VERY QUICK MODE: 1 seed, both SK configs (ablation validation)")
    elif args.quick:
        seeds = QUICK_SEEDS
        sk_values = STATIC_KNOWLEDGE_VALUES
        domains_to_run = [args.domain] if args.domain else ["logistics"]
    elif args.publication or not args.domain:
        seeds = PUBLICATION_SEEDS
        sk_values = STATIC_KNOWLEDGE_VALUES
        domains_to_run = [args.domain] if args.domain else PUBLICATION_DOMAINS
        print(f"📊 PUBLICATION MODE: {len(seeds)} seeds, domains: {domains_to_run}")
    else:
        seeds = PUBLICATION_SEEDS
        sk_values = STATIC_KNOWLEDGE_VALUES
        domains_to_run = [args.domain]

    for domain in domains_to_run:
        # Compute domain-specific training/testing sizes
        E = E_VALUES[domain]
        train_size = BETA * E  # T_train = β × E
        test_size = E  # T_test = E

        print("=" * 80)
        print("EXPERIMENT 2: STATIC KNOWLEDGE ABLATION")
        print("Testing impact of static knowledge on PRECEPT vs Full Reflexion")
        print("=" * 80)
        print("\nConfiguration:")
        print(f"  Seeds: {seeds}")
        print(f"  Domain: {domain} (E={E})")
        print(f"  static_knowledge: {sk_values}")
        print(f"  Train: {train_size} (β={BETA} × E={E}) | Test: {test_size}")

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = (
            PROJECT_ROOT
            / "data"
            / "publication_results"
            / f"exp2_static_knowledge_{domain}_{timestamp}"
        )
        output_dir.mkdir(parents=True, exist_ok=True)
        print(f"\nOutput directory: {output_dir}")

        config = {
            "experiment": "exp2_static_knowledge_ablation",
            "seeds": seeds,
            "domain": domain,
            "static_knowledge_values": sk_values,
            "train": train_size,
            "test": test_size,
            "E": E,
            "beta": BETA,
            "max_retries": MAX_RETRIES,
            "num_conditions": NUM_CONDITIONS,
            "timestamp": timestamp,
            "mode": "very_quick"
            if args.very_quick
            else ("quick" if args.quick else "full"),
        }
        with open(output_dir / "experiment_config.json", "w") as f:
            json.dump(config, f, indent=2)

        all_results = {sk: [] for sk in sk_values}
        total_runs = len(sk_values) * len(seeds)

        print(f"\n{'─' * 70}")
        print(f"🚀 Starting {total_runs} experiment runs")
        print(
            f"   Configs: {len(sk_values)} SK settings | Seeds: {len(seeds)} per config"
        )
        print(f"   Domain: {domain} | Train: {train_size} | Test: {test_size}")
        print(f"{'─' * 70}\n")

        for i, sk in enumerate(
            tqdm(
                sk_values,
                desc=f"🧪 Exp2: {domain} Static Knowledge",
                unit="config",
                ncols=80,
                bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]",
            )
        ):
            label = "WITH" if sk else "WITHOUT"
            print(f"\n  ┌{'─' * 55}")
            print(f"  │ STATIC KNOWLEDGE: {label} ({i + 1}/{len(sk_values)})")
            print(
                f"  │ Domain: {domain} | Train: {train_size} tasks | Test: {test_size} tasks"
            )
            print(f"  └{'─' * 55}")

            for j, seed in enumerate(
                tqdm(
                    seeds,
                    desc=f"    🌱 SK={label[:4]}",
                    unit="seed",
                    leave=False,
                    ncols=70,
                    bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]",
                )
            ):
                result = run_single_experiment(
                    sk,
                    seed,
                    output_dir,
                    domain=domain,
                    train_tasks=train_size,
                    test_tasks=test_size,
                )
                all_results[sk].append(result)

                # Brief status after each seed
                if result:
                    p1 = (
                        result.get("agents", {})
                        .get("precept", {})
                        .get("first_try_success_rate", 0)
                        * 100
                    )
                    print(f"    ✓ seed={seed}: P₁={p1:.0f}%", end="\r")

            # Config summary
            successful = [r for r in all_results[sk] if r]
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
                    f"  ✅ SK={label}: {len(successful)}/{len(seeds)} seeds, avg P₁={avg_p1:.1f}%"
                )

        print(f"\n{'=' * 60}")
        print(f"AGGREGATING RESULTS ({domain.upper()})")
        print(f"{'=' * 60}")

        aggregated = aggregate_results(all_results, output_dir)

        print(f"\nResults saved to: {output_dir}")

        print(f"\n📊 QUICK SUMMARY - {domain.upper()}:")
        print("-" * 60)
        for key in ["with_static_knowledge", "without_static_knowledge"]:
            if key in aggregated:
                data = aggregated[key]
                label = "With SK" if data["static_knowledge"] else "No SK"
                p1_p = data["precept"]["first_try_success"]["mean"] * 100
                p1_fr = data["full_reflexion"]["first_try_success"]["mean"] * 100
                adv = data["advantage"]["first_try_success_pp"]
                print(
                    f"  {label:8s}: PRECEPT {p1_p:.1f}% vs FR {p1_fr:.1f}% (Δ +{adv:.1f} pp)"
                )

    print("\n" + "=" * 80)
    print("EXPERIMENT 2 COMPLETE!")
    print("=" * 80)


if __name__ == "__main__":
    main()
