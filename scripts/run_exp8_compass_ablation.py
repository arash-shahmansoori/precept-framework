#!/usr/bin/env python3
"""
Experiment 8: COMPASS Prompt Evolution Ablation

PURPOSE:
    Isolate and validate COMPASS's outer-loop prompt evolution contribution,
    which is orthogonal to the inner-loop mechanisms validated in Experiments 1-7.

    This experiment compares four PRECEPT configurations:
      A. PRECEPT (full)        - All components active (baseline)
      B. PRECEPT (no COMPASS)  - COMPASS evolution disabled; static base prompt + rule injection
      C. PRECEPT (no rules)    - COMPASS evolution active but no learned rules baked in
      D. PRECEPT (base only)   - Neither COMPASS evolution nor rule injection; bare base prompt

    The delta between A and B isolates COMPASS's prompt evolution contribution.
    The delta between A and D shows the combined value of rules + evolution.
    B vs D isolates the value of deterministic rule injection alone.

DESIGN:
    - Uses the same integration domain as Exp 1 (hardest domain, E=6, N=5)
    - Training phase identical across all configs (β=3, same seeds)
    - Testing phase uses matched keys only (isolates prompt quality)
    - COMPASS env vars control ablation at the MCP server level

METRICS:
    - P₁ (first-try success rate)
    - Pₜ (overall success rate)
    - Avg Steps
    - COMPASS evolution events (mutations, Pareto front size, generations)

STATISTICAL REQUIREMENTS:
    - N = 10 independent runs per configuration (seeds: 42-7777)
    - Reports mean ± 95% CI, paired t-tests, Cohen's d

OUTPUT:
    - data/publication_results/exp8_compass_ablation_TIMESTAMP/

EXPECTED RUNTIME: ~3-4 hours (4 configs × 10 seeds × train + test)

Usage:
    python scripts/run_exp8_compass_ablation.py [--quick] [--very-quick]

    --quick: Run with 3 seeds instead of 10
    --very-quick: Run with 1 seed for validation
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

# ═══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

PUBLICATION_SEEDS = [42, 123, 456, 789, 999, 2024, 3141, 1337, 8888, 7777]
QUICK_SEEDS = [42, 123, 456]
VERY_QUICK_SEEDS = [42]

DOMAIN = "integration"
E = 6
NUM_CONDITIONS = 5
MAX_RETRIES = 4
BETA = 3
TRAIN_TASKS = BETA * E
TEST_TASKS = E

# Four ablation configurations
# Each is defined by env vars that control COMPASS behavior at the MCP server level
CONFIGS = {
    "full": {
        "label": "PRECEPT (full)",
        "desc": "All components: COMPASS evolution + learned rules",
        "env": {
            "PRECEPT_ENABLE_COMPASS": "1",
            "PRECEPT_INCLUDE_RULES_IN_PROMPT": "1",
        },
    },
    "no_compass": {
        "label": "PRECEPT (no COMPASS)",
        "desc": "COMPASS evolution disabled; static prompt + rule injection only",
        "env": {
            "PRECEPT_ENABLE_COMPASS": "0",
            "PRECEPT_INCLUDE_RULES_IN_PROMPT": "1",
        },
    },
    "no_rules": {
        "label": "PRECEPT (no rules in prompt)",
        "desc": "COMPASS evolution active but rules NOT baked into prompt",
        "env": {
            "PRECEPT_ENABLE_COMPASS": "1",
            "PRECEPT_INCLUDE_RULES_IN_PROMPT": "0",
        },
    },
    "base_only": {
        "label": "PRECEPT (base prompt only)",
        "desc": "Neither COMPASS evolution nor rule injection; bare base prompt",
        "env": {
            "PRECEPT_ENABLE_COMPASS": "0",
            "PRECEPT_INCLUDE_RULES_IN_PROMPT": "0",
        },
    },
}


def clean_data_directory():
    """Remove ALL persisted data to ensure zero contamination between runs.

    This must eliminate every state file that could leak learned knowledge
    between configs or seeds: vector stores, JSON state, baseline memory,
    and stale results. Static knowledge is NOT cleaned as it belongs to
    a separate experiment (Exp 6) and is not engaged here.
    """
    data_dir = PROJECT_ROOT / "data"

    # 1. Remove ChromaDB vector stores (learned embeddings, NOT static knowledge)
    for chroma_dir in data_dir.glob("chroma_*"):
        if chroma_dir.is_dir():
            shutil.rmtree(chroma_dir)
            print(f"  🧹 Removed: {chroma_dir.name}/")

    # 2. Remove ALL PRECEPT JSON state files (rules, experiences, procedures, progress, etc.)
    for json_file in data_dir.glob("precept_*.json"):
        json_file.unlink()
        print(f"  🧹 Removed: {json_file.name}")

    # 3. Clean baseline persistence files
    baseline_files = [
        data_dir / "full_reflexion_memory.json",
        data_dir / "expel_insights.json",
    ]
    for bf in baseline_files:
        if bf.exists():
            bf.unlink()
            print(f"    🧹 Cleaned: {bf.name}")

    # 4. Remove stale experiment results
    stale_results = list(data_dir.glob("experiment_results_*.json"))
    if stale_results:
        for f in stale_results:
            f.unlink()
        print(f"  🧹 Removed {len(stale_results)} stale result files")


def run_single_config(config_name: str, config: dict, seed: int, output_dir: Path) -> dict:
    """Run a single PRECEPT experiment with specific COMPASS ablation settings."""
    clean_data_directory()

    cmd = [
        "uv", "run", "examples/precept_autogen_mcp_full.py",
        "--domain", DOMAIN,
        "--train", str(TRAIN_TASKS),
        "--test", str(TEST_TASKS),
        "--test-mode", "matched",
        "--max-retries", str(MAX_RETRIES),
        "--num-conditions", str(NUM_CONDITIONS),
        "--no-static-knowledge",
        "--hybrid-retrieval",
        "--seed", str(seed),
        "-ct", "-tw", "4", "-c", "-w", "4", "-v",
    ]

    print(f"    🔄 [{config_name}] seed={seed}...")

    log_file = output_dir / f"{config_name}_seed{seed}.log"
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    for k, v in config["env"].items():
        env[k] = v

    process = subprocess.Popen(
        cmd, cwd=PROJECT_ROOT,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, env=env, bufsize=1,
    )

    output_lines = []
    compass_events = {"mutations": 0, "pareto_updates": 0, "gepa_triggers": 0, "generations": 0}

    try:
        for line in process.stdout:
            output_lines.append(line)
            if "GEPA PROMPT EVOLUTION TRIGGERED" in line:
                compass_events["gepa_triggers"] += 1
            if "EVOLVED PROMPT" in line and "Gen " in line:
                compass_events["generations"] += 1
            if "Mutation Type:" in line:
                compass_events["mutations"] += 1
            if "PARETO FRONT" in line:
                compass_events["pareto_updates"] += 1

        process.wait(timeout=1800)
    except subprocess.TimeoutExpired:
        process.kill()
        print(f"    ⏰ Timeout: [{config_name}] seed={seed}")
        return None
    except Exception as e:
        print(f"    ❌ Error: [{config_name}] seed={seed}: {e}")
        return None

    with open(log_file, "w") as f:
        f.write(f"Command: {' '.join(cmd)}\n")
        f.write(f"Config: {config_name} ({config['desc']})\n")
        f.write(f"Env overrides: {config['env']}\n")
        f.write(f"Exit code: {process.returncode}\n")
        f.write(f"COMPASS events: {json.dumps(compass_events)}\n")
        f.write("=" * 80 + "\n")
        f.writelines(output_lines)

    if process.returncode == 0:
        print(f"    ✅ [{config_name}] seed={seed} | GEPA={compass_events['gepa_triggers']} gens={compass_events['generations']}")

        data_dir = PROJECT_ROOT / "data"
        result_files = list(data_dir.glob(f"experiment_results_{DOMAIN}_*.json"))
        if result_files:
            latest = max(result_files, key=lambda p: p.stat().st_mtime)
            dest = output_dir / f"{config_name}_seed{seed}_results.json"
            shutil.copy(latest, dest)

            with open(latest) as f:
                result = json.load(f)
                result["compass_events"] = compass_events
                result["config_name"] = config_name
                return result
    else:
        print(f"    ❌ [{config_name}] seed={seed} (exit={process.returncode})")

    return None


def aggregate_results(all_results: dict, output_dir: Path):
    """Aggregate results across all configs with statistical analysis."""
    import numpy as np
    from scipy import stats

    aggregated = {}

    for config_name, runs in all_results.items():
        valid_runs = [r for r in runs if r is not None]
        if not valid_runs:
            continue

        p1_values = []
        pt_values = []
        steps_values = []
        compass_triggers = []

        for r in valid_runs:
            agents = r.get("agents", {})
            precept = agents.get("precept", {})
            if not precept:
                continue

            p1 = precept.get("first_try_success_rate", 0)
            pt = precept.get("success_rate", 0)
            avg_steps = precept.get("avg_steps", 0)

            p1_values.append(p1)
            pt_values.append(pt)
            steps_values.append(avg_steps)
            compass_triggers.append(r.get("compass_events", {}).get("gepa_triggers", 0))

        n = len(p1_values)
        if n == 0:
            continue

        p1_arr = np.array(p1_values)
        pt_arr = np.array(pt_values)
        steps_arr = np.array(steps_values)

        t_crit = stats.t.ppf(0.975, df=max(n - 1, 1))

        aggregated[config_name] = {
            "label": CONFIGS[config_name]["label"],
            "n": n,
            "p1": {
                "mean": float(np.mean(p1_arr)),
                "std": float(np.std(p1_arr, ddof=1)) if n > 1 else 0,
                "ci_95": float(t_crit * np.std(p1_arr, ddof=1) / np.sqrt(n)) if n > 1 else 0,
            },
            "pt": {
                "mean": float(np.mean(pt_arr)),
                "std": float(np.std(pt_arr, ddof=1)) if n > 1 else 0,
                "ci_95": float(t_crit * np.std(pt_arr, ddof=1) / np.sqrt(n)) if n > 1 else 0,
            },
            "steps": {
                "mean": float(np.mean(steps_arr)),
                "std": float(np.std(steps_arr, ddof=1)) if n > 1 else 0,
                "ci_95": float(t_crit * np.std(steps_arr, ddof=1) / np.sqrt(n)) if n > 1 else 0,
            },
            "compass_triggers_mean": float(np.mean(compass_triggers)),
        }

    # Pairwise comparisons: full vs each ablation
    comparisons = {}
    if "full" in all_results and len([r for r in all_results["full"] if r]) >= 2:
        full_p1 = [r.get("agents", {}).get("precept", {}).get("first_try_success_rate", 0)
                    for r in all_results["full"] if r is not None]

        for ablation in ["no_compass", "no_rules", "base_only"]:
            if ablation not in all_results:
                continue
            ablation_p1 = [r.get("agents", {}).get("precept", {}).get("first_try_success_rate", 0)
                           for r in all_results[ablation] if r is not None]

            n_min = min(len(full_p1), len(ablation_p1))
            if n_min >= 2:
                t_stat, p_val = stats.ttest_rel(full_p1[:n_min], ablation_p1[:n_min])
                pooled_std = np.sqrt((np.std(full_p1[:n_min], ddof=1)**2 + np.std(ablation_p1[:n_min], ddof=1)**2) / 2)
                cohens_d = (np.mean(full_p1[:n_min]) - np.mean(ablation_p1[:n_min])) / pooled_std if pooled_std > 0 else 0

                comparisons[f"full_vs_{ablation}"] = {
                    "delta_pp": float((np.mean(full_p1[:n_min]) - np.mean(ablation_p1[:n_min])) * 100),
                    "t_stat": float(t_stat),
                    "p_value": float(p_val),
                    "cohens_d": float(cohens_d),
                    "n": n_min,
                }

    results_json = {
        "experiment": "exp8_compass_ablation",
        "domain": DOMAIN,
        "timestamp": datetime.now().isoformat(),
        "config": {
            "E": E, "N": NUM_CONDITIONS, "beta": BETA,
            "max_retries": MAX_RETRIES, "test_mode": "matched",
        },
        "aggregated": aggregated,
        "comparisons": comparisons,
    }

    out_path = output_dir / "aggregated_results.json"
    with open(out_path, "w") as f:
        json.dump(results_json, f, indent=2)

    # Print summary
    print("\n" + "=" * 80)
    print("EXPERIMENT 8: COMPASS ABLATION RESULTS")
    print("=" * 80)

    print(f"\nDomain: {DOMAIN} (E={E}, N={NUM_CONDITIONS}, β={BETA})")
    print(f"{'Config':<30} {'P₁':>12} {'Pₜ':>12} {'Steps':>10} {'GEPA':>6}")
    print("-" * 72)

    for cfg_name in ["full", "no_compass", "no_rules", "base_only"]:
        if cfg_name in aggregated:
            a = aggregated[cfg_name]
            p1_m, p1_c = a["p1"]["mean"] * 100, a["p1"]["ci_95"] * 100
            pt_m, pt_c = a["pt"]["mean"] * 100, a["pt"]["ci_95"] * 100
            s_m, s_c = a["steps"]["mean"], a["steps"]["ci_95"]
            gepa = a["compass_triggers_mean"]
            print(f"{a['label']:<30} {p1_m:5.1f}±{p1_c:4.1f}%  {pt_m:5.1f}±{pt_c:4.1f}%  {s_m:5.2f}±{s_c:4.2f}  {gepa:5.1f}")

    if comparisons:
        print(f"\n{'Comparison':<30} {'Δ P₁':>8} {'t':>8} {'p':>10} {'d':>6}")
        print("-" * 65)
        for comp_name, comp in comparisons.items():
            label = comp_name.replace("full_vs_", "full vs ")
            sig = "***" if comp["p_value"] < 0.001 else "**" if comp["p_value"] < 0.01 else "*" if comp["p_value"] < 0.05 else "n.s."
            print(f"{label:<30} {comp['delta_pp']:+6.1f}pp  {comp['t_stat']:7.2f}  {comp['p_value']:9.4f}  {comp['cohens_d']:5.2f} {sig}")

    # Write markdown report
    report_path = output_dir / "experiment_report.md"
    with open(report_path, "w") as f:
        f.write("# Experiment 8: COMPASS Prompt Evolution Ablation\n\n")
        f.write(f"**Domain:** {DOMAIN} | **E:** {E} | **N:** {NUM_CONDITIONS} | **β:** {BETA}\n\n")
        f.write("## Research Question\n\n")
        f.write("Does COMPASS's outer-loop prompt evolution (Pareto selection, smart rollouts, ")
        f.write("GEPA mutation) contribute measurable value beyond the deterministic rule ")
        f.write("compilation that occurs independently?\n\n")
        f.write("## Configurations\n\n")
        f.write("| Config | COMPASS Evolution | Rules in Prompt |\n")
        f.write("|--------|-------------------|------------------|\n")
        for cfg_name, cfg in CONFIGS.items():
            compass = "✓" if cfg["env"]["PRECEPT_ENABLE_COMPASS"] == "1" else "✗"
            rules = "✓" if cfg["env"]["PRECEPT_INCLUDE_RULES_IN_PROMPT"] == "1" else "✗"
            f.write(f"| {cfg['label']} | {compass} | {rules} |\n")
        f.write("\n## Results\n\n")
        f.write("| Config | P₁ | Pₜ | Steps | GEPA Triggers |\n")
        f.write("|--------|-----|-----|-------|---------------|\n")
        for cfg_name in ["full", "no_compass", "no_rules", "base_only"]:
            if cfg_name in aggregated:
                a = aggregated[cfg_name]
                p1_m, p1_c = a["p1"]["mean"] * 100, a["p1"]["ci_95"] * 100
                pt_m, pt_c = a["pt"]["mean"] * 100, a["pt"]["ci_95"] * 100
                s_m = a["steps"]["mean"]
                gepa = a["compass_triggers_mean"]
                f.write(f"| {a['label']} | {p1_m:.1f}±{p1_c:.1f}% | {pt_m:.1f}±{pt_c:.1f}% | {s_m:.2f} | {gepa:.1f} |\n")
        if comparisons:
            f.write("\n## Statistical Comparisons (vs Full PRECEPT)\n\n")
            f.write("| Comparison | Δ P₁ | t | p | Cohen's d |\n")
            f.write("|------------|-------|---|---|----------|\n")
            for comp_name, comp in comparisons.items():
                label = comp_name.replace("full_vs_", "vs ")
                f.write(f"| {label} | {comp['delta_pp']:+.1f}pp | {comp['t_stat']:.2f} | {comp['p_value']:.4f} | {comp['cohens_d']:.2f} |\n")

    print(f"\nResults saved to: {output_dir}")
    return results_json


def main():
    parser = argparse.ArgumentParser(description="Experiment 8: COMPASS Ablation")
    parser.add_argument("--quick", action="store_true", help="3 seeds instead of 10")
    parser.add_argument("--very-quick", action="store_true", help="1 seed for validation")
    args = parser.parse_args()

    if args.very_quick:
        seeds = VERY_QUICK_SEEDS
    elif args.quick:
        seeds = QUICK_SEEDS
    else:
        seeds = PUBLICATION_SEEDS

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = PROJECT_ROOT / "data" / "publication_results" / f"exp8_compass_ablation_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    exp_config = {
        "experiment": "exp8_compass_ablation",
        "domain": DOMAIN,
        "E": E,
        "N": NUM_CONDITIONS,
        "beta": BETA,
        "max_retries": MAX_RETRIES,
        "seeds": seeds,
        "configs": {k: v["desc"] for k, v in CONFIGS.items()},
        "timestamp": timestamp,
    }
    with open(output_dir / "experiment_config.json", "w") as f:
        json.dump(exp_config, f, indent=2)

    print("=" * 80)
    print("EXPERIMENT 8: COMPASS PROMPT EVOLUTION ABLATION")
    print("=" * 80)
    print(f"Domain: {DOMAIN} (E={E}, N={NUM_CONDITIONS}, β={BETA})")
    print(f"Seeds: {len(seeds)} ({seeds})")
    print(f"Configs: {len(CONFIGS)} ablation variants")
    print(f"Total runs: {len(seeds) * len(CONFIGS)}")
    print(f"Output: {output_dir}")
    print("=" * 80)

    all_results = {cfg_name: [] for cfg_name in CONFIGS}

    for cfg_name, cfg in CONFIGS.items():
        print(f"\n{'─' * 60}")
        print(f"  Config: {cfg['label']}")
        print(f"  {cfg['desc']}")
        print(f"{'─' * 60}")

        for seed in tqdm(seeds, desc=f"  {cfg['label']}", total=len(seeds)):
            result = run_single_config(cfg_name, cfg, seed, output_dir)
            all_results[cfg_name].append(result)

    aggregate_results(all_results, output_dir)


if __name__ == "__main__":
    main()
