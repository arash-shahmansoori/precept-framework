#!/usr/bin/env python3
"""
Experiment 9: COMPASS Stress-Test (Explicit OOD families + Post-Learning Reactivity)

Purpose:
    Evaluate where COMPASS outer-loop adaptation may help most:
      1) OOD / novelty-heavy scenarios
      2) higher semantic ambiguity
      3) faster post-learning policy shaping

Design:
    - Same 4 ablation configs as Exp 8
    - Multiple test regimes (including two explicit OOD families):
        * matched_control
        * ood_keyspace_random
        * ood_semantic_compositional
        * ood_semantic_compositional_hard
    - Adds TTFC-L metrics from execution traces:
        * mean_global_gap: tasks until first correct after learning a key
        * mean_samekey_gap: same-key re-encounters until first correct
"""

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

try:
    from tqdm import tqdm
except ImportError:
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

PUBLICATION_SEEDS = [42, 123, 456, 789, 999, 2024, 3141, 1337, 8888, 7777]
QUICK_SEEDS = [42, 123, 456]
VERY_QUICK_SEEDS = [42]

DOMAIN = "integration"
E = 6
NUM_CONDITIONS = 5
MAX_RETRIES = 4
BETA = 3
TRAIN_TASKS = BETA * E
# Larger default test set than Exp 8 to reduce metric quantization/ties in OOD.
TEST_TASKS = 24

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
        "desc": "COMPASS evolution active but rules not baked into prompt",
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

REGIMES = {
    "matched_control": {
        "label": "Matched Control",
        "desc": "Known keys, deterministic retrieval dominates",
        "train": TRAIN_TASKS,
        "test": TEST_TASKS,
        "args": ["--test-mode", "matched", "--num-conditions", str(NUM_CONDITIONS)],
    },
    "ood_keyspace_random": {
        "label": "OOD Keyspace Random",
        "desc": "Novel random keys/combinations (key-space OOD)",
        "train": TRAIN_TASKS,
        "test": TEST_TASKS,
        "args": ["--test-mode", "random", "--num-conditions", str(NUM_CONDITIONS)],
    },
    "ood_semantic_compositional": {
        "label": "OOD Semantic Compositional",
        "desc": "Semantic/compositional OOD: 1-condition train -> 3-condition test",
        "train": TRAIN_TASKS,
        "test": TEST_TASKS,
        "args": [
            "--semantic-compositional",
            "--beta",
            "3",
            "--train-num-conditions",
            "1",
            "--test-num-conditions",
            "3",
            "--filter-by-learned",
        ],
    },
    "ood_semantic_compositional_hard": {
        "label": "OOD Semantic Compositional (Hard)",
        "desc": "Harder semantic/compositional OOD: 1-condition train -> 4-condition test",
        "train": TRAIN_TASKS,
        "test": TEST_TASKS,
        "args": [
            "--semantic-compositional",
            "--beta",
            "3",
            "--train-num-conditions",
            "1",
            "--test-num-conditions",
            "4",
            "--filter-by-learned",
        ],
    },
}


def clean_data_directory():
    """Remove persisted state to avoid cross-run contamination."""
    data_dir = PROJECT_ROOT / "data"

    for chroma_dir in data_dir.glob("chroma_*"):
        if chroma_dir.is_dir():
            shutil.rmtree(chroma_dir)
            print(f"  cleaned: {chroma_dir.name}/")

    for json_file in data_dir.glob("precept_*.json"):
        json_file.unlink()
        print(f"  cleaned: {json_file.name}")

    for bf in [data_dir / "full_reflexion_memory.json", data_dir / "expel_insights.json"]:
        if bf.exists():
            bf.unlink()

    stale_results = list(data_dir.glob("experiment_results_*.json"))
    for f in stale_results:
        f.unlink()


def _extract_trace_path(output_lines):
    pattern = re.compile(r"Detailed execution traces saved to:\s*(/.+trace_[^\s]+\.json)")
    for line in output_lines:
        m = pattern.search(line)
        if m:
            return Path(m.group(1))
    return None


def _compute_ttfc_metrics(trace_path: Path):
    """Compute TTFC-L metrics using first learning event per key."""
    if not trace_path or not trace_path.exists():
        return None

    with open(trace_path) as f:
        data = json.load(f)

    traces = []
    for tr in data.get("training", {}).get("traces", []):
        if tr.get("agent_type") == "precept":
            traces.append(tr)
    for tr in data.get("testing", {}).get("precept", {}).get("traces", []):
        traces.append(tr)
    traces.sort(key=lambda t: t.get("start_time", 0))

    rows = []
    for tr in traces:
        task_complete = None
        learned_evt = None
        for e in tr.get("events", []):
            if e.get("event_type") == "task_complete":
                task_complete = e
            elif e.get("event_type") == "rule_learned":
                learned_evt = e
        details = (task_complete or {}).get("details", {})
        condition_key = details.get("condition_key") or (
            (learned_evt or {}).get("details", {}).get("condition_key")
        )
        rows.append(
            {
                "success": bool((tr.get("result") or {}).get("success", False)),
                "condition_key": condition_key,
                "rule_learned": learned_evt is not None,
                "learned_key": (learned_evt or {}).get("details", {}).get("condition_key"),
            }
        )

    seen = set()
    global_gaps = []
    samekey_gaps = []
    censored = 0

    for i, row in enumerate(rows):
        if not row["rule_learned"]:
            continue
        key = row["learned_key"] or row["condition_key"]
        if not key or key in seen:
            continue
        seen.add(key)

        samekey_seen = 0
        found = False
        for j in range(i + 1, len(rows)):
            nxt = rows[j]
            if nxt["condition_key"] != key:
                continue
            samekey_seen += 1
            if nxt["success"]:
                global_gaps.append(j - i)
                samekey_gaps.append(samekey_seen)
                found = True
                break
        if not found:
            censored += 1

    return {
        "learned_keys_total": len(seen),
        "resolved_keys": len(global_gaps),
        "censored_keys": censored,
        "mean_global_gap": (sum(global_gaps) / len(global_gaps)) if global_gaps else None,
        "mean_samekey_gap": (sum(samekey_gaps) / len(samekey_gaps)) if samekey_gaps else None,
    }


def run_single(regime_name: str, config_name: str, config: dict, seed: int, output_dir: Path):
    clean_data_directory()

    regime = REGIMES[regime_name]
    regime_args = regime["args"]
    train_tasks = int(regime.get("train", TRAIN_TASKS))
    test_tasks = int(regime.get("test", TEST_TASKS))
    cmd = [
        "uv",
        "run",
        "examples/precept_autogen_mcp_full.py",
        "--domain",
        DOMAIN,
        "--train",
        str(train_tasks),
        "--test",
        str(test_tasks),
        "--max-retries",
        str(MAX_RETRIES),
        "--no-static-knowledge",
        "--hybrid-retrieval",
        "--seed",
        str(seed),
        "-ct",
        "-tw",
        "4",
        "-c",
        "-w",
        "4",
        "--detailed-logs",
        "-v",
        *regime_args,
    ]

    print(f"    run [{regime_name}][{config_name}] seed={seed}")

    log_file = output_dir / f"{regime_name}__{config_name}_seed{seed}.log"
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    for k, v in config["env"].items():
        env[k] = v

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
        process.wait(timeout=2400)
    except subprocess.TimeoutExpired:
        process.kill()
        return None

    with open(log_file, "w") as f:
        f.write(f"Command: {' '.join(cmd)}\n")
        f.write(f"Regime: {regime_name}\n")
        f.write(f"Config: {config_name} ({config['desc']})\n")
        f.write(f"Env overrides: {config['env']}\n")
        f.write(f"Exit code: {process.returncode}\n")
        f.write(f"COMPASS events: {json.dumps(compass_events)}\n")
        f.write("=" * 80 + "\n")
        f.writelines(output_lines)

    if process.returncode != 0:
        return None

    data_dir = PROJECT_ROOT / "data"
    result_files = list(data_dir.glob(f"experiment_results_{DOMAIN}_*.json"))
    if not result_files:
        return None
    latest = max(result_files, key=lambda p: p.stat().st_mtime)
    copied = output_dir / f"{regime_name}__{config_name}_seed{seed}_results.json"
    shutil.copy(latest, copied)

    with open(latest) as f:
        result = json.load(f)
    result["compass_events"] = compass_events
    result["config_name"] = config_name
    result["regime_name"] = regime_name
    result["trace_path"] = None

    trace_path = _extract_trace_path(output_lines)
    if trace_path is not None:
        result["trace_path"] = str(trace_path)
        result["ttfc"] = _compute_ttfc_metrics(trace_path)
    else:
        result["ttfc"] = None

    return result


def _mean(vals):
    return sum(vals) / len(vals) if vals else None


def aggregate(all_results: dict, output_dir: Path):
    import numpy as np
    from scipy import stats

    aggregated = {}
    comparisons = {}

    for regime_name in REGIMES:
        aggregated[regime_name] = {}
        regime_runs = all_results.get(regime_name, {})
        for cfg_name in CONFIGS:
            runs = [r for r in regime_runs.get(cfg_name, []) if r is not None]
            p1_vals, pt_vals, steps_vals, ttfc_global, ttfc_same = [], [], [], [], []
            for r in runs:
                pre = r.get("agents", {}).get("precept", {})
                if not pre:
                    continue
                p1_vals.append(float(pre.get("first_try_success_rate", 0)))
                pt_vals.append(float(pre.get("success_rate", 0)))
                steps_vals.append(float(pre.get("avg_steps", 0)))
                ttfc = r.get("ttfc") or {}
                if ttfc.get("mean_global_gap") is not None:
                    ttfc_global.append(float(ttfc["mean_global_gap"]))
                if ttfc.get("mean_samekey_gap") is not None:
                    ttfc_same.append(float(ttfc["mean_samekey_gap"]))

            n = len(p1_vals)
            if n == 0:
                continue
            t_crit = stats.t.ppf(0.975, df=max(n - 1, 1))
            p1_arr = np.array(p1_vals)
            pt_arr = np.array(pt_vals)
            steps_arr = np.array(steps_vals)
            aggregated[regime_name][cfg_name] = {
                "label": CONFIGS[cfg_name]["label"],
                "n": n,
                "p1": {
                    "mean": float(np.mean(p1_arr)),
                    "std": float(np.std(p1_arr, ddof=1)) if n > 1 else 0.0,
                    "ci_95": float(t_crit * np.std(p1_arr, ddof=1) / np.sqrt(n)) if n > 1 else 0.0,
                },
                "pt": {
                    "mean": float(np.mean(pt_arr)),
                    "std": float(np.std(pt_arr, ddof=1)) if n > 1 else 0.0,
                    "ci_95": float(t_crit * np.std(pt_arr, ddof=1) / np.sqrt(n)) if n > 1 else 0.0,
                },
                "steps": {
                    "mean": float(np.mean(steps_arr)),
                    "std": float(np.std(steps_arr, ddof=1)) if n > 1 else 0.0,
                    "ci_95": float(t_crit * np.std(steps_arr, ddof=1) / np.sqrt(n)) if n > 1 else 0.0,
                },
                "ttfc_mean_global_gap": _mean(ttfc_global),
                "ttfc_mean_samekey_gap": _mean(ttfc_same),
            }

        # Full vs no_compass per regime on p1
        full_runs = [r for r in regime_runs.get("full", []) if r is not None]
        nc_runs = [r for r in regime_runs.get("no_compass", []) if r is not None]
        full_p1 = [r.get("agents", {}).get("precept", {}).get("first_try_success_rate", 0) for r in full_runs]
        nc_p1 = [r.get("agents", {}).get("precept", {}).get("first_try_success_rate", 0) for r in nc_runs]
        n_min = min(len(full_p1), len(nc_p1))
        if n_min >= 2:
            t_stat, p_val = stats.ttest_rel(full_p1[:n_min], nc_p1[:n_min])
            comparisons[f"{regime_name}__full_vs_no_compass"] = {
                "delta_pp": float((np.mean(full_p1[:n_min]) - np.mean(nc_p1[:n_min])) * 100),
                "t_stat": float(t_stat),
                "p_value": float(p_val),
                "n": n_min,
            }

    payload = {
        "experiment": "exp9_compass_stress",
        "domain": DOMAIN,
        "timestamp": datetime.now().isoformat(),
        "config": {"E": E, "N": NUM_CONDITIONS, "beta": BETA, "max_retries": MAX_RETRIES},
        "regimes": REGIMES,
        "aggregated": aggregated,
        "comparisons": comparisons,
    }
    out = output_dir / "aggregated_results.json"
    with open(out, "w") as f:
        json.dump(payload, f, indent=2)

    report = output_dir / "experiment_report.md"
    with open(report, "w") as f:
        f.write("# Experiment 9: COMPASS Stress Test\n\n")
        f.write(f"Domain: `{DOMAIN}` | E={E} | N={NUM_CONDITIONS} | beta={BETA}\n\n")
        for regime_name, regime_data in aggregated.items():
            f.write(f"## Regime: {regime_name}\n\n")
            f.write("| Config | P1 | Pt | Steps | TTFC global gap | TTFC same-key gap |\n")
            f.write("|---|---:|---:|---:|---:|---:|\n")
            for cfg in ["full", "no_compass", "no_rules", "base_only"]:
                if cfg not in regime_data:
                    continue
                a = regime_data[cfg]
                p1 = a["p1"]["mean"] * 100
                pt = a["pt"]["mean"] * 100
                st = a["steps"]["mean"]
                tg = a["ttfc_mean_global_gap"]
                ts = a["ttfc_mean_samekey_gap"]
                tg_str = f"{tg:.2f}" if tg is not None else "NA"
                ts_str = f"{ts:.2f}" if ts is not None else "NA"
                f.write(f"| {a['label']} | {p1:.1f}% | {pt:.1f}% | {st:.2f} | {tg_str} | {ts_str} |\n")
            f.write("\n")

    return payload


def main():
    parser = argparse.ArgumentParser(description="Experiment 9: COMPASS stress-test")
    parser.add_argument("--quick", action="store_true", help="Run 3 seeds")
    parser.add_argument("--very-quick", action="store_true", help="Run 1 seed")
    args = parser.parse_args()

    if args.very_quick:
        seeds = VERY_QUICK_SEEDS
    elif args.quick:
        seeds = QUICK_SEEDS
    else:
        seeds = PUBLICATION_SEEDS

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = PROJECT_ROOT / "data" / "publication_results" / f"exp9_compass_stress_{ts}"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("EXPERIMENT 9: COMPASS STRESS TEST")
    print("=" * 80)
    print(f"Domain: {DOMAIN} | Seeds: {len(seeds)} {seeds}")
    print(f"Regimes: {list(REGIMES.keys())}")
    print(f"Configs: {list(CONFIGS.keys())}")
    print(f"Default train/test per regime: {TRAIN_TASKS}/{TEST_TASKS}")
    print(f"Total runs: {len(seeds) * len(REGIMES) * len(CONFIGS)}")
    print(f"Output: {output_dir}")
    print("=" * 80)

    all_results = {reg: {cfg: [] for cfg in CONFIGS} for reg in REGIMES}

    for regime_name in REGIMES:
        print(f"\n{'-' * 72}\nRegime: {regime_name} - {REGIMES[regime_name]['desc']}\n{'-' * 72}")
        for cfg_name, cfg in CONFIGS.items():
            print(f"\n  Config: {cfg['label']}")
            for seed in tqdm(seeds, desc=f"  {regime_name}:{cfg_name}", total=len(seeds)):
                res = run_single(regime_name, cfg_name, cfg, seed, output_dir)
                all_results[regime_name][cfg_name].append(res)

    aggregate(all_results, output_dir)
    print(f"\nResults saved to: {output_dir}")


if __name__ == "__main__":
    main()

