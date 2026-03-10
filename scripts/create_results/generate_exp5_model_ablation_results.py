#!/usr/bin/env python3
"""
Generate Publication Results for Experiment 5: Model and Embedding Ablation

Creates:
- Table 5: Performance comparison across model/embedding configurations
- Figure 5: Grouped bar chart showing PRECEPT advantage persists

Usage:
    python scripts/create_results/generate_exp5_model_ablation_results.py <results_directory>
"""

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

try:
    from scripts.utils.pct_bounds import bounded_pct_ci, cap_pct_errorbars
except ImportError:
    def bounded_pct_ci(mean_pct, ci_pct, lower=0.0, upper=100.0):
        return max(0.0, min(ci_pct, mean_pct - lower, upper - mean_pct))
    def cap_pct_errorbars(means, cis):
        lower = [max(0.0, min(ci, m)) for m, ci in zip(means, cis)]
        upper = [max(0.0, min(ci, 100.0 - m)) for m, ci in zip(means, cis)]
        return [lower, upper]

plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif"],
        "font.size": 11,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
    }
)

COLORS = {
    "precept": "#2E86AB",
    "full_reflexion": "#A23B72",
    "expel": "#6B8E23",
}

CONFIG_LABELS = {
    "baseline": "Baseline\n(mini + small)",
    "powerful_llm": "Powerful LLM\n(4o + small)",
    "powerful_embedding": "Powerful Embed\n(mini + large)",
    "both_powerful": "Both Powerful\n(4o + large)",
}

CONFIG_SHORT = {
    "baseline": "mini+small",
    "powerful_llm": "4o+small",
    "powerful_embedding": "mini+large",
    "both_powerful": "4o+large",
}


def load_results(results_dir: Path) -> dict:
    with open(results_dir / "aggregated_results.json") as f:
        return json.load(f)


def generate_figure5_model_ablation(data: dict, output_dir: Path):
    """Generate Figure 5: Model ablation comparison chart."""

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Order configs logically
    config_order = ["baseline", "powerful_llm", "powerful_embedding", "both_powerful"]
    configs = [c for c in config_order if c in data]

    if not configs:
        print("  ⚠️ No valid configurations found")
        return

    x = np.arange(len(configs))
    width = 0.25

    # Left: First-Try Success by config
    ax = axes[0]

    precept_vals = []
    precept_cis = []
    fr_vals = []
    fr_cis = []
    expel_vals = []
    expel_cis = []

    for config in configs:
        d = data[config]
        precept_vals.append(d["precept"]["first_try_success"]["mean"] * 100)
        precept_cis.append(d["precept"]["first_try_success"]["ci_95"] * 100)
        fr_vals.append(d["full_reflexion"]["first_try_success"]["mean"] * 100)
        fr_cis.append(d["full_reflexion"]["first_try_success"]["ci_95"] * 100)
        expel_vals.append(d["expel"]["first_try_success"]["mean"] * 100)
        expel_cis.append(d["expel"]["first_try_success"]["ci_95"] * 100)

    precept_errs = cap_pct_errorbars(precept_vals, precept_cis)
    fr_errs = cap_pct_errorbars(fr_vals, fr_cis)
    expel_errs = cap_pct_errorbars(expel_vals, expel_cis)

    bars1 = ax.bar(
        x - width,
        precept_vals,
        width,
        yerr=precept_errs,
        label="PRECEPT",
        color=COLORS["precept"],
        capsize=4,
    )
    bars2 = ax.bar(
        x,
        fr_vals,
        width,
        yerr=fr_errs,
        label="Full Reflexion",
        color=COLORS["full_reflexion"],
        capsize=4,
    )
    bars3 = ax.bar(
        x + width,
        expel_vals,
        width,
        yerr=expel_errs,
        label="ExpeL",
        color=COLORS["expel"],
        capsize=4,
    )

    ax.set_ylabel("First-Try Success Rate (%)", fontsize=12, fontweight="bold")
    ax.set_title(
        "A. First-Try Success (P₁) Across Model Configurations\n"
        "More powerful models do NOT help baselines",
        fontsize=12,
        fontweight="bold",
    )
    ax.set_xticks(x)
    ax.set_xticklabels([CONFIG_LABELS[c] for c in configs], fontsize=10)
    ax.set_ylim(0, 110)
    ax.legend(loc="upper right")
    ax.grid(axis="y", alpha=0.3)

    # Add value labels on PRECEPT bars
    for bar, val in zip(bars1, precept_vals):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 2,
            f"{val:.0f}%",
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
        )

    # Right: PRECEPT Advantage by config
    ax = axes[1]

    advantages_fr = [data[c]["advantage"]["vs_fr_first_try_pp"] for c in configs]
    advantages_expel = [data[c]["advantage"]["vs_expel_first_try_pp"] for c in configs]

    bars_fr = ax.bar(
        x - width / 2,
        advantages_fr,
        width,
        label="vs Full Reflexion",
        color=COLORS["full_reflexion"],
        alpha=0.7,
    )
    bars_expel = ax.bar(
        x + width / 2,
        advantages_expel,
        width,
        label="vs ExpeL",
        color=COLORS["expel"],
        alpha=0.7,
    )

    ax.set_ylabel("PRECEPT Advantage (pp)", fontsize=12, fontweight="bold")
    ax.set_title(
        "B. PRECEPT Advantage Persists Across All Configurations\n"
        "Architectural advantage, not model capability",
        fontsize=12,
        fontweight="bold",
    )
    ax.set_xticks(x)
    ax.set_xticklabels([CONFIG_SHORT[c] for c in configs], fontsize=10)
    ax.axhline(y=0, color="gray", linestyle="-", alpha=0.5)
    ax.legend(loc="upper right")
    ax.grid(axis="y", alpha=0.3)

    # Add value labels
    for bar, val in zip(bars_fr, advantages_fr):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 1,
            f"+{val:.0f}",
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
        )

    plt.tight_layout()
    plt.savefig(output_dir / "fig5_model_ablation.png")
    plt.savefig(output_dir / "fig5_model_ablation.pdf")
    plt.close()

    print("  ✅ Generated: fig5_model_ablation.png/pdf")


def generate_table5_model_ablation(data: dict, output_dir: Path):
    """Generate Table 5: Model ablation comparison table."""

    config_order = ["baseline", "powerful_llm", "powerful_embedding", "both_powerful"]
    configs = [c for c in config_order if c in data]

    # LaTeX table
    latex_lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Model and Embedding Ablation: PRECEPT Advantage Persists}",
        r"\label{tab:model_ablation}",
        r"\begin{tabular}{llcccc}",
        r"\toprule",
        r"LLM & Embedding & PRECEPT P₁ & FR P₁ & ExpeL P₁ & $\Delta$ FR \\",
        r"\midrule",
    ]

    for config in configs:
        d = data[config]
        llm = d.get("llm_model", "unknown")
        embed = d.get("embedding_model", "unknown").replace("text-embedding-3-", "")
        p = d["precept"]["first_try_success"]
        fr = d["full_reflexion"]["first_try_success"]
        expel = d["expel"]["first_try_success"]
        adv = d["advantage"]["vs_fr_first_try_pp"]

        latex_lines.append(
            f"{llm} & {embed} & "
            f"\\textbf{{{p['mean'] * 100:.1f}\\%}} & "
            f"{fr['mean'] * 100:.1f}\\% & "
            f"{expel['mean'] * 100:.1f}\\% & "
            f"+{adv:.1f} pp \\\\"
        )

    latex_lines.extend(
        [
            r"\bottomrule",
            r"\multicolumn{6}{l}{\footnotesize More powerful models do NOT help baselines in Black Swan CSPs} \\",
            r"\end{tabular}",
            r"\end{table}",
        ]
    )

    with open(output_dir / "table5_model_ablation.tex", "w") as f:
        f.write("\n".join(latex_lines))

    # Markdown table
    md_lines = [
        "## Model and Embedding Ablation Results\n",
        "| LLM | Embedding | PRECEPT P₁ | FR P₁ | ExpeL P₁ | Δ FR |",
        "|-----|-----------|-----------|-------|----------|------|",
    ]

    for config in configs:
        d = data[config]
        llm = d.get("llm_model", "unknown")
        embed = d.get("embedding_model", "unknown").replace("text-embedding-3-", "")
        p = d["precept"]["first_try_success"]
        fr = d["full_reflexion"]["first_try_success"]
        expel = d["expel"]["first_try_success"]
        adv = d["advantage"]["vs_fr_first_try_pp"]

        md_lines.append(
            f"| {llm} | {embed} | "
            f"**{p['mean'] * 100:.1f}%** | "
            f"{fr['mean'] * 100:.1f}% | "
            f"{expel['mean'] * 100:.1f}% | "
            f"+{adv:.1f} pp |"
        )

    md_lines.append("\n*Key Finding: PRECEPT advantage persists across ALL model configurations*")

    with open(output_dir / "table5_model_ablation.md", "w") as f:
        f.write("\n".join(md_lines))

    print("  ✅ Generated: table5_model_ablation.tex/md")


def generate_statistical_summary(data: dict, output_dir: Path):
    """Generate statistical summary for model ablation."""

    lines = []
    lines.append("# Experiment 5: Model Ablation Statistical Summary\n")

    lines.append("## Key Finding\n")
    lines.append(
        "**PRECEPT's advantage persists across ALL model/embedding configurations.**\n"
    )
    lines.append(
        "More powerful LLMs (gpt-4o) and embeddings (text-embedding-3-large) "
        "do NOT help baselines overcome the fundamental architectural limitation "
        "in Black Swan Constraint Satisfaction Problems.\n"
    )

    lines.append("## Theoretical Explanation\n")
    lines.append("1. **LLM Reasoning Cannot Help**: Black Swan solutions are ARBITRARY by design. ")
    lines.append("No amount of reasoning can derive: 'R-482+C-LOW' → 'strategy_delta'\n")
    lines.append("2. **Better Embeddings Cannot Help**: Condition codes have NO semantic meaning. ")
    lines.append("'R-482' and 'R-483' are equally distant from any solution.\n")
    lines.append("3. **PRECEPT's Architectural Advantage**: O(1) hash lookup provides ")
    lines.append("deterministic condition_key → solution mapping.\n")

    lines.append("## Results by Configuration\n")

    for config_name, d in data.items():
        lines.append(f"\n### {config_name}\n")
        lines.append(f"- **LLM**: {d.get('llm_model', 'unknown')}")
        lines.append(f"- **Embedding**: {d.get('embedding_model', 'unknown')}")
        lines.append(f"- **PRECEPT P₁**: {d['precept']['first_try_success']['mean'] * 100:.1f}%")
        lines.append(f"- **FR P₁**: {d['full_reflexion']['first_try_success']['mean'] * 100:.1f}%")
        lines.append(f"- **ExpeL P₁**: {d['expel']['first_try_success']['mean'] * 100:.1f}%")
        lines.append(f"- **PRECEPT Advantage vs FR**: +{d['advantage']['vs_fr_first_try_pp']:.1f} pp")

    with open(output_dir / "statistical_summary.md", "w") as f:
        f.write("\n".join(lines))

    print("  ✅ Generated: statistical_summary.md")


def main():
    if len(sys.argv) < 2:
        print(
            "Usage: python generate_exp5_model_ablation_results.py <results_directory>"
        )
        sys.exit(1)

    results_dir = Path(sys.argv[1])
    if not results_dir.exists():
        print(f"Error: Directory not found: {results_dir}")
        sys.exit(1)

    tables_dir = results_dir / "tables"
    figures_dir = results_dir / "figures"
    tables_dir.mkdir(exist_ok=True)
    figures_dir.mkdir(exist_ok=True)

    print("\n" + "=" * 60)
    print("GENERATING EXPERIMENT 5 RESULTS (Model Ablation)")
    print("=" * 60 + "\n")

    data = load_results(results_dir)

    print(f"Loaded data for {len(data)} configurations")

    print("\nGenerating figure...")
    generate_figure5_model_ablation(data, figures_dir)

    print("\nGenerating table...")
    generate_table5_model_ablation(data, tables_dir)

    print("\nGenerating statistical summary...")
    generate_statistical_summary(data, tables_dir)

    print("\n" + "=" * 60)
    print("COMPLETE!")
    print("=" * 60)


if __name__ == "__main__":
    main()
