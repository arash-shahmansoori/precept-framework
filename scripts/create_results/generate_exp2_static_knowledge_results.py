#!/usr/bin/env python3
"""
Generate Publication Results for Experiment 2: Static Knowledge Ablation

Creates:
- Table 2: Impact of static knowledge on all agents
- Figure 2: Grouped bar chart comparing with/without static knowledge

Usage:
    python scripts/create_results/generate_exp2_static_knowledge_results.py <results_directory>
"""

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["DejaVu Serif", "Times New Roman", "Computer Modern Roman"],
        "font.size": 12,
        "axes.labelsize": 13,
        "axes.titlesize": 14,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "legend.fontsize": 11,
        "figure.dpi": 600,
        "savefig.dpi": 600,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.15,
        "axes.linewidth": 1.0,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "grid.alpha": 0.4,
        "errorbar.capsize": 4,
        "mathtext.fontset": "dejavuserif",
    }
)

COLORS = {
    "precept": "#2563eb",
    "full_reflexion": "#dc2626",
    "expel": "#ea580c",
    "positive": "#1B7A3D",
    "negative": "#B71C1C",
}
HATCHES = {"precept": None, "full_reflexion": "//", "expel": "\\\\"}


def _bounded_pct_ci(mean_pct: float, ci_pct: float, lower: float = 0.0, upper: float = 100.0) -> float:
    """Bound symmetric CI so mean±CI stays within [lower, upper]."""
    return max(0.0, min(ci_pct, mean_pct - lower, upper - mean_pct))


def _bounded_pct_from_frac(metric: dict) -> tuple[float, float]:
    """Convert fractional metric dict to bounded percentage mean/CI."""
    mean_pct = metric.get("mean", 0) * 100
    ci_pct = metric.get("ci_95", 0) * 100
    return mean_pct, _bounded_pct_ci(mean_pct, ci_pct)


def _cap_pct_errorbars(means_pct: list[float], cis_pct: list[float]) -> list[list[float]]:
    """Create asymmetric error bars bounded to [0, 100]."""
    lower = [max(0.0, min(ci, mean)) for mean, ci in zip(means_pct, cis_pct)]
    upper = [max(0.0, min(ci, 100.0 - mean)) for mean, ci in zip(means_pct, cis_pct)]
    return [lower, upper]


def load_results(results_dir: Path) -> dict:
    with open(results_dir / "aggregated_results.json") as f:
        return json.load(f)


def generate_figure2_static_knowledge(data: dict, output_dir: Path):
    """Generate Figure 2: Static knowledge ablation chart (publication style)."""

    configs = ["with_static_knowledge", "without_static_knowledge"]
    labels = ["With Adv. SK", "Without SK"]
    agents = [
        ("precept", "PRECEPT", COLORS["precept"], HATCHES["precept"]),
        ("full_reflexion", "Full Reflexion", COLORS["full_reflexion"], HATCHES["full_reflexion"]),
        ("expel", "ExpeL", COLORS["expel"], HATCHES["expel"]),
    ]

    has_expel = any(
        config in data
        and "expel" in data[config]
        and data[config]["expel"]["first_try_success"]["mean"] > 0
        for config in configs
    )
    if not has_expel:
        agents = agents[:2]

    n_agents = len(agents)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5),
                                    gridspec_kw={"width_ratios": [3, 2]})

    # === Panel (a): First-Try Success P₁ grouped bars ===
    x = np.arange(len(configs))
    width = 0.24
    offsets = [-(n_agents - 1) * width / 2 + i * width for i in range(n_agents)]

    all_vals = {}
    all_cis = {}
    for key, label, color, hatch in agents:
        vals = []
        cis = []
        for config in configs:
            if config in data:
                d = data[config]
                agent_data = d.get(key, {}).get("first_try_success", {"mean": 0, "ci_95": 0})
                vals.append(agent_data["mean"] * 100)
                cis.append(agent_data["ci_95"] * 100)
            else:
                vals.append(0)
                cis.append(0)
        all_vals[key] = vals
        all_cis[key] = cis

    bar_groups = {}
    for idx, (key, label, color, hatch) in enumerate(agents):
        ci_bounded = _cap_pct_errorbars(all_vals[key], all_cis[key])
        bar_groups[key] = ax1.bar(
            x + offsets[idx], all_vals[key], width,
            yerr=ci_bounded,
            label=label, color=color, hatch=hatch,
            capsize=4, edgecolor="black", linewidth=0.8, zorder=3,
            error_kw={"linewidth": 1.2, "capthick": 1.2},
        )

    for idx, (key, label, color, hatch) in enumerate(agents):
        for i, (bar, val) in enumerate(zip(bar_groups[key], all_vals[key])):
            bounded_ci = _bounded_pct_ci(val, all_cis[key][i])
            ax1.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + bounded_ci + 1.5,
                f"{val:.1f}%",
                ha="center", va="bottom", fontsize=8.5,
                fontweight="bold" if key == "precept" else "normal",
                color=color,
            )

    sk_effect = data.get("static_knowledge_effect", {})
    precept_p1_gain = sk_effect.get("precept_p1_gain", 0)
    effect_color = COLORS["positive"] if precept_p1_gain >= 0 else COLORS["negative"]
    max_bar = max(
        v + _bounded_pct_ci(v, c)
        for vals, cis in zip(all_vals.values(), all_cis.values())
        for v, c in zip(vals, cis)
    )
    ax1.annotate(
        f"SK Effect: {precept_p1_gain:+.1f}pp",
        xy=(0.5, min(max_bar + 10, 112)), xycoords=("data", "data"),
        ha="center", fontsize=10, fontweight="bold", color=effect_color,
    )

    ax1.set_ylabel(r"First-Try Success Rate $P_1$ (%)", fontweight="bold")
    ax1.set_title(r"(a) First-Try Success $P_1$", fontweight="bold", pad=12)
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels)
    ax1.set_xlim(-0.55, 1.55)
    ax1.set_ylim(0, 120)
    ax1.axhline(y=100, color="#999999", linestyle="--", linewidth=0.6, alpha=0.5)
    ax1.legend(loc="upper right", frameon=True, fancybox=False, edgecolor="#cccccc")
    ax1.grid(axis="y", linestyle=":", alpha=0.4)

    # === Panel (b): PRECEPT Advantage (pp) ===
    advantages = {}
    for config_idx, config in enumerate(configs):
        if config not in data:
            continue
        adv = data[config]["advantage"]
        for key, label, color, hatch in agents[1:]:
            adv_key = f"vs_{key}_first_try_pp" if key != "full_reflexion" else "vs_fr_first_try_pp"
            val = adv.get(adv_key, adv.get("first_try_success_pp", 0))
            if key not in advantages:
                advantages[key] = []
            advantages[key].append(val)

    baseline_keys = [a for a in agents[1:]]
    n_baselines = len(baseline_keys)
    adv_width = 0.24
    adv_offsets = [-(n_baselines - 1) * adv_width / 2 + i * adv_width for i in range(n_baselines)]

    for idx, (key, label, color, hatch) in enumerate(baseline_keys):
        vals = advantages.get(key, [0, 0])
        bars = ax2.bar(
            x + adv_offsets[idx], vals, adv_width,
            color=color, hatch=hatch, edgecolor="black", linewidth=0.8,
            label=f"vs {label}", zorder=3, alpha=0.85,
        )
        for bar, val in zip(bars, vals):
            ax2.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.8,
                f"+{val:.1f}pp",
                ha="center", va="bottom", fontsize=9, fontweight="bold",
                color=color,
            )

    ax2.set_ylabel(r"PRECEPT $\Delta P_1$ Advantage (pp)", fontweight="bold")
    ax2.set_title("(b) PRECEPT Advantage over Baselines", fontweight="bold", pad=12)
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels)
    ax2.set_xlim(-0.55, 1.55)
    ax2.axhline(y=0, color="#999999", linestyle="-", linewidth=0.8, alpha=0.5)
    ax2.legend(loc="upper right", frameon=True, fancybox=False, edgecolor="#cccccc")
    ax2.grid(axis="y", linestyle=":", alpha=0.4)

    plt.tight_layout(w_pad=3)
    plt.savefig(output_dir / "fig2_static_knowledge.png", dpi=600, bbox_inches="tight")
    plt.savefig(output_dir / "fig2_static_knowledge.pdf", bbox_inches="tight")
    plt.close()

    print("  ✅ Generated: fig2_static_knowledge.png/pdf")


def generate_table2_static_knowledge(data: dict, output_dir: Path):
    """Generate Table 2: Static knowledge ablation table."""

    # Check if ExpeL data exists
    has_expel = any(
        "expel" in data.get(config, {})
        and data[config]["expel"]["first_try_success"]["mean"] > 0
        for config in ["with_static_knowledge", "without_static_knowledge"]
        if config in data
    )

    # LaTeX table
    if has_expel:
        latex_lines = [
            r"\begin{table}[t]",
            r"\centering",
            r"\caption{Impact of Static Knowledge on PRECEPT, Full Reflexion, and ExpeL}",
            r"\label{tab:static_knowledge}",
            r"\begin{tabular}{lcccccc}",
            r"\toprule",
            r"Configuration & PRECEPT P₁ & FR P₁ & ExpeL P₁ & PRECEPT Pₜ & FR Pₜ & ExpeL Pₜ \\",
            r"\midrule",
        ]
    else:
        latex_lines = [
            r"\begin{table}[t]",
            r"\centering",
            r"\caption{Impact of Static Knowledge on PRECEPT and Full Reflexion}",
            r"\label{tab:static_knowledge}",
            r"\begin{tabular}{lcccc}",
            r"\toprule",
            r"Configuration & PRECEPT P₁ & FR P₁ & PRECEPT Pₜ & FR Pₜ \\",
            r"\midrule",
        ]

    for config, label in [
        ("with_static_knowledge", "With Static KB"),
        ("without_static_knowledge", "Without Static KB"),
    ]:
        if config not in data:
            continue
        d = data[config]
        p_p1 = d["precept"]["first_try_success"]
        fr_p1 = d["full_reflexion"]["first_try_success"]
        p_pt = d["precept"]["success_rate"]
        fr_pt = d["full_reflexion"]["success_rate"]
        p_p1_mean, p_p1_ci = _bounded_pct_from_frac(p_p1)
        fr_p1_mean, fr_p1_ci = _bounded_pct_from_frac(fr_p1)
        p_pt_mean, p_pt_ci = _bounded_pct_from_frac(p_pt)
        fr_pt_mean, fr_pt_ci = _bounded_pct_from_frac(fr_pt)

        if has_expel:
            e_p1 = d.get("expel", {}).get("first_try_success", {"mean": 0, "ci_95": 0})
            e_pt = d.get("expel", {}).get("success_rate", {"mean": 0, "ci_95": 0})
            e_p1_mean, e_p1_ci = _bounded_pct_from_frac(e_p1)
            e_pt_mean, e_pt_ci = _bounded_pct_from_frac(e_pt)
            latex_lines.append(
                f"{label} & "
                f"\\textbf{{{p_p1_mean:.1f}\\%}} $\\pm$ {p_p1_ci:.1f} & "
                f"{fr_p1_mean:.1f}\\% $\\pm$ {fr_p1_ci:.1f} & "
                f"{e_p1_mean:.1f}\\% $\\pm$ {e_p1_ci:.1f} & "
                f"{p_pt_mean:.1f}\\% $\\pm$ {p_pt_ci:.1f} & "
                f"{fr_pt_mean:.1f}\\% $\\pm$ {fr_pt_ci:.1f} & "
                f"{e_pt_mean:.1f}\\% $\\pm$ {e_pt_ci:.1f} \\\\"
            )
        else:
            latex_lines.append(
                f"{label} & "
                f"\\textbf{{{p_p1_mean:.1f}\\%}} $\\pm$ {p_p1_ci:.1f} & "
                f"{fr_p1_mean:.1f}\\% $\\pm$ {fr_p1_ci:.1f} & "
                f"{p_pt_mean:.1f}\\% $\\pm$ {p_pt_ci:.1f} & "
                f"{fr_pt_mean:.1f}\\% $\\pm$ {fr_pt_ci:.1f} \\\\"
            )

    # Add effect row
    if "static_knowledge_effect" in data:
        effect = data["static_knowledge_effect"]
        latex_lines.append(r"\midrule")
        if has_expel:
            latex_lines.append(
                f"SK Effect & +{effect['precept_p1_gain']:.1f} pp & +{effect['fr_p1_gain']:.1f} pp & "
                f"+{effect.get('expel_p1_gain', 0):.1f} pp & "
                f"+{effect['precept_pt_gain']:.1f} pp & +{effect['fr_pt_gain']:.1f} pp & "
                f"+{effect.get('expel_pt_gain', 0):.1f} pp \\\\"
            )
        else:
            latex_lines.append(
                f"SK Effect & +{effect['precept_p1_gain']:.1f} pp & +{effect['fr_p1_gain']:.1f} pp & "
                f"+{effect['precept_pt_gain']:.1f} pp & +{effect['fr_pt_gain']:.1f} pp \\\\"
            )

    latex_lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table}",
        ]
    )

    with open(output_dir / "table2_static_knowledge.tex", "w") as f:
        f.write("\n".join(latex_lines))

    # Markdown table
    md_lines = [
        "## Table 2: Static Knowledge Ablation Results\n",
        "| Configuration | PRECEPT P₁ | FR P₁ | PRECEPT Pₜ | FR Pₜ |",
        "|---------------|-----------|-------|-----------|-------|",
    ]

    for config, label in [
        ("with_static_knowledge", "With Static KB"),
        ("without_static_knowledge", "Without Static KB"),
    ]:
        if config not in data:
            continue
        d = data[config]
        p_p1 = d["precept"]["first_try_success"]
        fr_p1 = d["full_reflexion"]["first_try_success"]
        p_pt = d["precept"]["success_rate"]
        fr_pt = d["full_reflexion"]["success_rate"]
        p_p1_mean, p_p1_ci = _bounded_pct_from_frac(p_p1)
        fr_p1_mean, fr_p1_ci = _bounded_pct_from_frac(fr_p1)
        p_pt_mean, p_pt_ci = _bounded_pct_from_frac(p_pt)
        fr_pt_mean, fr_pt_ci = _bounded_pct_from_frac(fr_pt)

        md_lines.append(
            f"| {label} | "
            f"**{p_p1_mean:.1f}%** ± {p_p1_ci:.1f} | "
            f"{fr_p1_mean:.1f}% ± {fr_p1_ci:.1f} | "
            f"{p_pt_mean:.1f}% ± {p_pt_ci:.1f} | "
            f"{fr_pt_mean:.1f}% ± {fr_pt_ci:.1f} |"
        )

    with open(output_dir / "table2_static_knowledge.md", "w") as f:
        f.write("\n".join(md_lines))

    print("  ✅ Generated: table2_static_knowledge.tex/md")


def main():
    if len(sys.argv) < 2:
        print(
            "Usage: python generate_exp2_static_knowledge_results.py <results_directory>"
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
    print("GENERATING EXPERIMENT 2 RESULTS (Static Knowledge Ablation)")
    print("=" * 60 + "\n")

    data = load_results(results_dir)

    print("Generating figure...")
    generate_figure2_static_knowledge(data, figures_dir)

    print("\nGenerating table...")
    generate_table2_static_knowledge(data, tables_dir)

    print("\n" + "=" * 60)
    print("COMPLETE!")
    print("=" * 60)


if __name__ == "__main__":
    main()
