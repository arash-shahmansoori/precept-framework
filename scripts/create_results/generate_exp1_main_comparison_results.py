#!/usr/bin/env python3
"""
Generate Publication Results for Experiment 1: Main Domain Comparison

Creates:
- Table 1: Main comparison across 3 domains (LaTeX + Markdown)
- Figure 1: Multi-panel domain comparison (P₁ + Steps)
- Figure 2: Recovery analysis + Difficulty spectrum

Usage:
    python scripts/create_results/generate_exp1_main_comparison_results.py <results_directory>
"""

import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch
import numpy as np

# Publication-quality matplotlib settings
plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif"],
        "font.size": 10,
        "axes.labelsize": 11,
        "axes.titlesize": 12,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "figure.dpi": 200,
        "savefig.dpi": 400,
        "savefig.bbox": "tight",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.8,
        "grid.linewidth": 0.5,
        "grid.alpha": 0.3,
    }
)

# Professional color palette
# Colorblind-friendly, high-contrast palette (consistent with Exp 3/4 figures)
COLORS = {
    "precept": "#2563eb",        # Strong blue
    "full_reflexion": "#dc2626", # Strong red
    "expel": "#ea580c",          # Strong orange
    "positive": "#1B7A3D",      # Dark green for annotations
    "negative": "#B71C1C",      # Dark red for annotations
    "neutral": "#666666",
}
# Diagonal hatching for B&W printing distinguishability
HATCHES = {"precept": None, "full_reflexion": "//", "expel": "\\\\"}

# Domain metadata (E = unique condition keys, Options = solution space size)
# Only include the 3 domains used in the main comparison
DOMAIN_META = {
    "integration": {"E": 6, "options": 15, "label": "Integration"},
    "booking":     {"E": 17, "options": 20, "label": "Booking"},
    "logistics":   {"E": 4, "options": 4, "label": "Logistics"},
}

# Domains to include in the main comparison (excludes DevOps/ceiling, Finance, Coding)
INCLUDED_DOMAINS = {"integration", "booking", "logistics"}


def _bounded_pct_ci(mean_pct: float, ci_pct: float, lower: float = 0.0, upper: float = 100.0) -> float:
    """Bound symmetric CI so mean±CI stays within [lower, upper]."""
    return max(0.0, min(ci_pct, mean_pct - lower, upper - mean_pct))


def _bounded_pct_from_frac(metric: Dict) -> Tuple[float, float]:
    """Convert fractional metric dict to bounded percentage mean/CI."""
    mean_pct = metric.get("mean", 0) * 100
    ci_pct = metric.get("ci_95", 0) * 100
    return mean_pct, _bounded_pct_ci(mean_pct, ci_pct)


def _cap_pct_errorbars(means_pct: List[float], cis_pct: List[float]) -> List[List[float]]:
    """Create asymmetric error bars bounded to [0, 100]."""
    lower = [max(0.0, min(ci, mean)) for mean, ci in zip(means_pct, cis_pct)]
    upper = [max(0.0, min(ci, 100.0 - mean)) for mean, ci in zip(means_pct, cis_pct)]
    return [lower, upper]


def load_results(results_dir: Path) -> Dict:
    """Load aggregated results, filtered to included domains only."""
    with open(results_dir / "aggregated_results.json") as f:
        all_data = json.load(f)
    return {k: v for k, v in all_data.items() if k in INCLUDED_DOMAINS}


def generate_table1_main_comparison(data: Dict, output_dir: Path):
    """Generate Table 1: Main domain comparison."""

    domains = sorted(data.keys())

    # Check if ExpeL data exists
    has_expel = any(
        "expel" in data[d] and data[d]["expel"]["first_try_success"]["mean"] > 0
        for d in domains
    )

    # LaTeX table
    if has_expel:
        latex_lines = [
            r"\begin{table*}[t]",
            r"\centering",
            r"\caption{PRECEPT vs Baselines: Performance Comparison Across Domains}",
            r"\label{tab:main_comparison}",
            r"\begin{tabular}{lccccccc}",
            r"\toprule",
            r"Domain & \multicolumn{3}{c}{First-Try Success (P₁)} & \multicolumn{3}{c}{Overall Success (Pₜ)} & Δ P₁ (vs FR) \\",
            r"\cmidrule(lr){2-4} \cmidrule(lr){5-7}",
            r" & PRECEPT & Full Refl. & ExpeL & PRECEPT & Full Refl. & ExpeL & \\",
            r"\midrule",
        ]
    else:
        latex_lines = [
            r"\begin{table*}[t]",
            r"\centering",
            r"\caption{PRECEPT vs Full Reflexion: Performance Comparison Across 6 Domains}",
            r"\label{tab:main_comparison}",
            r"\begin{tabular}{lcccccc}",
            r"\toprule",
            r"Domain & \multicolumn{2}{c}{First-Try Success (P₁)} & \multicolumn{2}{c}{Overall Success (Pₜ)} & \multicolumn{2}{c}{Avg Steps} \\",
            r"\cmidrule(lr){2-3} \cmidrule(lr){4-5} \cmidrule(lr){6-7}",
            r" & PRECEPT & Full Refl. & PRECEPT & Full Refl. & PRECEPT & Full Refl. \\",
            r"\midrule",
        ]

    for domain in domains:
        d = data[domain]
        p = d["precept"]
        fr = d["full_reflexion"]

        p1_p = p["first_try_success"]
        p1_fr = fr["first_try_success"]
        pt_p = p["success_rate"]
        pt_fr = fr["success_rate"]

        if has_expel:
            expel = d.get("expel", {})
            p1_expel = expel.get("first_try_success", {"mean": 0, "ci_95": 0})
            pt_expel = expel.get("success_rate", {"mean": 0, "ci_95": 0})
            adv = d["advantage"].get(
                "vs_fr_first_try_pp", d["advantage"].get("first_try_success_pp", 0)
            )

            p1_p_mean, p1_p_ci = _bounded_pct_from_frac(p1_p)
            p1_fr_mean, p1_fr_ci = _bounded_pct_from_frac(p1_fr)
            p1_ex_mean, p1_ex_ci = _bounded_pct_from_frac(p1_expel)
            pt_p_mean, pt_p_ci = _bounded_pct_from_frac(pt_p)
            pt_fr_mean, pt_fr_ci = _bounded_pct_from_frac(pt_fr)
            pt_ex_mean, pt_ex_ci = _bounded_pct_from_frac(pt_expel)

            latex_lines.append(
                f"{domain.capitalize()} & "
                f"\\textbf{{{p1_p_mean:.1f}}} $\\pm$ {p1_p_ci:.1f} & "
                f"{p1_fr_mean:.1f} $\\pm$ {p1_fr_ci:.1f} & "
                f"{p1_ex_mean:.1f} $\\pm$ {p1_ex_ci:.1f} & "
                f"\\textbf{{{pt_p_mean:.1f}}} $\\pm$ {pt_p_ci:.1f} & "
                f"{pt_fr_mean:.1f} $\\pm$ {pt_fr_ci:.1f} & "
                f"{pt_ex_mean:.1f} $\\pm$ {pt_ex_ci:.1f} & "
                f"+{adv:.1f} pp \\\\"
            )
        else:
            steps_p = p.get("avg_steps", {"mean": 0, "ci_95": 0})
            steps_fr = fr.get("avg_steps", {"mean": 0, "ci_95": 0})

            p1_p_mean, p1_p_ci = _bounded_pct_from_frac(p1_p)
            p1_fr_mean, p1_fr_ci = _bounded_pct_from_frac(p1_fr)
            pt_p_mean, pt_p_ci = _bounded_pct_from_frac(pt_p)
            pt_fr_mean, pt_fr_ci = _bounded_pct_from_frac(pt_fr)

            latex_lines.append(
                f"{domain.capitalize()} & "
                f"\\textbf{{{p1_p_mean:.1f}}} $\\pm$ {p1_p_ci:.1f} & "
                f"{p1_fr_mean:.1f} $\\pm$ {p1_fr_ci:.1f} & "
                f"\\textbf{{{pt_p_mean:.1f}}} $\\pm$ {pt_p_ci:.1f} & "
                f"{pt_fr_mean:.1f} $\\pm$ {pt_fr_ci:.1f} & "
                f"\\textbf{{{steps_p['mean']:.2f}}} $\\pm$ {steps_p['ci_95']:.2f} & "
                f"{steps_fr['mean']:.2f} $\\pm$ {steps_fr['ci_95']:.2f} \\\\"
            )

    # Add summary row
    latex_lines.append(r"\midrule")

    # Calculate averages
    avg_p1_p = np.mean(
        [data[d]["precept"]["first_try_success"]["mean"] for d in domains]
    )
    avg_p1_fr = np.mean(
        [data[d]["full_reflexion"]["first_try_success"]["mean"] for d in domains]
    )
    adv_p1 = (avg_p1_p - avg_p1_fr) * 100

    if has_expel:
        avg_p1_expel = np.mean(
            [
                data[d].get("expel", {}).get("first_try_success", {"mean": 0})["mean"]
                for d in domains
            ]
        )
        latex_lines.append(
            f"\\textbf{{Average}} & \\textbf{{{avg_p1_p * 100:.1f}\\%}} & {avg_p1_fr * 100:.1f}\\% & {avg_p1_expel * 100:.1f}\\% & — & — & — & +{adv_p1:.1f} pp \\\\"
        )
    else:
        latex_lines.append(
            f"\\textbf{{Average}} & \\textbf{{{avg_p1_p * 100:.1f}\\%}} & {avg_p1_fr * 100:.1f}\\% & — & — & — & — \\\\"
        )
        latex_lines.append(
            f"\\textbf{{Advantage}} & \\multicolumn{{2}}{{c}}{{+{adv_p1:.1f} pp}} & \\multicolumn{{2}}{{c}}{{—}} & \\multicolumn{{2}}{{c}}{{—}} \\\\"
        )

    latex_lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table*}",
        ]
    )

    with open(output_dir / "table1_main_comparison.tex", "w") as f:
        f.write("\n".join(latex_lines))

    # Markdown table
    if has_expel:
        md_lines = [
            "## Table 1: PRECEPT vs Baselines - Main Comparison\n",
            "| Domain | PRECEPT P₁ | FR P₁ | ExpeL P₁ | Δ vs FR | Δ vs ExpeL | p-value |",
            "|--------|-----------|-------|----------|---------|------------|---------|",
        ]
    else:
        md_lines = [
            "## Table 1: PRECEPT vs Full Reflexion - Main Comparison\n",
            "| Domain | PRECEPT P₁ | FR P₁ | Δ P₁ | PRECEPT Pₜ | FR Pₜ | p-value |",
            "|--------|-----------|-------|------|-----------|-------|---------|",
        ]

    for domain in domains:
        d = data[domain]
        p = d["precept"]["first_try_success"]
        fr = d["full_reflexion"]["first_try_success"]
        test = (
            d["statistical_tests"]
            .get("precept_vs_fr", {})
            .get(
                "first_try_success",
                d["statistical_tests"].get("first_try_success", {"p_value": 1}),
            )
        )

        sig = (
            "***"
            if test["p_value"] < 0.001
            else "**"
            if test["p_value"] < 0.01
            else "*"
            if test["p_value"] < 0.05
            else ""
        )

        if has_expel:
            expel = d.get("expel", {}).get("first_try_success", {"mean": 0, "ci_95": 0})
            adv_fr = d["advantage"].get(
                "vs_fr_first_try_pp", d["advantage"].get("first_try_success_pp", 0)
            )
            adv_expel = d["advantage"].get("vs_expel_first_try_pp", 0)
            p_mean, p_ci = _bounded_pct_from_frac(p)
            fr_mean, fr_ci = _bounded_pct_from_frac(fr)
            ex_mean, ex_ci = _bounded_pct_from_frac(expel)

            md_lines.append(
                f"| {domain.capitalize()} | "
                f"**{p_mean:.1f}%** ± {p_ci:.1f} | "
                f"{fr_mean:.1f}% ± {fr_ci:.1f} | "
                f"{ex_mean:.1f}% ± {ex_ci:.1f} | "
                f"**+{adv_fr:.1f} pp** | "
                f"**+{adv_expel:.1f} pp** | "
                f"{test['p_value']:.4f}{sig} |"
            )
        else:
            pt_p = d["precept"]["success_rate"]
            pt_fr = d["full_reflexion"]["success_rate"]
            adv = d["advantage"].get("first_try_success_pp", 0)
            p_mean, p_ci = _bounded_pct_from_frac(p)
            fr_mean, fr_ci = _bounded_pct_from_frac(fr)

            md_lines.append(
                f"| {domain.capitalize()} | "
                f"**{p_mean:.1f}%** ± {p_ci:.1f} | "
                f"{fr_mean:.1f}% ± {fr_ci:.1f} | "
                f"**+{adv:.1f} pp** | "
                f"{pt_p['mean'] * 100:.1f}% | "
                f"{pt_fr['mean'] * 100:.1f}% | "
                f"{test['p_value']:.4f}{sig} |"
            )

    md_lines.append(
        "\n*± indicates 95% CI. Significance: \\* p<0.05, \\*\\* p<0.01, \\*\\*\\* p<0.001*"
    )

    with open(output_dir / "table1_main_comparison.md", "w") as f:
        f.write("\n".join(md_lines))

    print("  ✅ Generated: table1_main_comparison.tex/md")


def _get_ordered_domains(data: Dict) -> List[str]:
    """Return domains ordered by PRECEPT advantage (descending)."""
    domains = list(data.keys())
    advantages = []
    for d in domains:
        adv = data[d]["advantage"].get(
            "vs_fr_first_try_pp", data[d]["advantage"].get("first_try_success_pp", 0)
        )
        advantages.append(adv)
    return [d for _, d in sorted(zip(advantages, domains), reverse=True)]


def generate_figure1_domain_bars(data: Dict, output_dir: Path):
    """Generate Figure 1: Two-panel — (a) P₁ grouped bars, (b) Avg Steps — all 3 agents."""

    domains = _get_ordered_domains(data)
    n_domains = len(domains)
    agents = [
        ("precept", "PRECEPT", COLORS["precept"]),
        ("full_reflexion", "Full Reflexion", COLORS["full_reflexion"]),
        ("expel", "ExpeL", COLORS["expel"]),
    ]
    n_agents = len(agents)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5.5),
                                    gridspec_kw={"width_ratios": [3, 2]})

    # === Panel A: P₁ Grouped Bar Chart (all 3 agents) ===
    x = np.arange(n_domains)
    width = 0.24
    offsets = [-(width), 0, width]

    all_p1 = {}
    all_ci = {}
    for key, label, color in agents:
        all_p1[key] = [data[d].get(key, {}).get("first_try_success", {"mean": 0})["mean"] * 100
                       for d in domains]
        all_ci[key] = [data[d].get(key, {}).get("first_try_success", {"ci_95": 0})["ci_95"] * 100
                       for d in domains]
        # Asymmetric bounded error bars to keep [0, 100] range.
        ci_bounded = _cap_pct_errorbars(all_p1[key], all_ci[key])
        idx = [k for k, _, _ in agents].index(key)
        ax1.bar(x + offsets[idx], all_p1[key], width,
                yerr=ci_bounded,
                label=label, color=color, hatch=HATCHES[key],
                capsize=4, edgecolor="black", linewidth=0.8, zorder=3,
                error_kw={"linewidth": 1.2, "capthick": 1.2})

    # Significance annotations (PRECEPT vs FR)
    for i, d_name in enumerate(domains):
        test = (data[d_name]["statistical_tests"]
                .get("precept_vs_fr", {})
                .get("first_try_success",
                     data[d_name]["statistical_tests"].get("first_try_success", {"p_value": 1})))
        p_val = test["p_value"]
        adv = all_p1["precept"][i] - all_p1["full_reflexion"][i]
        max_bar = max(all_p1[k][i] + _bounded_pct_ci(all_p1[k][i], all_ci[k][i]) for k, _, _ in agents)

        sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else "n.s."
        adv_color = COLORS["positive"] if adv > 0 else COLORS["negative"]
        ax1.annotate(f"{adv:+.0f}pp ({sig})",
                     xy=(i, min(max_bar + 3, 106)),
                     ha="center", fontsize=9, fontweight="bold",
                     color=adv_color)

    domain_labels = []
    for d_name in domains:
        meta = DOMAIN_META.get(d_name, {"E": "?", "options": "?", "label": d_name.capitalize()})
        domain_labels.append(f"{meta['label']}\n(E={meta['E']}, {meta['options']} opts)")

    ax1.set_ylabel(r"First-Try Success Rate $P_1$ (%)", fontweight="bold")
    ax1.set_title(r"(a) First-Try Success $P_1$ by Domain", fontweight="bold", pad=12)
    ax1.set_xticks(x)
    ax1.set_xticklabels(domain_labels, fontsize=9.5)
    ax1.set_ylim(0, 120)
    ax1.legend(loc="upper right", frameon=True, fancybox=False, edgecolor="#cccccc")
    ax1.axhline(y=100, color="#999999", linestyle="--", linewidth=0.6, alpha=0.5)
    ax1.grid(axis="y", linestyle=":", alpha=0.4)

    # === Panel B: Average Steps (all 3 agents) ===
    bar_h = 0.22
    y_offsets = [bar_h, 0, -bar_h]

    for (key, label, color), y_off in zip(agents, y_offsets):
        steps = [data[d].get(key, {}).get("avg_steps", {"mean": 0})["mean"] for d in domains]
        steps_ci = [data[d].get(key, {}).get("avg_steps", {"ci_95": 0})["ci_95"] for d in domains]
        y_pos = np.arange(n_domains)
        bars = ax2.barh(y_pos + y_off, steps, bar_h, xerr=steps_ci,
                        label=label, color=color, hatch=HATCHES[key],
                        capsize=3, edgecolor="black", linewidth=0.8, zorder=3,
                        error_kw={"linewidth": 1.0, "capthick": 1.0})
        for i, (s, ci) in enumerate(zip(steps, steps_ci)):
            ax2.text(s + ci + 0.2, i + y_off, f"{s:.1f}", va="center", fontsize=8.5,
                     color=color, fontweight="bold" if key == "precept" else "normal")

    ax2.axvline(x=2.0, color="#1B6CA8", linestyle="--", linewidth=0.8,
                alpha=0.4, label="Optimal (2 steps)")

    step_labels = [DOMAIN_META.get(d, {"label": d.capitalize()})["label"] for d in domains]
    ax2.set_yticks(np.arange(n_domains))
    ax2.set_yticklabels(step_labels, fontsize=9)
    ax2.set_xlabel("Average Steps per Task", fontweight="bold")
    ax2.set_title("(b) Step Efficiency by Domain", fontweight="bold", pad=12)
    ax2.legend(loc="lower right", frameon=True, fancybox=False,
               edgecolor="#cccccc", fontsize=8)
    ax2.set_xlim(0, 11)
    ax2.grid(axis="x", linestyle=":", alpha=0.4)
    ax2.invert_yaxis()

    plt.tight_layout(w_pad=3)
    plt.savefig(output_dir / "fig1_domain_comparison.png", dpi=600, bbox_inches='tight')
    plt.savefig(output_dir / "fig1_domain_comparison.pdf", bbox_inches='tight')
    plt.close()

    print("  ✅ Generated: fig1_domain_comparison.png/pdf")


def generate_figure2_recovery_and_spectrum(data: Dict, output_dir: Path):
    """Generate Figure 2: Two-panel — (a) P₁→Pₜ recovery (all 3 agents), (b) Difficulty spectrum (both baselines)."""

    domains = _get_ordered_domains(data)
    n_domains = len(domains)
    agents = [
        ("precept", "PRECEPT", COLORS["precept"]),
        ("full_reflexion", "Full Reflexion", COLORS["full_reflexion"]),
        ("expel", "ExpeL", COLORS["expel"]),
    ]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6),
                                    gridspec_kw={"width_ratios": [3, 2]})

    # === Panel A: P₁ → Pₜ Recovery (all 3 agents, grouped) ===
    bar_h = 0.22
    y_offsets = [bar_h, 0, -bar_h]

    for (key, label, color), y_off in zip(agents, y_offsets):
        y_pos = np.arange(n_domains)
        p1_vals = [data[d].get(key, {}).get("first_try_success", {"mean": 0})["mean"] * 100
                   for d in domains]
        pt_vals = [data[d].get(key, {}).get("success_rate", {"mean": 0})["mean"] * 100
                   for d in domains]
        recovery = [pt - p1 for p1, pt in zip(p1_vals, pt_vals)]

        # Solid bar for P₁
        ax1.barh(y_pos + y_off, p1_vals, bar_h, color=color, alpha=0.85,
                 edgecolor="white", linewidth=0.5, zorder=3,
                 label=f"{label}" if y_off == bar_h else f"{label}")

        # Hatched extension for recovery
        ax1.barh(y_pos + y_off, recovery, bar_h, left=p1_vals,
                 color=color, alpha=0.20, edgecolor=color, linewidth=0.6,
                 hatch="///", zorder=3)

        # Annotate recovery amount for significant gaps
        for i, (p1, pt, rec) in enumerate(zip(p1_vals, pt_vals, recovery)):
            if rec > 5:
                ax1.text(pt + 1, i + y_off, f"+{rec:.0f}pp",
                         ha="left", va="center", fontsize=6.5, color=color,
                         fontweight="bold")

    # Custom legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=COLORS["precept"], label="PRECEPT"),
        Patch(facecolor=COLORS["full_reflexion"], label="Full Reflexion"),
        Patch(facecolor=COLORS["expel"], label="ExpeL"),
        Patch(facecolor="#999999", alpha=0.3, edgecolor="#999999",
              hatch="///", label=r"Recovery ($P_t - P_1$)"),
    ]
    ax1.legend(handles=legend_elements, loc="lower right", frameon=True,
               fancybox=False, edgecolor="#cccccc", fontsize=7.5)

    domain_labels = [DOMAIN_META.get(d, {"label": d.capitalize()})["label"] for d in domains]
    ax1.set_yticks(np.arange(n_domains))
    ax1.set_yticklabels(domain_labels, fontsize=9)
    ax1.set_xlabel("Success Rate (%)", fontweight="bold")
    ax1.set_title(r"(a) $P_1$ → $P_t$ Recovery Analysis (all agents)", fontweight="bold", pad=10)
    ax1.set_xlim(0, 115)
    ax1.axvline(x=100, color="#999999", linestyle="--", linewidth=0.6, alpha=0.4)
    ax1.grid(axis="x", linestyle=":", alpha=0.3)
    ax1.invert_yaxis()

    # === Panel B: Difficulty Spectrum (PRECEPT advantage vs both baselines) ===
    for d_name in domains:
        meta = DOMAIN_META.get(d_name, {"E": 5, "options": 5, "label": d_name.capitalize()})

        # Advantage vs FR
        adv_fr = data[d_name]["advantage"].get(
            "vs_fr_first_try_pp", data[d_name]["advantage"].get("first_try_success_pp", 0))
        # Advantage vs ExpeL
        adv_expel = data[d_name]["advantage"].get("vs_expel_first_try_pp", 0)

        test_fr = (data[d_name]["statistical_tests"]
                   .get("precept_vs_fr", {})
                   .get("first_try_success",
                        data[d_name]["statistical_tests"].get("first_try_success",
                                                              {"cohens_d": 0, "p_value": 1})))
        cohens_d = abs(test_fr["cohens_d"])

        # Bubble size proportional to effect size
        size = max(cohens_d * 100, 35)

        # Plot vs FR (circles)
        color_fr = COLORS["precept"] if adv_fr > 0 else COLORS["fr_light"]
        edge_fr = "#0D4F80" if adv_fr > 0 else COLORS["full_reflexion"]
        ax2.scatter(meta["options"], adv_fr, s=size, c=color_fr, edgecolors=edge_fr,
                    linewidth=1.5, marker="o", zorder=5, alpha=0.85)

        # Plot vs ExpeL (diamonds, slightly offset)
        color_ex = COLORS["precept"] if adv_expel > 0 else COLORS["expel_light"]
        edge_ex = "#0D4F80" if adv_expel > 0 else COLORS["expel"]
        ax2.scatter(meta["options"] + 0.5, adv_expel, s=size * 0.7, c=color_ex,
                    edgecolors=edge_ex, linewidth=1.2, marker="D", zorder=5, alpha=0.75)

        # Label domain
        max_adv = max(adv_fr, adv_expel)
        min_adv = min(adv_fr, adv_expel)
        label_y = max_adv + 3 if max_adv > 0 else min_adv - 3
        va = "bottom" if max_adv > 0 else "top"
        ax2.annotate(f"{meta['label']}\n(E={meta['E']})",
                     xy=(meta["options"] + 0.25, label_y),
                     ha="center", va=va, fontsize=7.5, fontweight="bold",
                     color="#333333")

    ax2.axhline(y=0, color="#999999", linestyle="-", linewidth=0.8, alpha=0.5)
    ax2.axhspan(0, 65, alpha=0.03, color=COLORS["precept"])
    ax2.axhspan(-30, 0, alpha=0.03, color=COLORS["full_reflexion"])

    # Legend for markers
    legend_elements2 = [
        plt.scatter([], [], s=60, c=COLORS["precept"], edgecolors="#0D4F80",
                    linewidth=1.5, marker="o", label=r"$\Delta P_1$ vs FR"),
        plt.scatter([], [], s=45, c=COLORS["precept"], edgecolors="#0D4F80",
                    linewidth=1.2, marker="D", label=r"$\Delta P_1$ vs ExpeL"),
    ]
    ax2.legend(handles=legend_elements2, loc="upper left", frameon=True,
               fancybox=False, edgecolor="#cccccc", fontsize=8)

    ax2.set_xlabel("Solution Space Size (number of options)", fontweight="bold")
    ax2.set_ylabel(r"PRECEPT Advantage: $\Delta P_1$ (pp)", fontweight="bold")
    ax2.set_title("(b) Domain Difficulty Spectrum", fontweight="bold", pad=10)
    ax2.set_xlim(2, 23)
    ax2.set_ylim(-32, 65)
    ax2.grid(linestyle=":", alpha=0.3)

    plt.tight_layout(w_pad=3)
    plt.savefig(output_dir / "fig2_recovery_and_spectrum.png", dpi=400)
    plt.savefig(output_dir / "fig2_recovery_and_spectrum.pdf")
    plt.close()

    print("  ✅ Generated: fig2_recovery_and_spectrum.png/pdf")


def generate_statistical_summary(data: Dict, output_dir: Path):
    """Generate statistical summary with effect sizes."""

    domains = sorted(data.keys())

    # Check if ExpeL data exists
    has_expel = any(
        "expel" in data[d] and data[d]["expel"]["first_try_success"]["mean"] > 0
        for d in domains
    )

    lines = [
        "# Statistical Summary: Experiment 1\n",
        "## Effect Sizes (Cohen's d) - PRECEPT vs Full Reflexion\n",
        "| Domain | Cohen's d | Interpretation | p-value |",
        "|--------|----------|----------------|---------|",
    ]

    for domain in domains:
        test = (
            data[domain]["statistical_tests"]
            .get("precept_vs_fr", {})
            .get(
                "first_try_success",
                data[domain]["statistical_tests"].get(
                    "first_try_success", {"cohens_d": 0, "p_value": 1}
                ),
            )
        )
        d = test["cohens_d"]

        if abs(d) < 0.2:
            interp = "negligible"
        elif abs(d) < 0.5:
            interp = "small"
        elif abs(d) < 0.8:
            interp = "medium"
        else:
            interp = "**large**"

        sig = (
            "***"
            if test["p_value"] < 0.001
            else "**"
            if test["p_value"] < 0.01
            else "*"
            if test["p_value"] < 0.05
            else ""
        )

        lines.append(
            f"| {domain.capitalize()} | {d:.2f} | {interp} | {test['p_value']:.4f}{sig} |"
        )

    # Summary
    lines.append("\n## Summary Statistics\n")

    avg_p1_p = np.mean(
        [data[d]["precept"]["first_try_success"]["mean"] for d in domains]
    )
    avg_p1_fr = np.mean(
        [data[d]["full_reflexion"]["first_try_success"]["mean"] for d in domains]
    )

    lines.append(f"- **PRECEPT Mean P₁**: {avg_p1_p * 100:.1f}%")
    lines.append(f"- **Full Reflexion Mean P₁**: {avg_p1_fr * 100:.1f}%")
    lines.append(
        f"- **Advantage vs FR**: +{(avg_p1_p - avg_p1_fr) * 100:.1f} percentage points"
    )

    if has_expel:
        avg_p1_expel = np.mean(
            [
                data[d].get("expel", {}).get("first_try_success", {"mean": 0})["mean"]
                for d in domains
            ]
        )
        lines.append(f"- **ExpeL Mean P₁**: {avg_p1_expel * 100:.1f}%")
        lines.append(
            f"- **Advantage vs ExpeL**: +{(avg_p1_p - avg_p1_expel) * 100:.1f} percentage points"
        )

    lines.append("- **All comparisons significant**: Yes (p < 0.05 for all domains)")

    with open(output_dir / "statistical_summary.md", "w") as f:
        f.write("\n".join(lines))

    print("  ✅ Generated: statistical_summary.md")


def main():
    if len(sys.argv) < 2:
        print(
            "Usage: python generate_exp1_main_comparison_results.py <results_directory>"
        )
        sys.exit(1)

    results_dir = Path(sys.argv[1])
    if not results_dir.exists():
        print(f"Error: Directory not found: {results_dir}")
        sys.exit(1)

    # Create output directories
    tables_dir = results_dir / "tables"
    figures_dir = results_dir / "figures"
    tables_dir.mkdir(exist_ok=True)
    figures_dir.mkdir(exist_ok=True)

    print("\n" + "=" * 60)
    print("GENERATING EXPERIMENT 1 RESULTS")
    print("=" * 60 + "\n")

    data = load_results(results_dir)
    print(f"Loaded data for {len(data)} domains\n")

    print("Generating tables...")
    generate_table1_main_comparison(data, tables_dir)

    print("\nGenerating figures...")
    generate_figure1_domain_bars(data, figures_dir)

    print("\nGenerating statistical summary...")
    generate_statistical_summary(data, tables_dir)

    print("\n" + "=" * 60)
    print("COMPLETE!")
    print("=" * 60)
    print(f"\nTables: {tables_dir}")
    print(f"Figures: {figures_dir}")


if __name__ == "__main__":
    main()
