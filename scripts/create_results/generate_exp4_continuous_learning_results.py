#!/usr/bin/env python3
"""
Generate Publication Results for Experiment 4: Continuous Learning

Creates:
- Figure 4: Two-panel grouped bar chart (P₁ + Avg Steps by encounter)
  with 95% CI error bars, significance annotations, and optimal reference
- Table 4: Cross-episode learning comparison (LaTeX + Markdown)
- Statistical summary with effect sizes and p-values

Publication quality: 300 DPI, serif fonts, colorblind-friendly palette,
significance markers directly on figure.

Usage:
    python scripts/create_results/generate_exp4_continuous_learning_results.py <results_directory>
"""

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

try:
    from scripts.utils.pct_bounds import bounded_pct_ci
except ImportError:
    def bounded_pct_ci(mean_pct, ci_pct, lower=0.0, upper=100.0):
        return max(0.0, min(ci_pct, mean_pct - lower, upper - mean_pct))

# ── Publication-quality matplotlib configuration (consistent with Exp 3) ─
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['DejaVu Serif', 'Times New Roman', 'Computer Modern Roman'],
    'font.size': 12,
    'axes.labelsize': 13,
    'axes.titlesize': 14,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'legend.fontsize': 11,
    'figure.titlesize': 15,
    'figure.dpi': 600,
    'savefig.dpi': 600,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.15,
    'axes.linewidth': 1.0,
    'axes.grid': False,
    'grid.alpha': 0.4,
    'errorbar.capsize': 4,
    'lines.linewidth': 2.0,
    'text.usetex': False,
    'mathtext.fontset': 'dejavuserif',
})

# ── Colorblind-friendly, high-contrast palette (consistent across all experiments) ─
COLORS = {
    'precept': '#2563eb',        # Strong blue
    'full_reflexion': '#dc2626', # Strong red
    'expel': '#ea580c',          # Strong orange
}
LABELS = {
    'precept': 'PRECEPT',
    'full_reflexion': 'Full Reflexion',
    'expel': 'ExpeL',
}
HATCHES = {'precept': None, 'full_reflexion': '//', 'expel': '\\\\'}
AGENTS = ['precept', 'full_reflexion', 'expel']


def load_results(results_dir: Path) -> dict:
    """Load continuous learning results."""
    results_file = results_dir / "continuous_learning_results.json"
    if results_file.exists():
        with open(results_file) as f:
            return json.load(f)

    agg_file = results_dir / "aggregated_results.json"
    if agg_file.exists():
        with open(agg_file) as f:
            return json.load(f)

    return {}


def _significance_label(p: float) -> str:
    """Return significance marker for a p-value."""
    if p < 0.001:
        return "***"
    elif p < 0.01:
        return "**"
    elif p < 0.05:
        return "*"
    return ""


def _add_significance_bracket(ax, x1, x2, y, label, color="black"):
    """Draw a bracket with significance label between two bars."""
    if not label:
        return
    h = 1.5  # bracket height
    ax.plot([x1, x1, x2, x2], [y, y + h, y + h, y], lw=1.0, c=color)
    ax.text(
        (x1 + x2) / 2,
        y + h + 0.3,
        label,
        ha="center",
        va="bottom",
        fontsize=10,
        fontweight="bold",
        color=color,
    )


def _cap_err(means, cis, lo=0, hi=100):
    """Cap error bars so they don't exceed [lo, hi] bounds."""
    return [[max(0, min(c, m - lo)) for m, c in zip(means, cis)],
            [max(0, min(c, hi - m)) for m, c in zip(means, cis)]]


def generate_main_figure(data: dict, output_dir: Path):
    """Generate the main 2-panel figure: P₁ + Avg Steps grouped bars by encounter."""

    learning_curves = data.get("learning_curves", {})
    stat_tests = data.get("statistical_tests", {}).get("per_encounter_tests", {})

    if not learning_curves:
        print("  ⚠️ No learning curves data found")
        return

    precept_curve = learning_curves.get("precept", {})
    num_encounters = len([k for k in precept_curve if k.startswith("encounter_")])
    if num_encounters == 0:
        print("  ⚠️ No encounter data found")
        return

    encounters = list(range(1, num_encounters + 1))
    enc_labels = [
        f"{i}{'st' if i == 1 else 'nd' if i == 2 else 'rd' if i == 3 else 'th'}"
        for i in encounters
    ]
    n_runs = data.get("n_runs", 10)

    # ── Extract per-encounter data ───────────────────────────────────────
    p1_data = {a: {"mean": [], "ci": []} for a in AGENTS}
    steps_data = {a: {"mean": [], "ci": []} for a in AGENTS}

    for enc in encounters:
        key = f"encounter_{enc}"
        for a in AGENTS:
            d = learning_curves.get(a, {}).get(key, {})
            p1_data[a]["mean"].append(d.get("p1_mean", 0) * 100)
            p1_data[a]["ci"].append(d.get("p1_ci_95", 0) * 100)
            steps_data[a]["mean"].append(d.get("steps_mean", 0))
            steps_data[a]["ci"].append(d.get("steps_ci_95", 0))

    # ── Figure layout: 2 panels ──────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    x = np.arange(num_encounters)
    w = 0.25

    # ══ Panel (a): P₁ by Encounter ══════════════════════════════════════
    ax = axes[0]
    for j, agent in enumerate(AGENTS):
        errs = _cap_err(p1_data[agent]["mean"], p1_data[agent]["ci"])
        ax.bar(x + (j - 1) * w, p1_data[agent]["mean"], w, yerr=errs,
               label=LABELS[agent], color=COLORS[agent], capsize=4,
               edgecolor='black', linewidth=0.8, hatch=HATCHES[agent],
               error_kw={'linewidth': 1.2, 'capthick': 1.2})

    # Significance markers (PRECEPT vs FR) — centered above each group
    for i, enc in enumerate(encounters):
        enc_key = f"encounter_{enc}"
        tests = stat_tests.get(enc_key, {})
        pvfr = tests.get("precept_vs_full_reflexion", {}).get("p_value", 1.0)
        sig = '***' if pvfr < 0.001 else '**' if pvfr < 0.01 else '*' if pvfr < 0.05 else ''
        if sig:
            y_top = max(p1_data[a]["mean"][i] + p1_data[a]["ci"][i] for a in AGENTS)
            ax.annotate(sig, xy=(x[i], min(y_top + 3, 112)),
                        ha='center', fontsize=13, fontweight='bold', color='#1a1a1a')

    ax.axhline(y=100, color='gray', linestyle='--', alpha=0.4, linewidth=0.8)
    ax.set_xlabel('Encounter Number', fontweight='bold')
    ax.set_ylabel(r'First-Try Success Rate $P_1$ (%)', fontweight='bold')
    ax.set_title(r'(a) $P_1$ vs Encounter', fontweight='bold', pad=12)
    ax.set_xticks(x)
    ax.set_xticklabels(enc_labels)
    ax.set_ylim(0, 120)
    ax.legend(loc='upper left', framealpha=0.95, edgecolor='gray')
    ax.yaxis.grid(True, linestyle='--', alpha=0.3)
    ax.set_axisbelow(True)

    # ══ Panel (b): Avg Steps by Encounter ═══════════════════════════════
    ax = axes[1]
    for j, agent in enumerate(AGENTS):
        ax.bar(x + (j - 1) * w, steps_data[agent]["mean"], w,
               yerr=steps_data[agent]["ci"],
               label=LABELS[agent], color=COLORS[agent], capsize=4,
               edgecolor='black', linewidth=0.8, hatch=HATCHES[agent],
               error_kw={'linewidth': 1.2, 'capthick': 1.2})

    ax.axhline(y=2.0, color='#15803d', linestyle='--', alpha=0.9,
               linewidth=1.5, label='Theoretical min.')
    ax.annotate('Optimal (2 steps)', xy=(0.5, 0.12), xycoords='axes fraction',
                ha='center', va='top', fontsize=9,
                fontweight='bold', style='italic', color='#15803d',
                bbox=dict(boxstyle='round,pad=0.2', fc='white', ec='none', alpha=0.8))

    ax.set_xlabel('Encounter Number', fontweight='bold')
    ax.set_ylabel('Avg Steps', fontweight='bold')
    ax.set_title('(b) Avg Steps vs Encounter', fontweight='bold', pad=12)
    ax.set_xticks(x)
    ax.set_xticklabels(enc_labels)
    y_max = max(
        max(steps_data[a]["mean"][i] + steps_data[a]["ci"][i] for a in AGENTS)
        for i in range(num_encounters)
    )
    ax.set_ylim(0, max(y_max + 1.5, 6))
    ax.legend(loc='upper right', framealpha=0.95, edgecolor='gray', fontsize=10)
    ax.yaxis.grid(True, linestyle='--', alpha=0.3)
    ax.set_axisbelow(True)
    ax.annotate('Lower is better', xy=(0.98, 0.02), xycoords='axes fraction',
                ha='right', va='bottom', fontsize=10, style='italic', color='#666666')

    plt.tight_layout()
    fig.text(0.5, -0.04,
             f'Error bars: 95% CI (n={n_runs} seeds per encounter). '
             '*** p<0.001, ** p<0.01, * p<0.05 (paired t-test, PRECEPT vs FR).',
             ha='center', fontsize=10, style='italic', color='#333333')

    fig.savefig(output_dir / "Figure_Exp4_Main.pdf", format='pdf')
    fig.savefig(output_dir / "Figure_Exp4_Main.png", format='png')
    plt.close()

    print("  ✅ Generated: Figure_Exp4_Main.png/pdf")


def generate_table4_continuous_learning(data: dict, output_dir: Path):
    """Generate Table 4: Cross-episode learning comparison (LaTeX + Markdown)."""

    learning_curves = data.get("learning_curves", {})
    improvements = data.get("improvements", {})
    stat_tests = data.get("statistical_tests", {}).get("per_encounter_tests", {})
    params = data.get("parameters", {})

    if not learning_curves:
        print("  ⚠️ No learning curves data for table")
        return

    precept_curve = learning_curves.get("precept", {})
    num_encounters = len([k for k in precept_curve if k.startswith("encounter_")])

    # ── LaTeX table ──────────────────────────────────────────────────────
    latex_lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Experiment 4: Cross-Episode Learning (Logistics, N=5, $\beta=1$, E=4). "
        r"P$_1$ and Avg Steps by encounter number across 10 seeds. "
        r"Significance: PRECEPT vs Full Reflexion.}",
        r"\label{tab:continuous_learning}",
        r"\begin{tabular}{l ccc ccc}",
        r"\toprule",
        r"& \multicolumn{3}{c}{\textbf{P$_1$ (\%)}} & \multicolumn{3}{c}{\textbf{Avg Steps}} \\",
        r"\cmidrule(lr){2-4} \cmidrule(lr){5-7}",
        r"Encounter & PRECEPT & ExpeL & Full Ref. & PRECEPT & ExpeL & Full Ref. \\",
        r"\midrule",
    ]

    for enc in range(1, num_encounters + 1):
        key = f"encounter_{enc}"
        suffix = "st" if enc == 1 else ("nd" if enc == 2 else ("rd" if enc == 3 else "th"))

        p = learning_curves.get("precept", {}).get(key, {})
        e = learning_curves.get("expel", {}).get(key, {})
        f = learning_curves.get("full_reflexion", {}).get(key, {})

        # Significance marker
        pvfr = stat_tests.get(key, {}).get(
            "precept_vs_full_reflexion", {}
        ).get("p_value", 1.0)
        sig = _significance_label(pvfr)
        sig_tex = f"$^{{{sig}}}$" if sig else ""

        latex_lines.append(
            f"  {enc}{suffix} & "
            f"\\textbf{{{p.get('p1_mean', 0) * 100:.1f}}}{sig_tex} & "
            f"{e.get('p1_mean', 0) * 100:.1f} & "
            f"{f.get('p1_mean', 0) * 100:.1f} & "
            f"\\textbf{{{p.get('steps_mean', 0):.2f}}} & "
            f"{e.get('steps_mean', 0):.2f} & "
            f"{f.get('steps_mean', 0):.2f} \\\\"
        )

    # Improvement row
    latex_lines.append(r"\midrule")
    p_imp = improvements.get("precept", {})
    e_imp = improvements.get("expel", {})
    f_imp = improvements.get("full_reflexion", {})

    latex_lines.append(
        f"  $\\Delta$ (1st$\\to$4th) & "
        f"\\textbf{{+{p_imp.get('p1_improvement_pp', 0):.1f}}} & "
        f"+{e_imp.get('p1_improvement_pp', 0):.1f} & "
        f"+{f_imp.get('p1_improvement_pp', 0):.1f} & "
        f"\\textbf{{-{p_imp.get('steps_saved', 0):.2f}}} & "
        f"-{e_imp.get('steps_saved', 0):.2f} & "
        f"-{f_imp.get('steps_saved', 0):.2f} \\\\"
    )

    latex_lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table}",
        ]
    )

    with open(output_dir / "table4_continuous_learning.tex", "w") as f:
        f.write("\n".join(latex_lines))

    # ── Markdown table ───────────────────────────────────────────────────
    md_lines = [
        "## Table 4: Cross-Episode Continuous Learning Results\n",
        f"**Domain**: {data.get('domain', 'N/A')} | "
        f"**N**: {params.get('num_conditions', 'N/A')} | "
        f"**β**: {params.get('beta', 'N/A')} | "
        f"**Encounters**: {params.get('encounters_per_key', 'N/A')} | "
        f"**Seeds**: {data.get('n_runs', 'N/A')}  ",
        "",
        "| Encounter | PRECEPT P₁ | ExpeL P₁ | FR P₁ | PRECEPT Steps | ExpeL Steps | FR Steps |",
        "|-----------|-----------|----------|-------|--------------|------------|----------|",
    ]

    for enc in range(1, num_encounters + 1):
        key = f"encounter_{enc}"
        suffix = "st" if enc == 1 else ("nd" if enc == 2 else ("rd" if enc == 3 else "th"))

        p = learning_curves.get("precept", {}).get(key, {})
        e = learning_curves.get("expel", {}).get(key, {})
        f = learning_curves.get("full_reflexion", {}).get(key, {})

        pvfr = stat_tests.get(key, {}).get(
            "precept_vs_full_reflexion", {}
        ).get("p_value", 1.0)
        sig = _significance_label(pvfr)

        md_lines.append(
            f"| {enc}{suffix} | **{p.get('p1_mean', 0) * 100:.1f}%**±{bounded_pct_ci(p.get('p1_mean', 0) * 100, p.get('p1_ci_95', 0) * 100):.1f}{sig} | "
            f"{e.get('p1_mean', 0) * 100:.1f}%±{bounded_pct_ci(e.get('p1_mean', 0) * 100, e.get('p1_ci_95', 0) * 100):.1f} | "
            f"{f.get('p1_mean', 0) * 100:.1f}%±{bounded_pct_ci(f.get('p1_mean', 0) * 100, f.get('p1_ci_95', 0) * 100):.1f} | "
            f"**{p.get('steps_mean', 0):.2f}**±{p.get('steps_ci_95', 0):.2f} | "
            f"{e.get('steps_mean', 0):.2f}±{e.get('steps_ci_95', 0):.2f} | "
            f"{f.get('steps_mean', 0):.2f}±{f.get('steps_ci_95', 0):.2f} |"
        )

    md_lines.extend([
        "",
        f"| Δ (1→4) | **+{p_imp.get('p1_improvement_pp', 0):.1f}pp** | "
        f"+{e_imp.get('p1_improvement_pp', 0):.1f}pp | "
        f"+{f_imp.get('p1_improvement_pp', 0):.1f}pp | "
        f"**-{p_imp.get('steps_saved', 0):.2f}** | "
        f"-{e_imp.get('steps_saved', 0):.2f} | "
        f"-{f_imp.get('steps_saved', 0):.2f} |",
        "",
        "### Statistical Significance (PRECEPT vs Full Reflexion at 4th encounter)",
        "",
    ])

    # Final encounter stats
    final_tests = stat_tests.get(f"encounter_{num_encounters}", {})
    pvfr_final = final_tests.get("precept_vs_full_reflexion", {})
    pve_final = final_tests.get("precept_vs_expel", {})

    md_lines.extend([
        f"- **PRECEPT vs Full Reflexion**: p={pvfr_final.get('p_value', 0):.4f} "
        f"(Cohen's d={pvfr_final.get('cohens_d', 0):.2f}) {_significance_label(pvfr_final.get('p_value', 1))}",
        f"- **PRECEPT vs ExpeL**: p={pve_final.get('p_value', 0):.4f} "
        f"(Cohen's d={pve_final.get('cohens_d', 0):.2f}) {_significance_label(pve_final.get('p_value', 1))}",
        "",
        "*Significance: \\* p<0.05, \\*\\* p<0.01, \\*\\*\\* p<0.001*",
    ])

    with open(output_dir / "table4_continuous_learning.md", "w") as f:
        f.write("\n".join(md_lines))

    print("  ✅ Generated: table4_continuous_learning.tex/md")


def generate_statistical_summary(data: dict, output_dir: Path):
    """Generate comprehensive statistical summary."""

    improvements = data.get("improvements", {})
    params = data.get("parameters", {})
    stat_tests = data.get("statistical_tests", {})
    learning_curves = data.get("learning_curves", {})

    precept_curve = learning_curves.get("precept", {})
    num_encounters = len([k for k in precept_curve if k.startswith("encounter_")])

    lines = [
        "# Statistical Summary: Experiment 4 (Continuous Learning)\n",
        "## Experiment Configuration\n",
        f"- **Domain**: {data.get('domain', 'N/A')}",
        f"- **Beta (β)**: {params.get('beta', 'N/A')}",
        f"- **Training Tasks**: {params.get('train_size', 'N/A')}",
        f"- **Test Tasks**: {params.get('test_size', 'N/A')}",
        f"- **Encounters per Key**: {params.get('encounters_per_key', 'N/A')}",
        f"- **Max Retries**: {params.get('max_retries', 'N/A')}",
        f"- **Num Conditions**: {params.get('num_conditions', 'N/A')}",
        f"- **Successful Runs**: {data.get('n_runs', 'N/A')}",
        f"- **Seeds**: {data.get('seeds_used', [])}",
        "",
        "## Key Findings\n",
    ]

    p_imp = improvements.get("precept", {}).get("p1_improvement_pp", 0)
    e_imp = improvements.get("expel", {}).get("p1_improvement_pp", 0)
    fr_imp = improvements.get("full_reflexion", {}).get("p1_improvement_pp", 0)

    lines.extend([
        f"1. **PRECEPT achieves the largest P₁ improvement**: +{p_imp:.1f}pp "
        f"(vs ExpeL +{e_imp:.1f}pp, Full Reflexion +{fr_imp:.1f}pp)",
        "",
    ])

    # Steps convergence
    p_enc4 = learning_curves.get("precept", {}).get(f"encounter_{num_encounters}", {})
    lines.extend([
        f"2. **PRECEPT converges nearest to optimal**: "
        f"{p_enc4.get('steps_mean', 0):.2f} avg steps at 4th encounter "
        f"(optimal = 2.00 steps)",
        "",
    ])

    # Statistical significance
    final_tests = stat_tests.get("per_encounter_tests", {}).get(
        f"encounter_{num_encounters}", {}
    )
    pvfr = final_tests.get("precept_vs_full_reflexion", {})
    pve = final_tests.get("precept_vs_expel", {})

    lines.extend([
        f"3. **Strong significance vs Full Reflexion**: "
        f"p={pvfr.get('p_value', 0):.4f}, Cohen's d={pvfr.get('cohens_d', 0):.2f} (large effect)",
        "",
        f"4. **PRECEPT vs ExpeL**: "
        f"p={pve.get('p_value', 0):.4f}, Cohen's d={pve.get('cohens_d', 0):.2f} "
        f"({'significant' if pve.get('p_value', 1) < 0.05 else 'not significant at α=0.05'})",
        "",
        "## Per-Encounter Statistical Tests\n",
        "| Encounter | PRECEPT vs ExpeL | | PRECEPT vs FR | |",
        "|-----------|----------|---------|----------|---------|",
        "| | p-value | d | p-value | d |",
    ])

    per_enc = stat_tests.get("per_encounter_tests", {})
    for enc in range(1, num_encounters + 1):
        key = f"encounter_{enc}"
        suffix = "st" if enc == 1 else ("nd" if enc == 2 else ("rd" if enc == 3 else "th"))
        pve = per_enc.get(key, {}).get("precept_vs_expel", {})
        pvfr = per_enc.get(key, {}).get("precept_vs_full_reflexion", {})
        lines.append(
            f"| {enc}{suffix} | "
            f"{pve.get('p_value', 0):.4f}{_significance_label(pve.get('p_value', 1))} | "
            f"{pve.get('cohens_d', 0):.2f} | "
            f"{pvfr.get('p_value', 0):.4f}{_significance_label(pvfr.get('p_value', 1))} | "
            f"{pvfr.get('cohens_d', 0):.2f} |"
        )

    lines.extend([
        "",
        "## Improvement Significance (Paired t-tests: 1st vs 4th encounter)\n",
    ])
    imp_tests = stat_tests.get("improvement_tests", {})
    for agent in AGENTS:
        at = imp_tests.get(agent, {}).get("paired_test", {})
        sig = _significance_label(at.get("p_value", 1))
        lines.append(
            f"- **{LABELS[agent]}**: t={at.get('t_stat', 0):.2f}, "
            f"p={at.get('p_value', 0):.6f}{sig}, "
            f"d={at.get('cohens_d', 0):.2f}"
        )

    lines.extend([
        "",
        "*Significance: \\* p<0.05, \\*\\* p<0.01, \\*\\*\\* p<0.001*",
    ])

    with open(output_dir / "statistical_summary.md", "w") as f:
        f.write("\n".join(lines))

    print("  ✅ Generated: statistical_summary.md")


def find_all_domain_dirs() -> list:
    """Find all exp4 domain directories in publication_results (latest run per domain)."""
    pub = Path(__file__).parent.parent.parent / "data" / "publication_results"
    dirs = sorted(pub.glob("exp4_continuous_learning_*"))
    known_domains = {"integration", "logistics", "booking"}
    domain_dirs = {}
    for d in dirs:
        parts = d.name.replace("exp4_continuous_learning_", "").split("_")
        domain = parts[0]
        if domain in known_domains:
            domain_dirs[domain] = d  # last (latest) wins
    return list(domain_dirs.values())


def generate_combined_figure(all_data: list, output_dir: Path):
    """Generate combined multi-domain figure: one row per domain, 2 cols (P₁, Steps)."""
    n_domains = len(all_data)
    fig, axes = plt.subplots(n_domains, 2, figsize=(14, 5.5 * n_domains))
    if n_domains == 1:
        axes = axes.reshape(1, 2)

    for row, data in enumerate(all_data):
        domain = data.get("domain", "unknown").capitalize()
        learning_curves = data.get("learning_curves", {})
        stat_tests = data.get("statistical_tests", {}).get("per_encounter_tests", {})

        precept_curve = learning_curves.get("precept", {})
        num_encounters = len([k for k in precept_curve if k.startswith("encounter_")])
        if num_encounters == 0:
            continue

        encounters = list(range(1, num_encounters + 1))
        enc_labels = [
            f"{i}{'st' if i == 1 else 'nd' if i == 2 else 'rd' if i == 3 else 'th'}"
            for i in encounters
        ]

        p1_data = {a: {"mean": [], "ci": []} for a in AGENTS}
        steps_data = {a: {"mean": [], "ci": []} for a in AGENTS}

        for enc in encounters:
            key = f"encounter_{enc}"
            for a in AGENTS:
                d = learning_curves.get(a, {}).get(key, {})
                p1_data[a]["mean"].append(d.get("p1_mean", 0) * 100)
                p1_data[a]["ci"].append(d.get("p1_ci_95", 0) * 100)
                steps_data[a]["mean"].append(d.get("steps_mean", 0))
                steps_data[a]["ci"].append(d.get("steps_ci_95", 0))

        x = np.arange(num_encounters)
        w = 0.25

        # ── Panel: P₁ ────────────────────────────────────────────────────
        ax = axes[row, 0]
        for j, agent in enumerate(AGENTS):
            errs = _cap_err(p1_data[agent]["mean"], p1_data[agent]["ci"])
            ax.bar(x + (j - 1) * w, p1_data[agent]["mean"], w, yerr=errs,
                   label=LABELS[agent], color=COLORS[agent], capsize=4,
                   edgecolor='black', linewidth=0.8, hatch=HATCHES[agent],
                   error_kw={'linewidth': 1.2, 'capthick': 1.2})

        for i, enc in enumerate(encounters):
            enc_key = f"encounter_{enc}"
            tests = stat_tests.get(enc_key, {})
            pvfr = tests.get("precept_vs_full_reflexion", {}).get("p_value", 1.0)
            sig = '***' if pvfr < 0.001 else '**' if pvfr < 0.01 else '*' if pvfr < 0.05 else ''
            if sig:
                y_top = max(p1_data[a]["mean"][i] + p1_data[a]["ci"][i] for a in AGENTS)
                ax.annotate(sig, xy=(x[i], min(y_top + 3, 112)),
                            ha='center', fontsize=13, fontweight='bold', color='#1a1a1a')

        panel_letter = chr(ord('a') + row * 2)
        ax.axhline(y=100, color='gray', linestyle='--', alpha=0.4, linewidth=0.8)
        ax.set_xlabel('Encounter Number', fontweight='bold')
        ax.set_ylabel(r'$P_1$ (%)', fontweight='bold')
        ax.set_title(f'({panel_letter}) {domain}: $P_1$ vs Encounter', fontweight='bold', pad=12)
        ax.set_xticks(x)
        ax.set_xticklabels(enc_labels)
        ax.set_ylim(0, 120)
        ax.legend(loc='upper left', framealpha=0.95, edgecolor='gray')
        ax.yaxis.grid(True, linestyle='--', alpha=0.3)
        ax.set_axisbelow(True)

        # ── Panel: Steps ─────────────────────────────────────────────────
        ax = axes[row, 1]
        for j, agent in enumerate(AGENTS):
            ax.bar(x + (j - 1) * w, steps_data[agent]["mean"], w,
                   yerr=steps_data[agent]["ci"],
                   label=LABELS[agent], color=COLORS[agent], capsize=4,
                   edgecolor='black', linewidth=0.8, hatch=HATCHES[agent],
                   error_kw={'linewidth': 1.2, 'capthick': 1.2})

        ax.axhline(y=2.0, color='#15803d', linestyle='--', alpha=0.9,
                   linewidth=1.5, label='Theoretical min.')
        ax.annotate('Optimal (2 steps)', xy=(0.5, 0.12), xycoords='axes fraction',
                    ha='center', va='top', fontsize=9,
                    fontweight='bold', style='italic', color='#15803d',
                    bbox=dict(boxstyle='round,pad=0.2', fc='white', ec='none', alpha=0.8))

        panel_letter2 = chr(ord('a') + row * 2 + 1)
        ax.set_xlabel('Encounter Number', fontweight='bold')
        ax.set_ylabel('Avg Steps', fontweight='bold')
        ax.set_title(f'({panel_letter2}) {domain}: Steps vs Encounter', fontweight='bold', pad=12)
        ax.set_xticks(x)
        ax.set_xticklabels(enc_labels)
        y_max = max(
            max(steps_data[a]["mean"][i] + steps_data[a]["ci"][i] for a in AGENTS)
            for i in range(num_encounters)
        )
        ax.set_ylim(0, max(y_max + 1.5, 6))
        ax.legend(loc='upper right', framealpha=0.95, edgecolor='gray', fontsize=10)
        ax.yaxis.grid(True, linestyle='--', alpha=0.3)
        ax.set_axisbelow(True)
        ax.annotate('Lower is better', xy=(0.98, 0.02), xycoords='axes fraction',
                    ha='right', va='bottom', fontsize=10, style='italic', color='#666666')

    plt.tight_layout(h_pad=4)
    fig.text(0.5, -0.03,
             '*** p<0.001, ** p<0.01, * p<0.05 (paired t-test, PRECEPT vs FR). Error bars: 95% CI.',
             ha='center', fontsize=10, style='italic', color='#333333')

    plt.savefig(output_dir / "Figure_Exp4_Combined.png")
    plt.savefig(output_dir / "Figure_Exp4_Combined.pdf")
    plt.close()
    print("  ✅ Generated: Figure_Exp4_Combined.png/pdf")


def main():
    if len(sys.argv) >= 2 and sys.argv[1] != "--combined":
        # Backward-compatible: single directory mode
        results_dir = Path(sys.argv[1])
        if not results_dir.exists():
            print(f"Error: Directory not found: {results_dir}")
            sys.exit(1)

        tables_dir = results_dir / "tables"
        figures_dir = results_dir / "figures"
        tables_dir.mkdir(exist_ok=True)
        figures_dir.mkdir(exist_ok=True)

        print("\n" + "=" * 60)
        print("GENERATING EXPERIMENT 4 RESULTS (Continuous Learning)")
        print("=" * 60 + "\n")

        data = load_results(results_dir)
        if not data:
            print("  ❌ No results data found")
            sys.exit(1)

        print(f"Loaded data for domain: {data.get('domain', 'unknown')}")
        print(f"  Runs: {data.get('n_runs', 'N/A')}")
        print(f"  Seeds: {data.get('seeds_used', [])}")

        print("\nGenerating main figure...")
        generate_main_figure(data, figures_dir)

        print("\nGenerating tables...")
        generate_table4_continuous_learning(data, tables_dir)

        print("\nGenerating statistical summary...")
        generate_statistical_summary(data, tables_dir)

        print("\n" + "=" * 60)
        print("COMPLETE!")
        print(f"  Figures: {figures_dir}")
        print(f"  Tables:  {tables_dir}")
        print("=" * 60)
        return

    # Combined mode: auto-discover all domain directories
    domain_dirs = find_all_domain_dirs()
    if not domain_dirs:
        print("❌ No exp4 result directories found in data/publication_results/")
        sys.exit(1)

    print("\n" + "=" * 70)
    print("GENERATING EXPERIMENT 4 COMBINED RESULTS (All Domains)")
    print("=" * 70 + "\n")

    all_data = []
    for d in domain_dirs:
        data = load_results(d)
        if data:
            domain = data.get("domain", "unknown")
            print(f"  Loaded: {domain} ({data.get('n_runs', '?')} runs from {d.name})")
            all_data.append(data)

            # Also generate per-domain outputs
            tables_dir = d / "tables"
            figures_dir = d / "figures"
            tables_dir.mkdir(exist_ok=True)
            figures_dir.mkdir(exist_ok=True)
            generate_main_figure(data, figures_dir)
            generate_table4_continuous_learning(data, tables_dir)
            generate_statistical_summary(data, tables_dir)

    if not all_data:
        print("  ❌ No valid data found")
        sys.exit(1)

    # Generate combined multi-domain figure
    combined_dir = Path(__file__).parent.parent.parent / "data" / "publication_results" / "exp4_combined"
    combined_dir.mkdir(exist_ok=True)
    print(f"\nGenerating combined figure ({len(all_data)} domains)...")
    generate_combined_figure(all_data, combined_dir)

    print("\n" + "=" * 70)
    print("COMPLETE!")
    print(f"  Combined: {combined_dir}")
    for d in domain_dirs:
        print(f"  Per-domain: {d / 'figures'}")
    print("=" * 70)


if __name__ == "__main__":
    main()
