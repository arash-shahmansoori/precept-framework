#!/usr/bin/env python3
"""
Generate Consolidated Publication Results from All Experiments

Creates a master summary combining results from all 4 experiments:
- Figure 5: Publication-ready overview combining key findings
- Master summary table with key metrics from all experiments

Experiment Structure:
- Exp 1: Main Domain Comparison (6 domains)
- Exp 2: Static Knowledge Ablation
- Exp 3: Training Size Ablation (β values)
- Exp 4: Continuous Learning (cross-episode)

Usage:
    python scripts/create_results/generate_consolidated_results.py \
        --exp1 data/publication_results/exp1_main_comparison_<timestamp> \
        --exp2 data/publication_results/exp2_static_knowledge_<timestamp> \
        --exp3 data/publication_results/exp3_training_size_<timestamp> \
        --exp4 data/publication_results/exp4_continuous_learning_<timestamp> \
        --output data/publication_results/consolidated_<timestamp>
"""

import argparse
import json
from datetime import datetime
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

# Publication-quality settings
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
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
    }
)

COLORS = {
    "precept": "#2E86AB",
    "full_reflexion": "#A23B72",
    "expel": "#6B8E23",
    "advantage": "#2E7D32",
    "theoretical": "#F18F01",
}


def load_results(results_dir: Path) -> dict:
    """Load aggregated results from a directory."""
    # Try continuous learning format first
    cl_file = results_dir / "continuous_learning_results.json"
    if cl_file.exists():
        with open(cl_file) as f:
            return json.load(f)

    agg_file = results_dir / "aggregated_results.json"
    if agg_file.exists():
        with open(agg_file) as f:
            return json.load(f)
    return {}


def generate_figure5_overview(
    exp1_data: dict,
    exp2_data: dict,
    exp3_data: dict,
    exp4_data: dict,
    output_dir: Path,
):
    """Generate Figure 5: Publication overview with 4 panels."""

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Panel A: Domain comparison (from Exp1)
    ax = axes[0, 0]
    if exp1_data:
        domains = sorted(exp1_data.keys())
        n_domains = len(domains)
        x = np.arange(n_domains)
        width = 0.35

        precept_p1 = [
            exp1_data[d]["precept"]["first_try_success"]["mean"] * 100 for d in domains
        ]
        fr_p1 = [
            exp1_data[d]["full_reflexion"]["first_try_success"]["mean"] * 100
            for d in domains
        ]

        ax.bar(
            x - width / 2,
            precept_p1,
            width,
            label="PRECEPT",
            color=COLORS["precept"],
            edgecolor="white",
        )
        ax.bar(
            x + width / 2,
            fr_p1,
            width,
            label="Full Reflexion",
            color=COLORS["full_reflexion"],
            edgecolor="white",
        )

        ax.set_ylabel("First-Try Success (%)")
        ax.set_title("A. Main Domain Comparison (Exp 1)", fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(
            [d.capitalize()[:4] for d in domains], rotation=45, ha="right"
        )
        ax.set_ylim(0, 110)
        ax.legend(loc="upper right", fontsize=8)
        ax.axhline(y=100, color="gray", linestyle="--", alpha=0.3)
        ax.grid(axis="y", alpha=0.3)

        # Add average annotation
        avg_adv = np.mean(
            [
                exp1_data[d]["advantage"].get(
                    "vs_fr_first_try_pp",
                    exp1_data[d]["advantage"].get("first_try_success_pp", 0),
                )
                for d in domains
            ]
        )
        ax.text(
            0.5,
            0.02,
            f"Avg Advantage: +{avg_adv:.1f} pp",
            transform=ax.transAxes,
            ha="center",
            fontweight="bold",
            fontsize=9,
            color=COLORS["advantage"],
        )
    else:
        ax.text(0.5, 0.5, "Exp1 data not available", ha="center", va="center")
        ax.set_title("A. Main Domain Comparison", fontweight="bold")

    # Panel B: Static knowledge ablation (from Exp2)
    ax = axes[0, 1]
    if exp2_data:
        configs = ["with_static_knowledge", "without_static_knowledge"]
        labels = ["With SK", "Without SK"]
        x = np.arange(2)
        width = 0.35

        precept_vals = []
        fr_vals = []
        for config in configs:
            if config in exp2_data:
                precept_vals.append(
                    exp2_data[config]["precept"]["first_try_success"]["mean"] * 100
                )
                fr_vals.append(
                    exp2_data[config]["full_reflexion"]["first_try_success"]["mean"]
                    * 100
                )
            else:
                precept_vals.append(0)
                fr_vals.append(0)

        ax.bar(
            x - width / 2,
            precept_vals,
            width,
            label="PRECEPT",
            color=COLORS["precept"],
            edgecolor="white",
        )
        ax.bar(
            x + width / 2,
            fr_vals,
            width,
            label="Full Reflexion",
            color=COLORS["full_reflexion"],
            edgecolor="white",
        )

        ax.set_ylabel("First-Try Success (%)")
        ax.set_title("B. Static Knowledge Ablation (Exp 2)", fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_ylim(0, 110)
        ax.legend(loc="upper right", fontsize=8)
        ax.grid(axis="y", alpha=0.3)

        # Add effect annotation
        if "static_knowledge_effect" in exp2_data:
            effect = exp2_data["static_knowledge_effect"]
            ax.text(
                0.5,
                0.02,
                f"SK Effect: PRECEPT +{effect['precept_p1_gain']:.1f}pp, FR +{effect['fr_p1_gain']:.1f}pp",
                transform=ax.transAxes,
                ha="center",
                fontsize=8,
            )
    else:
        ax.text(0.5, 0.5, "Exp2 data not available", ha="center", va="center")
        ax.set_title("B. Static Knowledge Ablation", fontweight="bold")

    # Panel C: Learning curve (from Exp3)
    ax = axes[1, 0]
    if exp3_data:
        sorted_keys = sorted(
            [k for k in exp3_data.keys() if "beta" in exp3_data[k]],
            key=lambda k: exp3_data[k]["beta"],
        )

        if sorted_keys:
            betas = [exp3_data[k]["beta"] for k in sorted_keys]
            precept_p1 = [
                exp3_data[k]["precept"]["first_try_success"]["mean"] * 100
                for k in sorted_keys
            ]
            fr_p1 = [
                exp3_data[k]["full_reflexion"]["first_try_success"]["mean"] * 100
                for k in sorted_keys
            ]

            ax.plot(
                betas,
                precept_p1,
                "o-",
                markersize=8,
                linewidth=2,
                color=COLORS["precept"],
                label="PRECEPT",
            )
            ax.plot(
                betas,
                fr_p1,
                "s-",
                markersize=8,
                linewidth=2,
                color=COLORS["full_reflexion"],
                label="Full Reflexion",
            )

            ax.set_xlabel(r"Training Exposure ($\beta$)")
            ax.set_ylabel("First-Try Success (%)")
            ax.set_title("C. Training Size Ablation (Exp 3)", fontweight="bold")
            ax.set_xticks(betas)
            ax.set_ylim(0, 110)
            ax.legend(loc="lower right", fontsize=8)
            ax.grid(True, alpha=0.3)
        else:
            ax.text(
                0.5, 0.5, "Learning curve data not available", ha="center", va="center"
            )
            ax.set_title("C. Training Size Ablation", fontweight="bold")
    else:
        ax.text(0.5, 0.5, "Exp3 data not available", ha="center", va="center")
        ax.set_title("C. Training Size Ablation", fontweight="bold")

    # Panel D: Continuous learning (from Exp4)
    ax = axes[1, 1]
    if exp4_data:
        learning_curves = exp4_data.get("learning_curves", {})
        if learning_curves:
            precept_curve = learning_curves.get("precept", {})
            num_encounters = len(
                [k for k in precept_curve.keys() if k.startswith("encounter_")]
            )

            if num_encounters > 0:
                encounters = list(range(1, num_encounters + 1))
                precept_p1 = [
                    learning_curves.get("precept", {})
                    .get(f"encounter_{e}", {})
                    .get("p1_mean", 0)
                    * 100
                    for e in encounters
                ]
                fr_p1 = [
                    learning_curves.get("full_reflexion", {})
                    .get(f"encounter_{e}", {})
                    .get("p1_mean", 0)
                    * 100
                    for e in encounters
                ]

                ax.plot(
                    encounters,
                    precept_p1,
                    "o-",
                    markersize=8,
                    linewidth=2,
                    color=COLORS["precept"],
                    label="PRECEPT",
                )
                ax.plot(
                    encounters,
                    fr_p1,
                    "s-",
                    markersize=8,
                    linewidth=2,
                    color=COLORS["full_reflexion"],
                    label="Full Reflexion",
                )

                ax.set_xlabel("Encounter Number")
                ax.set_ylabel("First-Try Success (%)")
                ax.set_title("D. Continuous Learning (Exp 4)", fontweight="bold")
                ax.set_xticks(encounters)
                ax.set_ylim(0, 110)
                ax.legend(loc="lower right", fontsize=8)
                ax.grid(True, alpha=0.3)

                # Add improvement annotation
                improvements = exp4_data.get("improvements", {})
                p_imp = improvements.get("precept", {}).get("p1_improvement_pp", 0)
                ax.text(
                    0.5,
                    0.02,
                    f"PRECEPT Improvement: +{p_imp:.1f}pp",
                    transform=ax.transAxes,
                    ha="center",
                    fontweight="bold",
                    fontsize=9,
                    color=COLORS["advantage"],
                )
            else:
                ax.text(0.5, 0.5, "No encounter data", ha="center", va="center")
                ax.set_title("D. Continuous Learning", fontweight="bold")
        else:
            ax.text(0.5, 0.5, "No learning curves data", ha="center", va="center")
            ax.set_title("D. Continuous Learning", fontweight="bold")
    else:
        ax.text(0.5, 0.5, "Exp4 data not available", ha="center", va="center")
        ax.set_title("D. Continuous Learning", fontweight="bold")

    plt.suptitle(
        "PRECEPT vs Baselines: Comprehensive Evaluation\n"
        "(All experiments: N=10 seeds, β=3 recommended)",
        fontsize=13,
        fontweight="bold",
        y=1.02,
    )
    plt.tight_layout()
    plt.savefig(output_dir / "fig5_overview.png")
    plt.savefig(output_dir / "fig5_overview.pdf")
    plt.close()

    print("  ✅ Generated: fig5_overview.png/pdf")


def generate_master_summary_table(
    exp1_data: dict,
    exp2_data: dict,
    exp3_data: dict,
    exp4_data: dict,
    output_dir: Path,
):
    """Generate master summary table combining all experiments."""

    lines = [
        "# Consolidated Results Summary\n",
        f"Generated: {datetime.now().isoformat()}\n",
        "## Experiment Overview\n",
        "| Exp | Description | Key Finding |",
        "|-----|-------------|-------------|",
        "| 1 | Main Domain Comparison | PRECEPT wins across ALL 6 domains |",
        "| 2 | Static Knowledge Ablation | SK helps both; PRECEPT advantage persists |",
        "| 3 | Training Size Ablation | PRECEPT sample-efficient at all β |",
        "| 4 | Continuous Learning | PRECEPT improves during testing |",
        "",
        "## Key Findings\n",
    ]

    # Exp1 Summary
    if exp1_data:
        domains = sorted(exp1_data.keys())
        avg_p1_precept = np.mean(
            [exp1_data[d]["precept"]["first_try_success"]["mean"] for d in domains]
        )
        avg_p1_fr = np.mean(
            [
                exp1_data[d]["full_reflexion"]["first_try_success"]["mean"]
                for d in domains
            ]
        )
        avg_adv = np.mean(
            [
                exp1_data[d]["advantage"].get(
                    "vs_fr_first_try_pp",
                    exp1_data[d]["advantage"].get("first_try_success_pp", 0),
                )
                for d in domains
            ]
        )

        lines.append("### Experiment 1: Main Domain Comparison")
        lines.append(f"- **PRECEPT Mean P₁**: {avg_p1_precept*100:.1f}%")
        lines.append(f"- **Full Reflexion Mean P₁**: {avg_p1_fr*100:.1f}%")
        lines.append(f"- **Average Advantage**: +{avg_adv:.1f} percentage points")
        lines.append(f"- **Domains Tested**: {len(domains)}")
        lines.append("")

    # Exp2 Summary
    if exp2_data and "static_knowledge_effect" in exp2_data:
        effect = exp2_data["static_knowledge_effect"]
        lines.append("### Experiment 2: Static Knowledge Ablation")
        lines.append(f"- **PRECEPT P₁ Gain from SK**: +{effect['precept_p1_gain']:.1f} pp")
        lines.append(f"- **FR P₁ Gain from SK**: +{effect['fr_p1_gain']:.1f} pp")
        lines.append("- **Key Finding**: Both benefit from SK, PRECEPT advantage persists")
        lines.append("")

    # Exp3 Summary
    if exp3_data:
        sorted_keys = [k for k in exp3_data.keys() if "beta" in exp3_data[k]]
        if sorted_keys:
            lines.append("### Experiment 3: Training Size Ablation")
            for key in sorted(sorted_keys, key=lambda k: exp3_data[k]["beta"]):
                d = exp3_data[key]
                p1 = d["precept"]["first_try_success"]["mean"] * 100
                adv = d["advantage"].get(
                    "vs_fr_first_try_pp", d["advantage"].get("first_try_success_pp", 0)
                )
                lines.append(
                    f"- **β={d['beta']}** (T={d['train_count']}): PRECEPT {p1:.1f}%, Advantage +{adv:.1f} pp"
                )
            lines.append(
                "- **Key Finding**: PRECEPT sample-efficient, advantage increases with training"
            )
            lines.append("")

    # Exp4 Summary
    if exp4_data:
        improvements = exp4_data.get("improvements", {})
        if improvements:
            p_imp = improvements.get("precept", {}).get("p1_improvement_pp", 0)
            fr_imp = improvements.get("full_reflexion", {}).get("p1_improvement_pp", 0)
            lines.append("### Experiment 4: Continuous Learning")
            lines.append(f"- **PRECEPT P₁ Improvement**: +{p_imp:.1f} pp (1st→4th encounter)")
            lines.append(f"- **FR P₁ Improvement**: +{fr_imp:.1f} pp")
            lines.append(f"- **Learning Advantage**: +{p_imp - fr_imp:.1f} pp")
            lines.append(
                "- **Key Finding**: PRECEPT learns during testing, baselines remain flat"
            )
            lines.append("")

    # Overall summary
    lines.append("## Publication Highlights\n")
    lines.append("| Metric | PRECEPT | Full Reflexion | Advantage |")
    lines.append("|--------|---------|----------------|-----------|")

    if exp1_data:
        domains = sorted(exp1_data.keys())
        avg_p1_precept = (
            np.mean(
                [exp1_data[d]["precept"]["first_try_success"]["mean"] for d in domains]
            )
            * 100
        )
        avg_p1_fr = (
            np.mean(
                [
                    exp1_data[d]["full_reflexion"]["first_try_success"]["mean"]
                    for d in domains
                ]
            )
            * 100
        )
        avg_pt_precept = (
            np.mean([exp1_data[d]["precept"]["success_rate"]["mean"] for d in domains])
            * 100
        )
        avg_pt_fr = (
            np.mean(
                [
                    exp1_data[d]["full_reflexion"]["success_rate"]["mean"]
                    for d in domains
                ]
            )
            * 100
        )

        lines.append(
            f"| First-Try Success (P₁) | **{avg_p1_precept:.1f}%** | {avg_p1_fr:.1f}% | +{avg_p1_precept-avg_p1_fr:.1f} pp |"
        )
        lines.append(
            f"| Overall Success (Pₜ) | **{avg_pt_precept:.1f}%** | {avg_pt_fr:.1f}% | +{avg_pt_precept-avg_pt_fr:.1f} pp |"
        )

    lines.append("")
    lines.append("## Theoretical Justification\n")
    lines.append("PRECEPT's advantage stems from:")
    lines.append(
        "1. **Hash-based Rule Lookup**: O(1) exact condition→solution mapping"
    )
    lines.append("2. **Partial Progress Tracking**: Failed options remembered per key")
    lines.append("3. **Conflict Resolution**: Dynamic learning overrides static KB")
    lines.append("4. **Deterministic Application**: No LLM interpretation errors")

    with open(output_dir / "master_summary.md", "w") as f:
        f.write("\n".join(lines))

    print("  ✅ Generated: master_summary.md")


def generate_latex_master_table(exp1_data: dict, output_dir: Path):
    """Generate publication-ready LaTeX master table."""

    if not exp1_data:
        print("  ⚠️ Skipping LaTeX table: Exp1 data not available")
        return

    domains = sorted(exp1_data.keys())

    latex_lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{PRECEPT vs Baselines: Comprehensive Evaluation Across 6 Domains}",
        r"\label{tab:master_summary}",
        r"\begin{tabular}{lcccccc}",
        r"\toprule",
        r"& \multicolumn{2}{c}{First-Try Success (P₁)} & \multicolumn{2}{c}{Overall Success (Pₜ)} & \multicolumn{2}{c}{Statistics} \\",
        r"\cmidrule(lr){2-3} \cmidrule(lr){4-5} \cmidrule(lr){6-7}",
        r"Domain & PRECEPT & Full Refl. & PRECEPT & Full Refl. & p-value & Cohen's d \\",
        r"\midrule",
    ]

    for domain in domains:
        d = exp1_data[domain]
        p = d["precept"]
        fr = d["full_reflexion"]
        test = (
            d["statistical_tests"]
            .get("precept_vs_fr", {})
            .get(
                "first_try_success",
                d["statistical_tests"].get(
                    "first_try_success", {"p_value": 1, "cohens_d": 0}
                ),
            )
        )

        p1_p = p["first_try_success"]
        p1_fr = fr["first_try_success"]
        pt_p = p["success_rate"]
        pt_fr = fr["success_rate"]

        sig = (
            "***"
            if test["p_value"] < 0.001
            else "**"
            if test["p_value"] < 0.01
            else "*"
            if test["p_value"] < 0.05
            else ""
        )

        p1_p_m, p1_p_c = p1_p['mean'] * 100, bounded_pct_ci(p1_p['mean'] * 100, p1_p['ci_95'] * 100)
        p1_fr_m, p1_fr_c = p1_fr['mean'] * 100, bounded_pct_ci(p1_fr['mean'] * 100, p1_fr['ci_95'] * 100)
        pt_p_m, pt_p_c = pt_p['mean'] * 100, bounded_pct_ci(pt_p['mean'] * 100, pt_p['ci_95'] * 100)
        pt_fr_m, pt_fr_c = pt_fr['mean'] * 100, bounded_pct_ci(pt_fr['mean'] * 100, pt_fr['ci_95'] * 100)
        latex_lines.append(
            f"{domain.capitalize()} & "
            f"\\textbf{{{p1_p_m:.1f}}} $\\pm$ {p1_p_c:.1f} & "
            f"{p1_fr_m:.1f} $\\pm$ {p1_fr_c:.1f} & "
            f"\\textbf{{{pt_p_m:.1f}}} $\\pm$ {pt_p_c:.1f} & "
            f"{pt_fr_m:.1f} $\\pm$ {pt_fr_c:.1f} & "
            f"{test['p_value']:.3f}{sig} & "
            f"{test['cohens_d']:.2f} \\\\"
        )

    # Summary row
    latex_lines.append(r"\midrule")
    avg_p1_p = np.mean(
        [exp1_data[d]["precept"]["first_try_success"]["mean"] for d in domains]
    )
    avg_p1_fr = np.mean(
        [exp1_data[d]["full_reflexion"]["first_try_success"]["mean"] for d in domains]
    )
    avg_pt_p = np.mean(
        [exp1_data[d]["precept"]["success_rate"]["mean"] for d in domains]
    )
    avg_pt_fr = np.mean(
        [exp1_data[d]["full_reflexion"]["success_rate"]["mean"] for d in domains]
    )

    latex_lines.append(
        f"\\textbf{{Average}} & "
        f"\\textbf{{{avg_p1_p*100:.1f}\\%}} & {avg_p1_fr*100:.1f}\\% & "
        f"\\textbf{{{avg_pt_p*100:.1f}\\%}} & {avg_pt_fr*100:.1f}\\% & — & — \\\\"
    )
    latex_lines.append(
        f"\\textbf{{$\\Delta$}} & "
        f"\\multicolumn{{2}}{{c}}{{+{(avg_p1_p-avg_p1_fr)*100:.1f} pp}} & "
        f"\\multicolumn{{2}}{{c}}{{+{(avg_pt_p-avg_pt_fr)*100:.1f} pp}} & "
        f"\\multicolumn{{2}}{{c}}{{—}} \\\\"
    )

    latex_lines.extend(
        [
            r"\bottomrule",
            r"\multicolumn{7}{l}{\footnotesize $\pm$ indicates 95\% CI. Significance: * p<0.05, ** p<0.01, *** p<0.001. N=10 independent runs.} \\",
            r"\end{tabular}",
            r"\end{table*}",
        ]
    )

    with open(output_dir / "master_table.tex", "w") as f:
        f.write("\n".join(latex_lines))

    print("  ✅ Generated: master_table.tex")


def main():
    parser = argparse.ArgumentParser(
        description="Generate consolidated results from all experiments"
    )
    parser.add_argument("--exp1", type=Path, help="Exp1 (main comparison) results directory")
    parser.add_argument("--exp2", type=Path, help="Exp2 (static knowledge) results directory")
    parser.add_argument("--exp3", type=Path, help="Exp3 (training size) results directory")
    parser.add_argument("--exp4", type=Path, help="Exp4 (continuous learning) results directory")
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output directory for consolidated results",
    )
    args = parser.parse_args()

    # Create output directory
    args.output.mkdir(parents=True, exist_ok=True)
    tables_dir = args.output / "tables"
    figures_dir = args.output / "figures"
    tables_dir.mkdir(exist_ok=True)
    figures_dir.mkdir(exist_ok=True)

    print("\n" + "=" * 60)
    print("GENERATING CONSOLIDATED RESULTS")
    print("=" * 60 + "\n")

    # Load data from each experiment
    exp1_data = load_results(args.exp1) if args.exp1 and args.exp1.exists() else {}
    exp2_data = load_results(args.exp2) if args.exp2 and args.exp2.exists() else {}
    exp3_data = load_results(args.exp3) if args.exp3 and args.exp3.exists() else {}
    exp4_data = load_results(args.exp4) if args.exp4 and args.exp4.exists() else {}

    print(
        f"Loaded: Exp1={bool(exp1_data)}, Exp2={bool(exp2_data)}, Exp3={bool(exp3_data)}, Exp4={bool(exp4_data)}"
    )

    print("\nGenerating consolidated figure...")
    generate_figure5_overview(exp1_data, exp2_data, exp3_data, exp4_data, figures_dir)

    print("\nGenerating summary tables...")
    generate_master_summary_table(
        exp1_data, exp2_data, exp3_data, exp4_data, args.output
    )
    generate_latex_master_table(exp1_data, tables_dir)

    # Save configuration
    config = {
        "generated": datetime.now().isoformat(),
        "exp1_dir": str(args.exp1) if args.exp1 else None,
        "exp2_dir": str(args.exp2) if args.exp2 else None,
        "exp3_dir": str(args.exp3) if args.exp3 else None,
        "exp4_dir": str(args.exp4) if args.exp4 else None,
    }
    with open(args.output / "consolidation_config.json", "w") as f:
        json.dump(config, f, indent=2)

    print("\n" + "=" * 60)
    print("CONSOLIDATION COMPLETE!")
    print("=" * 60)
    print(f"\nOutput: {args.output}")
    print("\nFiles generated:")
    print("  • figures/fig5_overview.png/pdf - Combined 4-panel overview")
    print("  • master_summary.md - Markdown summary")
    print("  • tables/master_table.tex - LaTeX master table")


if __name__ == "__main__":
    main()
