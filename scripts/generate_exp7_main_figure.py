#!/usr/bin/env python3
"""
Generate Publication-Quality Figures for Experiment 7: Rule Drift / Non-Stationary CSPs

Creates highest-quality figures with:
- 95% Confidence Intervals as error bars
- Standard deviation annotations
- 600 DPI resolution
- Vector PDF output
- Proper LaTeX math rendering

Output: PDF and PNG files suitable for top-tier publication

Usage:
    uv run scripts/generate_exp7_main_figure.py [results_directory]
"""

import json
import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

try:
    from scripts.utils.pct_bounds import bounded_pct_ci
except ImportError:
    def bounded_pct_ci(mean_pct, ci_pct, lower=0.0, upper=100.0):
        return max(0.0, min(ci_pct, mean_pct - lower, upper - mean_pct))

# Use LaTeX-style rendering for highest quality
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

MARKERS = {
    'precept': 'o',
    'full_reflexion': 's',
    'expel': '^',
}

LABELS = {
    'precept': 'PRECEPT',
    'full_reflexion': 'Full Reflexion',
    'expel': 'ExpeL',
}

HATCHES = {'precept': None, 'full_reflexion': '//', 'expel': '\\\\'}
AGENTS = ['precept', 'full_reflexion', 'expel']


def find_latest_results_dir() -> Optional[Path]:
    """Find the most recent exp7 results directory."""
    pub_results = Path(__file__).parent.parent / "data" / "publication_results"
    exp7_dirs = sorted(pub_results.glob("exp7_rule_drift_*"), reverse=True)
    return exp7_dirs[0] if exp7_dirs else None


def parse_test_log_for_encounters(log_path: Path, encounters_per_key: int = 4) -> Dict[str, Dict]:
    """Parse a test log file to extract per-encounter P₁ for each agent.
    
    Returns dict with agent -> encounter_X -> {p1, pt, steps, n}
    """
    if not log_path.exists():
        return {}
    
    with open(log_path) as f:
        content = f.read()
    
    # Parse all test result lines
    pattern = re.compile(
        r"📊 \[(?:MATCHED|RANDOM) Test (\d+)/\d+\] "
        r"key=([^|]+?) \| "
        r"PRECEPT: ([✓✗]) \(P₁=([YN]), steps=(\d+)\) \| "
        r"ExpeL: ([✓✗]) \(P₁=([YN])\) \| "
        r"FullRef: ([✓✗]) \(P₁=([YN])\)"
    )
    
    # Group results by agent and condition key
    results_by_agent = {
        'precept': defaultdict(list),
        'expel': defaultdict(list),
        'full_reflexion': defaultdict(list),
    }
    
    for match in pattern.finditer(content):
        condition_key = match.group(2).strip()
        if condition_key.endswith('...'):
            condition_key = condition_key[:-3].strip()
        
        # PRECEPT
        results_by_agent['precept'][condition_key].append({
            'success': match.group(3) == '✓',
            'first_try': match.group(4) == 'Y',
            'steps': int(match.group(5)),
        })
        
        # ExpeL
        results_by_agent['expel'][condition_key].append({
            'success': match.group(6) == '✓',
            'first_try': match.group(7) == 'Y',
            'steps': 2 if match.group(7) == 'Y' else (4 if match.group(6) == '✓' else 6),
        })
        
        # Full Reflexion
        results_by_agent['full_reflexion'][condition_key].append({
            'success': match.group(8) == '✓',
            'first_try': match.group(9) == 'Y',
            'steps': 2 if match.group(9) == 'Y' else (4 if match.group(8) == '✓' else 6),
        })
    
    # Convert to encounter metrics
    encounter_metrics = {}
    for agent, by_key in results_by_agent.items():
        metrics_by_enc = defaultdict(lambda: {'first_try': 0, 'success': 0, 'steps': [], 'total': 0})
        
        for key, encounters in by_key.items():
            for enc_num, result in enumerate(encounters[:encounters_per_key]):
                enc_key = enc_num + 1
                metrics_by_enc[enc_key]['total'] += 1
                metrics_by_enc[enc_key]['steps'].append(result['steps'])
                if result['success']:
                    metrics_by_enc[enc_key]['success'] += 1
                if result['first_try']:
                    metrics_by_enc[enc_key]['first_try'] += 1
        
        encounter_metrics[agent] = {}
        for enc in range(1, encounters_per_key + 1):
            m = metrics_by_enc[enc]
            total = m['total']
            if total > 0:
                encounter_metrics[agent][f'encounter_{enc}'] = {
                    'p1': m['first_try'] / total,
                    'pt': m['success'] / total,
                    'avg_steps': sum(m['steps']) / len(m['steps']) if m['steps'] else 0,
                    'n': total,
                }
            else:
                encounter_metrics[agent][f'encounter_{enc}'] = {'p1': 0, 'pt': 0, 'avg_steps': 0, 'n': 0}
    
    return encounter_metrics


def load_and_reparse_results(results_dir: Path) -> Dict[str, Any]:
    """Load results and re-parse test logs to get accurate encounter data."""
    
    # Load main results file
    results_file = results_dir / "rule_drift_results.json"
    if not results_file.exists():
        print(f"❌ Results file not found: {results_file}")
        return {}
    
    with open(results_file) as f:
        data = json.load(f)
    
    # Re-parse all test logs to get accurate per-encounter data
    print("  📂 Re-parsing test logs for accurate encounter data...")
    
    seeds = data.get('seeds_used', [])
    domain = data.get('domain', 'logistics')
    encounters_per_key = data.get('parameters', {}).get('encounters_per_key', 4)
    
    # Aggregate encounter metrics across seeds
    all_p1_by_agent_enc = {
        'precept': defaultdict(list),
        'expel': defaultdict(list),
        'full_reflexion': defaultdict(list),
    }
    
    for seed in seeds:
        log_path = results_dir / f"{domain}_seed{seed}_test.log"
        enc_metrics = parse_test_log_for_encounters(log_path, encounters_per_key)
        
        for agent in ['precept', 'expel', 'full_reflexion']:
            agent_enc = enc_metrics.get(agent, {})
            for enc in range(1, encounters_per_key + 1):
                enc_key = f'encounter_{enc}'
                enc_data = agent_enc.get(enc_key, {})
                if enc_data.get('n', 0) > 0:
                    all_p1_by_agent_enc[agent][enc].append(enc_data['p1'])
    
    # Compute aggregated stats
    learning_curves = {}
    for agent in ['precept', 'expel', 'full_reflexion']:
        learning_curves[agent] = {}
        for enc in range(1, encounters_per_key + 1):
            p1_values = all_p1_by_agent_enc[agent][enc]
            if p1_values:
                mean = np.mean(p1_values)
                std = np.std(p1_values, ddof=1) if len(p1_values) > 1 else 0
                n = len(p1_values)
                # 95% CI using t-distribution
                t_val = stats.t.ppf(0.975, n - 1) if n > 1 else 1.96
                ci = t_val * (std / np.sqrt(n)) if n > 0 else 0
                
                learning_curves[agent][f'encounter_{enc}'] = {
                    'p1_mean': mean,
                    'p1_std': std,
                    'p1_ci_95': ci,
                    'n': n,
                }
            else:
                learning_curves[agent][f'encounter_{enc}'] = {
                    'p1_mean': 0, 'p1_std': 0, 'p1_ci_95': 0, 'n': 0
                }
    
    # Update data with corrected learning curves
    data['learning_curves_corrected'] = learning_curves
    
    return data


def compute_ci(mean: float, std: float, n: int, confidence: float = 0.95) -> float:
    """Compute confidence interval."""
    if n <= 1:
        return 0
    t_val = stats.t.ppf((1 + confidence) / 2, n - 1)
    se = std / math.sqrt(n)
    return t_val * se


def create_recovery_curve_figure(data: Dict, output_dir: Path):
    """Create Figure 7a: Recovery Curve (P₁ by Encounter)."""
    print("  📊 Creating Recovery Curve figure...")
    
    # Use corrected learning curves if available, else original
    learning_curves = data.get('learning_curves_corrected', data.get('learning_curves', {}))
    
    if not learning_curves:
        print("    ⚠️ No learning curves data found")
        return
    
    # Get number of encounters
    precept_curve = learning_curves.get('precept', {})
    num_encounters = len([k for k in precept_curve.keys() if k.startswith('encounter_')])
    
    if num_encounters == 0:
        print("    ⚠️ No encounter data found")
        return
    
    encounters = list(range(1, num_encounters + 1))
    encounter_labels = ['1st', '2nd', '3rd', '4th', '5th', '6th'][:num_encounters]
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Plot each agent
    for agent in ['precept', 'expel', 'full_reflexion']:
        agent_curve = learning_curves.get(agent, {})
        p1_means = []
        p1_cis = []
        
        for enc in encounters:
            enc_key = f'encounter_{enc}'
            enc_data = agent_curve.get(enc_key, {})
            p1_means.append(enc_data.get('p1_mean', 0) * 100)
            p1_cis.append(enc_data.get('p1_ci_95', 0) * 100)
        
        lower = [max(0.0, min(ci, m)) for m, ci in zip(p1_means, p1_cis)]
        upper = [max(0.0, min(ci, 100.0 - m)) for m, ci in zip(p1_means, p1_cis)]
        ax.errorbar(
            encounters, p1_means, yerr=[lower, upper],
            marker=MARKERS[agent], color=COLORS[agent],
            label=LABELS[agent], linewidth=2.5, markersize=10,
            capsize=5, capthick=2
        )
    
    ax.set_xlabel('Encounter Number', fontsize=14, fontweight='bold')
    ax.set_ylabel('First-Try Success Rate P₁ (%)', fontsize=14, fontweight='bold')
    ax.set_title('Rule Drift Recovery: P₁ by Encounter\n(Hash Seed Changed at Test Time)', 
                 fontsize=15, fontweight='bold')
    ax.set_xticks(encounters)
    ax.set_xticklabels(encounter_labels)
    ax.set_ylim(-5, 100)
    ax.legend(loc='lower right', fontsize=12)
    ax.grid(True, alpha=0.3, linestyle='--')
    
    # Add statistical significance markers
    stats_tests = data.get('statistical_tests', {})
    per_enc_tests = stats_tests.get('per_encounter_tests', {})
    
    for enc in encounters:
        enc_key = f'encounter_{enc}'
        enc_tests = per_enc_tests.get(enc_key, {})
        
        # Get PRECEPT mean for annotation position
        precept_p1 = learning_curves.get('precept', {}).get(enc_key, {}).get('p1_mean', 0) * 100
        
        # Check significance vs both baselines
        vs_expel = enc_tests.get('precept_vs_expel', {})
        vs_fr = enc_tests.get('precept_vs_full_reflexion', {})
        
        p_expel = vs_expel.get('p_value', 1.0)
        p_fr = vs_fr.get('p_value', 1.0)
        
        # Use minimum p-value for significance marker
        min_p = min(p_expel, p_fr)
        if min_p < 0.001:
            sig = '***'
        elif min_p < 0.01:
            sig = '**'
        elif min_p < 0.05:
            sig = '*'
        else:
            sig = ''
        
        if sig:
            ax.annotate(sig, xy=(enc, precept_p1 + 5), ha='center', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    
    output_path = output_dir / "Figure_Exp7_RecoveryCurve.png"
    plt.savefig(output_path, dpi=600, bbox_inches='tight')
    plt.savefig(output_path.with_suffix('.pdf'), bbox_inches='tight')
    print(f"    ✓ Saved: {output_path.name}")
    plt.close()


def create_improvement_comparison(data: Dict, output_dir: Path):
    """Create Figure 7b: Improvement Comparison (1st → last encounter)."""
    print("  📊 Creating Improvement Comparison figure...")
    
    learning_curves = data.get('learning_curves_corrected', data.get('learning_curves', {}))
    
    if not learning_curves:
        print("    ⚠️ No learning curves data found")
        return
    
    # Calculate improvements
    improvements = {}
    for agent in ['precept', 'expel', 'full_reflexion']:
        agent_curve = learning_curves.get(agent, {})
        enc1 = agent_curve.get('encounter_1', {}).get('p1_mean', 0) * 100
        enc4 = agent_curve.get('encounter_4', {}).get('p1_mean', 0) * 100
        improvements[agent] = enc4 - enc1
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    agents = AGENTS
    x_pos = np.arange(len(agents))
    imp_values = [improvements[a] for a in agents]

    bars = []
    for i, agent in enumerate(agents):
        bar = ax.bar(x_pos[i], imp_values[i], color=COLORS[agent],
                     edgecolor='black', linewidth=0.8, width=0.6,
                     hatch=HATCHES[agent])
        bars.append(bar[0])

    # Add value labels on bars
    for bar, val in zip(bars, imp_values):
        height = bar.get_height()
        sign = '+' if val >= 0 else ''
        ax.annotate(
            f'{sign}{val:.1f}pp',
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 5),
            textcoords='offset points',
            ha='center', va='bottom',
            fontsize=13, fontweight='bold'
        )

    # Add significance markers
    stats_tests = data.get('statistical_tests', {})
    improvement_tests = stats_tests.get('improvement_tests', {})

    for i, agent in enumerate(agents):
        agent_stats = improvement_tests.get(agent, {})
        p_val = agent_stats.get('improvement_significance', {}).get('p_value_one_tailed', 1.0)
        if p_val < 0.001:
            sig = '***'
        elif p_val < 0.01:
            sig = '**'
        elif p_val < 0.05:
            sig = '*'
        else:
            sig = 'n.s.'

        ax.annotate(
            sig,
            xy=(x_pos[i], imp_values[i] + 8),
            ha='center', va='bottom',
            fontsize=11, fontweight='bold',
            color='green' if sig != 'n.s.' else 'gray'
        )

    ax.set_xticks(x_pos)
    ax.set_xticklabels([LABELS[a] for a in agents], fontsize=11)
    ax.set_ylabel(r'$P_1$ Improvement (pp)', fontweight='bold')
    ax.set_title('Recovery Improvement: 1st → 4th Encounter', fontweight='bold')
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax.set_ylim(min(0, min(imp_values) - 10), max(imp_values) + 20)
    ax.yaxis.grid(True, alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)
    
    plt.tight_layout()
    
    output_path = output_dir / "Figure_Exp7_Improvement.png"
    plt.savefig(output_path, dpi=600, bbox_inches='tight')
    plt.savefig(output_path.with_suffix('.pdf'), bbox_inches='tight')
    print(f"    ✓ Saved: {output_path.name}")
    plt.close()


def create_combined_figure(data: Dict, output_dir: Path):
    """Create combined Figure 7: Recovery Curve + Improvement."""
    print("  📊 Creating Combined Main Figure...")
    
    learning_curves = data.get('learning_curves_corrected', data.get('learning_curves', {}))
    
    if not learning_curves:
        print("    ⚠️ No learning curves data found")
        return
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))
    
    # --- Panel A: Recovery Curve ---
    precept_curve = learning_curves.get('precept', {})
    num_encounters = len([k for k in precept_curve.keys() if k.startswith('encounter_')])
    encounters = list(range(1, num_encounters + 1))
    encounter_labels = ['1st', '2nd', '3rd', '4th'][:num_encounters]
    
    for agent in ['precept', 'expel', 'full_reflexion']:
        agent_curve = learning_curves.get(agent, {})
        p1_means = []
        p1_cis = []
        
        for enc in encounters:
            enc_key = f'encounter_{enc}'
            enc_data = agent_curve.get(enc_key, {})
            p1_means.append(enc_data.get('p1_mean', 0) * 100)
            p1_cis.append(enc_data.get('p1_ci_95', 0) * 100)
        
        lower = [max(0.0, min(ci, m)) for m, ci in zip(p1_means, p1_cis)]
        upper = [max(0.0, min(ci, 100.0 - m)) for m, ci in zip(p1_means, p1_cis)]
        ax1.errorbar(
            encounters, p1_means, yerr=[lower, upper],
            marker=MARKERS[agent], color=COLORS[agent],
            label=LABELS[agent], linewidth=2.5, markersize=10,
            capsize=5, capthick=2
        )
    
    ax1.set_xlabel('Encounter Number', fontsize=13, fontweight='bold')
    ax1.set_ylabel('First-Try Success Rate P₁ (%)', fontsize=13, fontweight='bold')
    ax1.set_title('(a) Recovery Curve: P₁ by Encounter', fontsize=14, fontweight='bold')
    ax1.set_xticks(encounters)
    ax1.set_xticklabels(encounter_labels)
    ax1.set_ylim(-5, 100)
    ax1.legend(loc='lower right', fontsize=11)
    ax1.grid(True, alpha=0.3, linestyle='--')
    
    # Add significance annotations
    stats_tests = data.get('statistical_tests', {})
    per_enc_tests = stats_tests.get('per_encounter_tests', {})
    
    for enc in encounters:
        enc_key = f'encounter_{enc}'
        enc_tests = per_enc_tests.get(enc_key, {})
        precept_p1 = learning_curves.get('precept', {}).get(enc_key, {}).get('p1_mean', 0) * 100
        
        vs_expel = enc_tests.get('precept_vs_expel', {})
        vs_fr = enc_tests.get('precept_vs_full_reflexion', {})
        min_p = min(vs_expel.get('p_value', 1.0), vs_fr.get('p_value', 1.0))
        
        if min_p < 0.001:
            sig = '***'
        elif min_p < 0.01:
            sig = '**'
        elif min_p < 0.05:
            sig = '*'
        else:
            sig = ''
        
        if sig:
            ax1.annotate(sig, xy=(enc, precept_p1 + 5), ha='center', fontsize=11, fontweight='bold')
    
    # --- Panel B: Improvement Comparison ---
    improvements = {}
    for agent in AGENTS:
        agent_curve = learning_curves.get(agent, {})
        enc1 = agent_curve.get('encounter_1', {}).get('p1_mean', 0) * 100
        enc4 = agent_curve.get('encounter_4', {}).get('p1_mean', 0) * 100
        improvements[agent] = enc4 - enc1

    x_pos = np.arange(len(AGENTS))
    imp_values = [improvements[a] for a in AGENTS]

    bars = []
    for i, agent in enumerate(AGENTS):
        bar = ax2.bar(x_pos[i], imp_values[i], color=COLORS[agent],
                      edgecolor='black', linewidth=0.8, width=0.6,
                      hatch=HATCHES[agent])
        bars.append(bar[0])

    for bar, val in zip(bars, imp_values):
        height = bar.get_height()
        sign = '+' if val >= 0 else ''
        ax2.annotate(
            f'{sign}{val:.1f}pp',
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 5),
            textcoords='offset points',
            ha='center', va='bottom',
            fontsize=12, fontweight='bold'
        )

    # Add significance markers
    improvement_tests = stats_tests.get('improvement_tests', {})
    for i, agent in enumerate(AGENTS):
        agent_stats = improvement_tests.get(agent, {})
        p_val = agent_stats.get('improvement_significance', {}).get('p_value_one_tailed', 1.0)
        sig = '***' if p_val < 0.001 else '**' if p_val < 0.01 else '*' if p_val < 0.05 else 'n.s.'

        ax2.annotate(
            sig,
            xy=(x_pos[i], imp_values[i] + 7),
            ha='center', fontsize=11, fontweight='bold',
            color='green' if sig != 'n.s.' else 'gray'
        )

    ax2.set_xticks(x_pos)
    ax2.set_xticklabels([LABELS[a] for a in AGENTS], fontsize=11)
    ax2.set_ylabel(r'$P_1$ Improvement (pp)', fontweight='bold')
    ax2.set_title('(b) Recovery: 1st → 4th Encounter', fontweight='bold')
    ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax2.set_ylim(min(0, min(imp_values) - 10), max(imp_values) + 18)
    ax2.yaxis.grid(True, alpha=0.3, linestyle='--')
    ax2.set_axisbelow(True)
    
    plt.tight_layout()
    
    output_path = output_dir / "Figure_Exp7_Main.png"
    plt.savefig(output_path, dpi=600, bbox_inches='tight')
    plt.savefig(output_path.with_suffix('.pdf'), bbox_inches='tight')
    print(f"    ✓ Saved: {output_path.name}")
    plt.close()


def generate_latex_table(data: Dict, output_dir: Path):
    """Generate publication-quality LaTeX table."""
    print("  📊 Generating LaTeX table...")
    
    learning_curves = data.get('learning_curves_corrected', data.get('learning_curves', {}))
    stats_tests = data.get('statistical_tests', {})
    
    latex = r"""\begin{table}[htbp]
\centering
\caption{Experiment 7: Rule Drift Recovery Statistics (N=10 seeds)}
\label{tab:exp7_rule_drift}
\begin{tabular}{@{}lccc@{}}
\toprule
\textbf{Metric} & \textbf{PRECEPT} & \textbf{ExpeL} & \textbf{Full Reflexion} \\
\midrule
\multicolumn{4}{@{}l}{\textit{P$_1$ by Encounter (\%)}} \\
"""
    
    enc_labels = ['1st', '2nd', '3rd', '4th']
    for enc in range(1, 5):
        enc_key = f'encounter_{enc}'
        latex += f"\\quad {enc_labels[enc-1]} Encounter"
        
        for agent in ['precept', 'expel', 'full_reflexion']:
            agent_curve = learning_curves.get(agent, {})
            enc_data = agent_curve.get(enc_key, {})
            mean = enc_data.get('p1_mean', 0) * 100
            std = bounded_pct_ci(mean, enc_data.get('p1_std', 0) * 100)
            latex += f" & {mean:.1f} $\\pm$ {std:.1f}"
        
        latex += " \\\\\n"
    
    # Add improvement row
    latex += r"""\midrule
\multicolumn{4}{@{}l}{\textit{Recovery Metrics}} \\
\quad Improvement (1st$\rightarrow$4th)"""
    
    improvement_tests = stats_tests.get('improvement_tests', {})
    for agent in ['precept', 'expel', 'full_reflexion']:
        agent_curve = learning_curves.get(agent, {})
        enc1 = agent_curve.get('encounter_1', {}).get('p1_mean', 0) * 100
        enc4 = agent_curve.get('encounter_4', {}).get('p1_mean', 0) * 100
        imp = enc4 - enc1
        
        agent_stats = improvement_tests.get(agent, {})
        p_val = agent_stats.get('improvement_significance', {}).get('p_value_one_tailed', 1.0)
        if p_val < 0.001:
            sig = '$^{***}$'
        elif p_val < 0.01:
            sig = '$^{**}$'
        elif p_val < 0.05:
            sig = '$^{*}$'
        else:
            sig = ''
        
        sign = '+' if imp >= 0 else ''
        latex += f" & {sign}{imp:.1f}pp{sig}"
    
    latex += r""" \\
\bottomrule
\end{tabular}

\vspace{0.5em}
\footnotesize{$^{*}p<0.05$, $^{**}p<0.01$, $^{***}p<0.001$ (paired t-test, one-tailed).}
\end{table}
"""
    
    output_path = output_dir / "Table_Exp7_Statistics.tex"
    with open(output_path, 'w') as f:
        f.write(latex)
    print(f"    ✓ Saved: {output_path.name}")


def generate_analysis_report(data: Dict, output_dir: Path):
    """Generate comprehensive analysis report."""
    print("  📊 Generating analysis report...")
    
    learning_curves = data.get('learning_curves_corrected', data.get('learning_curves', {}))
    
    # Extract key metrics
    def get_p1(agent, enc):
        return learning_curves.get(agent, {}).get(f'encounter_{enc}', {}).get('p1_mean', 0) * 100
    
    report = f"""# Experiment 7: Rule Drift Analysis Report

## Executive Summary

This experiment tests agent adaptation when learned rules become **stale** due to 
environmental changes (simulated via PYTHONHASHSEED change: 0 → 1).

## Key Findings

### P₁ by Encounter

| Encounter | PRECEPT | ExpeL | Full Reflexion |
|-----------|---------|-------|----------------|
| 1st | {get_p1('precept', 1):.1f}% | {get_p1('expel', 1):.1f}% | {get_p1('full_reflexion', 1):.1f}% |
| 2nd | {get_p1('precept', 2):.1f}% | {get_p1('expel', 2):.1f}% | {get_p1('full_reflexion', 2):.1f}% |
| 3rd | {get_p1('precept', 3):.1f}% | {get_p1('expel', 3):.1f}% | {get_p1('full_reflexion', 3):.1f}% |
| 4th | {get_p1('precept', 4):.1f}% | {get_p1('expel', 4):.1f}% | {get_p1('full_reflexion', 4):.1f}% |

### Recovery Improvement (1st → 4th Encounter)

- **PRECEPT**: +{get_p1('precept', 4) - get_p1('precept', 1):.1f}pp
- **ExpeL**: +{get_p1('expel', 4) - get_p1('expel', 1):.1f}pp
- **Full Reflexion**: +{get_p1('full_reflexion', 4) - get_p1('full_reflexion', 1):.1f}pp

## Interpretation

### 1st Encounter Advantage (Cold Start)

PRECEPT shows {'advantage' if get_p1('precept', 1) > get_p1('expel', 1) else 'similar performance'} on 1st encounter:
- PRECEPT's previously learned rules provide partial guidance even after drift
- Baselines start fresh with no persistent memory

### Recovery Pattern

{f'Baselines recover faster because they have no stale priors to unlearn.' 
 if get_p1('expel', 4) - get_p1('expel', 1) > get_p1('precept', 4) - get_p1('precept', 1)
 else 'PRECEPT maintains steady improvement across encounters.'}

## Statistical Significance

"""
    
    stats_tests = data.get('statistical_tests', {})
    per_enc_tests = stats_tests.get('per_encounter_tests', {})
    
    for enc in range(1, 5):
        enc_key = f'encounter_{enc}'
        enc_tests = per_enc_tests.get(enc_key, {})
        vs_expel = enc_tests.get('precept_vs_expel', {})
        vs_fr = enc_tests.get('precept_vs_full_reflexion', {})
        
        report += f"""### Encounter {enc}
- PRECEPT vs ExpeL: p={vs_expel.get('p_value', 1.0):.4f}, d={vs_expel.get('cohens_d', 0):.2f}
- PRECEPT vs Full Reflexion: p={vs_fr.get('p_value', 1.0):.4f}, d={vs_fr.get('cohens_d', 0):.2f}

"""
    
    report += """## Files Generated

- `Figure_Exp7_Main.png/pdf`: Combined recovery curve and improvement comparison
- `Figure_Exp7_RecoveryCurve.png/pdf`: Recovery curve only
- `Figure_Exp7_Improvement.png/pdf`: Improvement comparison only
- `Table_Exp7_Statistics.tex`: LaTeX table for publication
"""
    
    output_path = output_dir / "Exp7_Analysis_Report.md"
    with open(output_path, 'w') as f:
        f.write(report)
    print(f"    ✓ Saved: {output_path.name}")


def find_all_domain_dirs() -> list:
    """Find all exp7 domain directories in publication_results (latest run per domain)."""
    pub_results = Path(__file__).parent.parent / "data" / "publication_results"
    dirs = sorted(pub_results.glob("exp7_rule_drift_*"))
    known_domains = {"integration", "logistics", "booking"}
    domain_dirs = {}
    for d in dirs:
        parts = d.name.replace("exp7_rule_drift_", "").split("_")
        domain = parts[0]
        if domain in known_domains:
            domain_dirs[domain] = d
    return list(domain_dirs.values())


def generate_combined_recovery_figure(all_data: list, output_dir: Path):
    """Generate combined figure: one column per domain showing recovery curves only."""
    n_domains = len(all_data)
    fig, axes = plt.subplots(1, n_domains, figsize=(7 * n_domains, 5.5))
    if n_domains == 1:
        axes = [axes]

    for col, (data, domain_name) in enumerate(all_data):
        learning_curves = data.get('learning_curves_corrected', data.get('learning_curves', {}))
        if not learning_curves:
            continue

        precept_curve = learning_curves.get('precept', {})
        num_encounters = len([k for k in precept_curve.keys() if k.startswith('encounter_')])
        encounters = list(range(1, num_encounters + 1))
        encounter_labels = ['1st', '2nd', '3rd', '4th'][:num_encounters]
        domain_cap = domain_name.capitalize()

        ax1 = axes[col]
        for agent in ['precept', 'expel', 'full_reflexion']:
            agent_curve = learning_curves.get(agent, {})
            p1_means = [agent_curve.get(f'encounter_{e}', {}).get('p1_mean', 0) * 100 for e in encounters]
            p1_cis = [agent_curve.get(f'encounter_{e}', {}).get('p1_ci_95', 0) * 100 for e in encounters]
            lower = [max(0.0, min(ci, m)) for m, ci in zip(p1_means, p1_cis)]
            upper = [max(0.0, min(ci, 100.0 - m)) for m, ci in zip(p1_means, p1_cis)]
            ax1.errorbar(
                encounters, p1_means, yerr=[lower, upper],
                marker=MARKERS[agent], color=COLORS[agent],
                label=LABELS[agent], linewidth=2.5, markersize=10,
                capsize=5, capthick=2,
            )

        # Significance markers
        stats_tests = data.get('statistical_tests', {})
        per_enc_tests = stats_tests.get('per_encounter_tests', {})
        for enc in encounters:
            enc_key = f'encounter_{enc}'
            enc_tests = per_enc_tests.get(enc_key, {})
            precept_p1 = learning_curves.get('precept', {}).get(enc_key, {}).get('p1_mean', 0) * 100
            vs_expel = enc_tests.get('precept_vs_expel', {})
            vs_fr = enc_tests.get('precept_vs_full_reflexion', {})
            min_p = min(vs_expel.get('p_value', 1.0), vs_fr.get('p_value', 1.0))
            sig = '***' if min_p < 0.001 else '**' if min_p < 0.01 else '*' if min_p < 0.05 else ''
            if sig:
                ax1.annotate(sig, xy=(enc, precept_p1 + 5), ha='center', fontsize=11, fontweight='bold')

        panel_letter = chr(ord('a') + col)
        ax1.set_xlabel('Encounter Number', fontsize=13, fontweight='bold')
        ax1.set_ylabel(r'$P_1$ (%)', fontsize=13, fontweight='bold')
        ax1.set_title(f'({panel_letter}) {domain_cap}: Recovery Curve', fontsize=14, fontweight='bold')
        ax1.set_xticks(encounters)
        ax1.set_xticklabels(encounter_labels)
        ax1.set_ylim(-5, 110)
        ax1.legend(loc='lower right', fontsize=11)
        ax1.grid(True, alpha=0.3, linestyle='--')

    plt.tight_layout(w_pad=3)
    output_path = output_dir / "Figure_Exp7_Combined.png"
    plt.savefig(output_path, dpi=600, bbox_inches='tight')
    plt.savefig(output_path.with_suffix('.pdf'), bbox_inches='tight')
    plt.close()
    print(f"  ✅ Generated: Figure_Exp7_Combined.png/pdf")


def generate_combined_bar_figure(all_data: list, output_dir: Path, filename: str = "Figure_Exp7_Drift_Bar.png"):
    """Generate combined bar plot figure for rule drift: grouped bars per encounter."""
    n_domains = len(all_data)
    fig, axes = plt.subplots(1, n_domains, figsize=(7 * n_domains, 5.5))
    if n_domains == 1:
        axes = [axes]

    agents_order = ['precept', 'expel', 'full_reflexion']
    bar_width = 0.22

    for col, (data, domain_name) in enumerate(all_data):
        learning_curves = data.get('learning_curves_corrected', data.get('learning_curves', {}))
        if not learning_curves:
            continue

        precept_curve = learning_curves.get('precept', {})
        num_encounters = len([k for k in precept_curve.keys() if k.startswith('encounter_')])
        encounters = list(range(1, num_encounters + 1))
        encounter_labels = ['1st', '2nd', '3rd', '4th'][:num_encounters]
        domain_cap = domain_name.capitalize()
        x = np.arange(num_encounters)

        ax = axes[col]
        for i, agent in enumerate(agents_order):
            agent_curve = learning_curves.get(agent, {})
            p1_means = [agent_curve.get(f'encounter_{e}', {}).get('p1_mean', 0) * 100 for e in encounters]
            p1_cis = [agent_curve.get(f'encounter_{e}', {}).get('p1_ci_95', 0) * 100 for e in encounters]
            lower = [max(0.0, min(ci, m)) for m, ci in zip(p1_means, p1_cis)]
            upper = [max(0.0, min(ci, 100.0 - m)) for m, ci in zip(p1_means, p1_cis)]
            offset = (i - 1) * bar_width
            bars = ax.bar(
                x + offset, p1_means, bar_width,
                yerr=[lower, upper], capsize=4,
                color=COLORS[agent], label=LABELS[agent],
                edgecolor='black', linewidth=0.8,
                hatch=HATCHES[agent], alpha=0.9,
                error_kw={'linewidth': 1.2, 'capthick': 1.2},
            )

        # Significance markers above PRECEPT bars
        stats_tests = data.get('statistical_tests', {})
        per_enc_tests = stats_tests.get('per_encounter_tests', {})
        for enc_idx, enc in enumerate(encounters):
            enc_key = f'encounter_{enc}'
            enc_tests = per_enc_tests.get(enc_key, {})
            precept_p1 = learning_curves.get('precept', {}).get(enc_key, {}).get('p1_mean', 0) * 100
            precept_ci = bounded_pct_ci(precept_p1, learning_curves.get('precept', {}).get(enc_key, {}).get('p1_ci_95', 0) * 100)
            vs_expel = enc_tests.get('precept_vs_expel', {})
            vs_fr = enc_tests.get('precept_vs_full_reflexion', {})
            min_p = min(vs_expel.get('p_value', 1.0), vs_fr.get('p_value', 1.0))
            sig = '***' if min_p < 0.001 else '**' if min_p < 0.01 else '*' if min_p < 0.05 else ''
            if sig:
                bar_x = enc_idx + (0 - 1) * bar_width  # PRECEPT bar position
                ax.annotate(
                    sig, xy=(bar_x, precept_p1 + precept_ci + 2),
                    ha='center', fontsize=10, fontweight='bold', color=COLORS['precept'],
                )

        panel_letter = chr(ord('a') + col)
        ax.set_xlabel('Encounter Number', fontsize=13, fontweight='bold')
        ax.set_ylabel(r'$P_1$ (%)', fontsize=13, fontweight='bold')
        ax.set_title(f'({panel_letter}) {domain_cap}: Drift Recovery', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(encounter_labels)
        ax.set_ylim(0, 115)
        ax.legend(loc='upper left', fontsize=11)
        ax.grid(True, alpha=0.3, linestyle='--', axis='y')

    plt.tight_layout(w_pad=3)
    output_path = output_dir / filename
    plt.savefig(output_path, dpi=600, bbox_inches='tight')
    plt.savefig(output_path.with_suffix('.pdf'), bbox_inches='tight')
    plt.close()
    print(f"  ✅ Generated: {filename} / .pdf")


def find_domain_dirs_by_date(date_prefix: str) -> list:
    """Find exp7 domain directories matching a specific date prefix (e.g. '20260213')."""
    pub_results = Path(__file__).parent.parent / "data" / "publication_results"
    dirs = sorted(pub_results.glob("exp7_rule_drift_*"))
    known_domains = {"integration", "logistics", "booking"}
    domain_dirs = {}
    for d in dirs:
        parts = d.name.replace("exp7_rule_drift_", "").split("_")
        domain = parts[0]
        if domain in known_domains and date_prefix in d.name:
            domain_dirs[domain] = d
    return list(domain_dirs.values())


def main():
    import argparse
    import sys

    parser = argparse.ArgumentParser(description="Generate Experiment 7 publication figures")
    parser.add_argument("results_dir", nargs="?", help="Single results directory")
    parser.add_argument("--combined", action="store_true", help="Generate combined multi-domain figures")
    parser.add_argument("--dirs", nargs="+", help="Explicit result directories to use")
    parser.add_argument("--bar", action="store_true", help="Generate bar plots instead of recovery curves")
    parser.add_argument("--output-dir", help="Override output directory name (under publication_results/)")
    parser.add_argument("--date", help="Filter directories by date prefix (e.g. 20260213)")
    args = parser.parse_args()

    if args.combined or args.dirs or args.date:
        # Determine which directories to use
        if args.dirs:
            domain_dirs = [Path(d) for d in args.dirs]
        elif args.date:
            domain_dirs = find_domain_dirs_by_date(args.date)
        else:
            domain_dirs = find_all_domain_dirs()

        if not domain_dirs:
            print("❌ No exp7 results found. Run the experiment first.")
            sys.exit(1)

        style_label = "Bar Plots" if args.bar else "Recovery Curves"
        print(f"\n📊 Generating Experiment 7 Combined Publication Results ({style_label})")
        print(f"   Found {len(domain_dirs)} domain(s)\n")

        all_data = []
        for d in domain_dirs:
            domain_name = d.name.replace("exp7_rule_drift_", "").split("_")[0]
            print(f"  Loading: {domain_name} ({d.name})")
            data = load_and_reparse_results(d)
            if data:
                all_data.append((data, domain_name))

                # Per-domain outputs
                figures_dir = d / "figures"
                figures_dir.mkdir(exist_ok=True)
                create_combined_figure(data, figures_dir)
                create_recovery_curve_figure(data, figures_dir)
                create_improvement_comparison(data, figures_dir)
                generate_latex_table(data, figures_dir)
                generate_analysis_report(data, figures_dir)

        if all_data:
            if args.output_dir:
                combined_dir = Path(__file__).parent.parent / "data" / "publication_results" / args.output_dir
            else:
                combined_dir = Path(__file__).parent.parent / "data" / "publication_results" / "exp7_combined"
            combined_dir.mkdir(exist_ok=True)
            print(f"\n  Generating combined figure ({len(all_data)} domains)...")
            if args.bar:
                generate_combined_bar_figure(all_data, combined_dir, "Figure_Exp7_Drift_Bar.png")
            else:
                generate_combined_recovery_figure(all_data, combined_dir)
            print(f"\n✅ Combined results saved to: {combined_dir}")
        return

    # Single directory mode (backward compatible)
    results_dir = Path(args.results_dir) if args.results_dir else None
    if results_dir is None:
        results_dir = find_latest_results_dir()
    if results_dir is None or not results_dir.exists():
        print(f"❌ Results directory not found: {results_dir}")
        sys.exit(1)
    
    print(f"\n📊 Generating Experiment 7 Publication Results")
    print(f"   Source: {results_dir}")
    print()
    
    data = load_and_reparse_results(results_dir)
    if not data:
        print("❌ Could not load results")
        sys.exit(1)
    
    figures_dir = results_dir / "figures"
    figures_dir.mkdir(exist_ok=True)
    
    create_combined_figure(data, figures_dir)
    create_recovery_curve_figure(data, figures_dir)
    create_improvement_comparison(data, figures_dir)
    generate_latex_table(data, figures_dir)
    generate_analysis_report(data, figures_dir)
    
    print()
    print(f"✅ All results saved to: {figures_dir}")


if __name__ == "__main__":
    main()
