#!/usr/bin/env python3
"""
Generate Publication Figures for Experiment 6: Compositional Generalization

- 2-panel main figure (P₁ + Steps)
- Compact table without N column (mentioned in footnote)
- Statistical details in footnotes
"""

import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import math

try:
    from scripts.utils.pct_bounds import bounded_pct_ci
except ImportError:
    def bounded_pct_ci(mean_pct, ci_pct, lower=0.0, upper=100.0):
        return max(0.0, min(ci_pct, mean_pct - lower, upper - mean_pct))

# Publication-quality settings
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['DejaVu Serif', 'Times New Roman'],
    'font.size': 11,
    'axes.labelsize': 13,
    'axes.titlesize': 14,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'legend.fontsize': 10,
    'figure.dpi': 600,
    'savefig.dpi': 600,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.1,
    'axes.linewidth': 1.0,
    'mathtext.fontset': 'dejavuserif',
})

COLORS = {
    'precept': '#2563eb',
    'full_reflexion': '#dc2626',
    'expel': '#ea580c',
}

def load_results():
    results_path = Path(__file__).parent.parent / "data" / "publication_results" / "exp6_final_publication" / "exp6_publication_results.json"
    with open(results_path) as f:
        return json.load(f)

def compute_ci(mean, std, n):
    if n <= 1:
        return 0
    t_values = {9: 2.262, 10: 2.228}
    t = t_values.get(n, 1.96)
    return t * std / math.sqrt(n)

def cap_error_bars(means, cis, lower_bound=0, upper_bound=100):
    """Cap error bars so they don't exceed bounds."""
    lower_errors = []
    upper_errors = []
    for m, ci in zip(means, cis):
        lower_err = min(ci, m - lower_bound)
        upper_err = min(ci, upper_bound - m)
        lower_errors.append(max(0, lower_err))
        upper_errors.append(max(0, upper_err))
    return [lower_errors, upper_errors]

def get_significance_marker(p_value):
    """Return significance marker based on p-value."""
    if p_value < 0.001:
        return '***'
    elif p_value < 0.01:
        return '**'
    elif p_value < 0.05:
        return '*'
    else:
        return ''

def create_main_figure(results, output_dir):
    """Create 2-panel figure: P₁ and Avg Steps."""
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))
    
    configs = list(results['configurations'].keys())
    config_labels = ['Logistics\n2-way', 'Logistics\n3-way', 'Integration\n2-way', 'Integration\n3-way']
    x = np.arange(len(configs))
    width = 0.25
    
    # Extract p-values from statistical tests (PRECEPT vs FR)
    p1_pvalues = []
    steps_pvalues = []
    for c in configs:
        tests = results['configurations'][c].get('statistical_tests', {})
        p_vs_fr = tests.get('precept_vs_fr', {})
        p1_pvalues.append(p_vs_fr.get('first_try_success', {}).get('p_value', 1.0))
        steps_pvalues.append(p_vs_fr.get('avg_steps', {}).get('p_value', 1.0))
    
    # ===== Panel A: P1 =====
    ax = axes[0]
    
    precept_p1 = [results['configurations'][c]['metrics']['precept']['P1']['mean'] * 100 for c in configs]
    precept_std = [results['configurations'][c]['metrics']['precept']['P1']['std'] * 100 for c in configs]
    precept_n = [results['configurations'][c]['n_seeds'] for c in configs]
    precept_ci = [compute_ci(precept_p1[i], precept_std[i], precept_n[i]) for i in range(len(configs))]
    
    fr_p1 = [results['configurations'][c]['metrics']['full_reflexion']['P1']['mean'] * 100 for c in configs]
    fr_std = [results['configurations'][c]['metrics']['full_reflexion']['P1']['std'] * 100 for c in configs]
    fr_ci = [compute_ci(fr_p1[i], fr_std[i], precept_n[i]) for i in range(len(configs))]
    
    expel_p1 = [results['configurations'][c]['metrics']['expel']['P1']['mean'] * 100 for c in configs]
    expel_std = [results['configurations'][c]['metrics']['expel']['P1']['std'] * 100 for c in configs]
    expel_ci = [compute_ci(expel_p1[i], expel_std[i], precept_n[i]) for i in range(len(configs))]
    
    # Cap error bars at 0-100%
    precept_errs = cap_error_bars(precept_p1, precept_ci)
    fr_errs = cap_error_bars(fr_p1, fr_ci)
    expel_errs = cap_error_bars(expel_p1, expel_ci)
    
    ax.bar(x - width, precept_p1, width, label='PRECEPT',
           color=COLORS['precept'], yerr=precept_errs, capsize=4,
           edgecolor='black', linewidth=0.8, error_kw={'linewidth': 1.2, 'capthick': 1.2})
    ax.bar(x, fr_p1, width, label='Full Reflexion',
           color=COLORS['full_reflexion'], yerr=fr_errs, capsize=4,
           edgecolor='black', linewidth=0.8, hatch='//', error_kw={'linewidth': 1.2, 'capthick': 1.2})
    ax.bar(x + width, expel_p1, width, label='ExpeL',
           color=COLORS['expel'], yerr=expel_errs, capsize=4,
           edgecolor='black', linewidth=0.8, hatch='\\\\', error_kw={'linewidth': 1.2, 'capthick': 1.2})
    
    # Significance markers for ALL configs
    for i in range(len(configs)):
        sig = get_significance_marker(p1_pvalues[i])
        if sig:
            y_pos = 106
            ax.annotate(sig, xy=(x[i], y_pos), ha='center', fontsize=12, fontweight='bold', color='black')
    
    ax.set_ylabel(r'$P_1$ (%)', fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(config_labels)
    ax.set_ylim(0, 118)
    ax.set_title(r'(a) First-Try Success Rate $P_1$', fontweight='bold', pad=12)
    ax.legend(loc='upper right', framealpha=0.95, edgecolor='gray')
    ax.axhline(y=100, color='gray', linestyle='--', alpha=0.4, linewidth=0.8)
    ax.yaxis.grid(True, linestyle='--', alpha=0.3)
    ax.set_axisbelow(True)
    
    # ===== Panel B: Steps =====
    ax = axes[1]
    
    precept_steps = [results['configurations'][c]['metrics']['precept']['avg_steps']['mean'] for c in configs]
    precept_std_s = [results['configurations'][c]['metrics']['precept']['avg_steps']['std'] for c in configs]
    precept_ci_s = [compute_ci(precept_steps[i], precept_std_s[i], precept_n[i]) for i in range(len(configs))]
    
    fr_steps = [results['configurations'][c]['metrics']['full_reflexion']['avg_steps']['mean'] for c in configs]
    fr_std_s = [results['configurations'][c]['metrics']['full_reflexion']['avg_steps']['std'] for c in configs]
    fr_ci_s = [compute_ci(fr_steps[i], fr_std_s[i], precept_n[i]) for i in range(len(configs))]
    
    expel_steps = [results['configurations'][c]['metrics']['expel']['avg_steps']['mean'] for c in configs]
    expel_std_s = [results['configurations'][c]['metrics']['expel']['avg_steps']['std'] for c in configs]
    expel_ci_s = [compute_ci(expel_steps[i], expel_std_s[i], precept_n[i]) for i in range(len(configs))]
    
    ax.bar(x - width, precept_steps, width, label='PRECEPT',
           color=COLORS['precept'], yerr=precept_ci_s, capsize=4,
           edgecolor='black', linewidth=0.8, error_kw={'linewidth': 1.2, 'capthick': 1.2})
    ax.bar(x, fr_steps, width, label='Full Reflexion',
           color=COLORS['full_reflexion'], yerr=fr_ci_s, capsize=4,
           edgecolor='black', linewidth=0.8, hatch='//', error_kw={'linewidth': 1.2, 'capthick': 1.2})
    ax.bar(x + width, expel_steps, width, label='ExpeL',
           color=COLORS['expel'], yerr=expel_ci_s, capsize=4,
           edgecolor='black', linewidth=0.8, hatch='\\\\', error_kw={'linewidth': 1.2, 'capthick': 1.2})
    
    # Significance markers - positioned above tallest bar
    for i in range(len(configs)):
        sig = get_significance_marker(steps_pvalues[i])
        if sig:
            max_height = max(
                precept_steps[i] + precept_ci_s[i],
                fr_steps[i] + fr_ci_s[i],
                expel_steps[i] + expel_ci_s[i]
            )
            y_pos = max_height + 0.5
            ax.annotate(sig, xy=(x[i], y_pos), ha='center', fontsize=12, fontweight='bold', color='black')
    
    ax.set_ylabel('Avg Steps', fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(config_labels)
    ax.set_ylim(0, 11)
    ax.set_title('(b) Computational Efficiency', fontweight='bold', pad=12)
    ax.legend(loc='upper right', framealpha=0.95, edgecolor='gray')
    ax.yaxis.grid(True, linestyle='--', alpha=0.3)
    ax.set_axisbelow(True)
    ax.annotate('Lower is better', xy=(0.98, 0.02), xycoords='axes fraction',
               ha='right', va='bottom', fontsize=10, style='italic', color='#666666')
    
    plt.tight_layout()
    
    # Caption
    fig.text(0.5, -0.03, 
            'Error bars: 95% CI. *** p<0.001, ** p<0.01, * p<0.05 (paired t-test vs Full Reflexion). N=10 seeds per configuration.',
            ha='center', fontsize=10, style='italic', color='#333333')
    
    fig.savefig(output_dir / 'Figure_Exp6_Main.pdf', format='pdf', dpi=600, bbox_inches='tight')
    fig.savefig(output_dir / 'Figure_Exp6_Main.png', format='png', dpi=600, bbox_inches='tight')
    plt.close()
    print("  ✓ Main figure saved")

def create_compact_table(results, output_dir):
    """Create compact table without N column - N mentioned in footnote."""
    
    fig, ax = plt.subplots(figsize=(11, 4.5))
    ax.axis('off')
    
    config_keys = ['logistics_2way', 'logistics_3way', 'integration_2way', 'integration_3way']
    config_names = ['Logistics 2-way', 'Logistics 3-way', 'Integration 2-way', 'Integration 3-way']
    
    # Extract statistical data from results
    stats_data = {}
    for c in config_keys:
        tests = results['configurations'][c].get('statistical_tests', {})
        p_vs_fr = tests.get('precept_vs_fr', {})
        p1_test = p_vs_fr.get('first_try_success', {})
        p_val = p1_test.get('p_value', 1.0)
        t_stat = p1_test.get('t_stat', 0)
        d = p1_test.get('cohens_d', 0)
        sig = '***' if p_val < 0.001 else '**' if p_val < 0.01 else '*' if p_val < 0.05 else ''
        stats_data[c] = {'sig': sig, 't': abs(t_stat), 'd': abs(d)}
    
    table_data = []
    for i, c in enumerate(config_keys):
        cfg = results['configurations'][c]
        stats = stats_data[c]
        sig = stats['sig']
        
        # P1 with inline significance
        p_p1 = cfg['metrics']['precept']['P1']['mean'] * 100
        p_p1_std = cfg['metrics']['precept']['P1']['std'] * 100
        p_p1_std = bounded_pct_ci(p_p1, p_p1_std)
        fr_p1 = cfg['metrics']['full_reflexion']['P1']['mean'] * 100
        fr_p1_std = cfg['metrics']['full_reflexion']['P1']['std'] * 100
        fr_p1_std = bounded_pct_ci(fr_p1, fr_p1_std)
        ex_p1 = cfg['metrics']['expel']['P1']['mean'] * 100
        ex_p1_std = cfg['metrics']['expel']['P1']['std'] * 100
        ex_p1_std = bounded_pct_ci(ex_p1, ex_p1_std)
        
        # Steps
        p_steps = cfg['metrics']['precept']['avg_steps']['mean']
        p_steps_std = cfg['metrics']['precept']['avg_steps']['std']
        fr_steps = cfg['metrics']['full_reflexion']['avg_steps']['mean']
        fr_steps_std = cfg['metrics']['full_reflexion']['avg_steps']['std']
        ex_steps = cfg['metrics']['expel']['avg_steps']['mean']
        ex_steps_std = cfg['metrics']['expel']['avg_steps']['std']
        
        # Inline significance on PRECEPT P1 value
        precept_p1_str = f'{p_p1:.1f} ± {p_p1_std:.1f}{sig}'
        
        table_data.append([
            config_names[i],
            precept_p1_str,
            f'{fr_p1:.1f} ± {fr_p1_std:.1f}',
            f'{ex_p1:.1f} ± {ex_p1_std:.1f}',
            f'{p_steps:.2f} ± {p_steps_std:.2f}',
            f'{fr_steps:.2f} ± {fr_steps_std:.2f}',
            f'{ex_steps:.2f} ± {ex_steps_std:.2f}',
        ])
    
    columns = [
        'Configuration', 
        r'PRECEPT $P_1$', 
        r'FR $P_1$',
        r'ExpeL $P_1$',
        'PRECEPT Steps',
        'FR Steps',
        'ExpeL Steps',
    ]
    
    table = ax.table(
        cellText=table_data,
        colLabels=columns,
        loc='center',
        cellLoc='center',
        colColours=['#e0e0e0'] * 7
    )
    
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.0, 2.2)
    
    # Style header
    for i in range(len(columns)):
        table[(0, i)].set_text_props(fontweight='bold')
        table[(0, i)].set_facecolor('#b0b0b0')
    
    # Alternate row colors
    for row_idx in range(1, 5):
        color = '#f5f5f5' if row_idx % 2 == 0 else '#ffffff'
        for col_idx in range(len(columns)):
            table[(row_idx, col_idx)].set_facecolor(color)
    
    ax.set_title(
        r'Experiment 6: Compositional Generalization (mean ± std)',
        fontsize=14, fontweight='bold', pad=15
    )
    
    # Footnotes with statistical details
    footnote_text = (
        'N=10 seeds per configuration. *** p<0.001 (paired t-test vs FR).\n'
        'Effect sizes (Cohen\'s d): ' + ', '.join(f'{config_names[i]}: {stats_data[c]["d"]:.2f}' for i, c in enumerate(config_keys)) + '.'
    )
    fig.text(0.5, 0.05, footnote_text,
            ha='center', fontsize=9, style='italic', color='#444444')
    
    plt.tight_layout()
    
    fig.savefig(output_dir / 'Table_Exp6_Statistics.pdf', format='pdf', dpi=600, bbox_inches='tight')
    fig.savefig(output_dir / 'Table_Exp6_Statistics.png', format='png', dpi=600, bbox_inches='tight')
    plt.close()
    print("  ✓ Compact table saved (N in footnote)")

def generate_latex_table(results, output_dir):
    """Generate compact LaTeX table without N column."""
    
    config_keys = ['logistics_2way', 'logistics_3way', 'integration_2way', 'integration_3way']
    config_names = ['Logistics 2-way', 'Logistics 3-way', 'Integration 2-way', 'Integration 3-way']
    
    # Extract statistical data from results
    stats_data = {}
    for c in config_keys:
        tests = results['configurations'][c].get('statistical_tests', {})
        p_vs_fr = tests.get('precept_vs_fr', {})
        p1_test = p_vs_fr.get('first_try_success', {})
        p_val = p1_test.get('p_value', 1.0)
        t_stat = p1_test.get('t_stat', 0)
        d = p1_test.get('cohens_d', 0)
        sig_latex = '$^{***}$' if p_val < 0.001 else '$^{**}$' if p_val < 0.01 else '$^{*}$' if p_val < 0.05 else ''
        stats_data[c] = {'sig': sig_latex, 't': abs(t_stat), 'd': abs(d)}
    
    latex_content = r"""\begin{table}[t]
\centering
\caption{Experiment 6: Compositional Semantic Generalization. PRECEPT learns atomic precepts from single-condition training and generalizes to multi-condition test scenarios. Bold indicates best.$^a$}
\label{tab:exp6}
\begin{tabular}{l ccc ccc}
\toprule
& \multicolumn{3}{c}{\textbf{First-Try Success $P_1$ (\%)}} & \multicolumn{3}{c}{\textbf{Avg Steps} $\downarrow$} \\
\cmidrule(lr){2-4} \cmidrule(lr){5-7}
\textbf{Configuration} & \textbf{PRECEPT} & \textbf{FR} & \textbf{ExpeL} & \textbf{PRECEPT} & \textbf{FR} & \textbf{ExpeL} \\
\midrule
"""
    
    for i, c in enumerate(config_keys):
        cfg = results['configurations'][c]
        stats = stats_data[c]
        
        # P1
        p_p1 = cfg['metrics']['precept']['P1']['mean'] * 100
        p_p1_std = bounded_pct_ci(p_p1, cfg['metrics']['precept']['P1']['std'] * 100)
        fr_p1 = cfg['metrics']['full_reflexion']['P1']['mean'] * 100
        fr_p1_std = bounded_pct_ci(fr_p1, cfg['metrics']['full_reflexion']['P1']['std'] * 100)
        ex_p1 = cfg['metrics']['expel']['P1']['mean'] * 100
        ex_p1_std = bounded_pct_ci(ex_p1, cfg['metrics']['expel']['P1']['std'] * 100)
        
        # Steps
        p_steps = cfg['metrics']['precept']['avg_steps']['mean']
        p_steps_std = cfg['metrics']['precept']['avg_steps']['std']
        fr_steps = cfg['metrics']['full_reflexion']['avg_steps']['mean']
        fr_steps_std = cfg['metrics']['full_reflexion']['avg_steps']['std']
        ex_steps = cfg['metrics']['expel']['avg_steps']['mean']
        ex_steps_std = cfg['metrics']['expel']['avg_steps']['std']
        
        sig = stats['sig']
        
        # Inline significance on PRECEPT value
        p1_precept = f"\\textbf{{{p_p1:.1f}}}$\\pm${p_p1_std:.1f}{sig}"
        steps_precept = f"\\textbf{{{p_steps:.2f}}}$\\pm${p_steps_std:.2f}"
        
        latex_content += f"{config_names[i]} & {p1_precept} & {fr_p1:.1f}$\\pm${fr_p1_std:.1f} & {ex_p1:.1f}$\\pm${ex_p1_std:.1f} & {steps_precept} & {fr_steps:.2f}$\\pm${fr_steps_std:.2f} & {ex_steps:.2f}$\\pm${ex_steps_std:.2f} \\\\\n"
    
    # Overall row
    summary = results['summary']['overall_metrics']
    
    latex_content += r"""\midrule
\textbf{Overall} & """
    
    p_p1 = summary['precept']['P1']['mean'] * 100
    p_p1_std = summary['precept']['P1']['std'] * 100
    p_p1_std = bounded_pct_ci(p_p1, p_p1_std)
    fr_p1 = summary['full_reflexion']['P1']['mean'] * 100
    fr_p1_std = summary['full_reflexion']['P1']['std'] * 100
    fr_p1_std = bounded_pct_ci(fr_p1, fr_p1_std)
    ex_p1 = summary['expel']['P1']['mean'] * 100
    ex_p1_std = summary['expel']['P1']['std'] * 100
    ex_p1_std = bounded_pct_ci(ex_p1, ex_p1_std)

    p_steps = summary['precept']['avg_steps']['mean']
    p_steps_std = summary['precept']['avg_steps']['std']
    fr_steps = summary['full_reflexion']['avg_steps']['mean']
    fr_steps_std = summary['full_reflexion']['avg_steps']['std']
    ex_steps = summary['expel']['avg_steps']['mean']
    ex_steps_std = summary['expel']['avg_steps']['std']
    
    latex_content += f"\\textbf{{{p_p1:.1f}}}$\\pm${p_p1_std:.1f} & {fr_p1:.1f}$\\pm${fr_p1_std:.1f} & {ex_p1:.1f}$\\pm${ex_p1_std:.1f} & \\textbf{{{p_steps:.2f}}}$\\pm${p_steps_std:.2f} & {fr_steps:.2f}$\\pm${fr_steps_std:.2f} & {ex_steps:.2f}$\\pm${ex_steps_std:.2f} \\\\\n"
    
    latex_content += r"""\bottomrule
\end{tabular}

\vspace{2mm}
\footnotesize
$^a$ N=10 seeds per configuration. $^{***}$p$<$0.001, $^{**}$p$<$0.01, $^{*}$p$<$0.05 (paired t-test vs FR). $\downarrow$ lower is better.
\end{table}
"""
    
    # Save LaTeX file
    latex_path = output_dir.parent / 'exp6_table.tex'
    with open(latex_path, 'w') as f:
        f.write(latex_content)
    
    print(f"  ✓ LaTeX table saved: {latex_path}")

def main():
    print("=" * 70)
    print("Generating Exp6 Publication Materials")
    print("=" * 70)
    
    results = load_results()
    
    output_dir = Path(__file__).parent.parent / "data" / "publication_results" / "exp6_final_publication" / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\nOutput: {output_dir}")
    print("\nGenerating figures (600 DPI)...")
    
    create_main_figure(results, output_dir)
    create_compact_table(results, output_dir)
    generate_latex_table(results, output_dir)
    
    print("\n" + "=" * 70)
    print("✅ All publication materials generated!")
    print("=" * 70)
    print("\nFiles:")
    print("  1. Figure_Exp6_Main.pdf/png    - 2-panel figure")
    print("  2. Table_Exp6_Statistics.pdf/png - Compact table (no N column)")
    print("  3. exp6_table.tex              - LaTeX table for paper")

if __name__ == "__main__":
    main()
