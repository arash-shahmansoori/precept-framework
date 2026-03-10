#!/usr/bin/env python3
"""
Generate Publication-Quality Figures for Experiment 6: Compositional Generalization

Creates highest-quality figures with:
- 95% Confidence Intervals as error bars
- Standard deviation annotations
- 600 DPI resolution
- Vector PDF output
- Proper LaTeX math rendering

Output: PDF and PNG files suitable for top-tier publication
"""

import json
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from pathlib import Path

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
    'lines.linewidth': 1.5,
    'text.usetex': False,  # Set True if LaTeX is installed
    'mathtext.fontset': 'dejavuserif',
})

# Color scheme (colorblind-friendly, high contrast)
COLORS = {
    'precept': '#1f77b4',        # Strong blue
    'full_reflexion': '#d62728',  # Strong red
    'expel': '#ff7f0e',          # Strong orange
}

HATCHES = {
    'precept': '',
    'full_reflexion': '//',
    'expel': '\\\\',
}

LABELS = {
    'precept': 'PRECEPT',
    'full_reflexion': 'Full Reflexion',
    'expel': 'ExpeL',
}

def load_results():
    """Load publication results."""
    results_path = Path(__file__).parent.parent / "data" / "publication_results" / "exp6_final_publication" / "exp6_publication_results.json"
    with open(results_path) as f:
        return json.load(f)

def compute_ci(mean, std, n, confidence=0.95):
    """Compute confidence interval."""
    import math
    if n <= 1:
        return 0
    # t-value for 95% CI
    t_values = {9: 2.262, 10: 2.228, 39: 2.023}
    t = t_values.get(n, 1.96)
    se = std / math.sqrt(n)
    return t * se

def _cap_pct_errorbars(means, cis):
    """Create asymmetric [lower, upper] error-bar lists bounded to [0, 100]."""
    lower = [max(0.0, min(ci, m)) for m, ci in zip(means, cis)]
    upper = [max(0.0, min(ci, 100.0 - m)) for m, ci in zip(means, cis)]
    return [lower, upper]

def create_p1_comparison_with_ci(results, output_dir):
    """Create P1 (First-Try Success) comparison bar chart with 95% CI."""
    configs = list(results['configurations'].keys())
    config_labels = ['Logistics\n2-way', 'Logistics\n3-way', 'Booking\n2-way', 'Booking\n3-way']
    
    x = np.arange(len(configs))
    width = 0.26
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Extract data with CI
    precept_p1 = [results['configurations'][c]['metrics']['precept']['P1']['mean'] * 100 for c in configs]
    precept_std = [results['configurations'][c]['metrics']['precept']['P1']['std'] * 100 for c in configs]
    precept_n = [results['configurations'][c]['n_seeds'] for c in configs]
    precept_ci = [compute_ci(precept_p1[i], precept_std[i], precept_n[i]) for i in range(len(configs))]
    
    fr_p1 = [results['configurations'][c]['metrics']['full_reflexion']['P1']['mean'] * 100 for c in configs]
    fr_std = [results['configurations'][c]['metrics']['full_reflexion']['P1']['std'] * 100 for c in configs]
    fr_n = [results['configurations'][c]['n_seeds'] for c in configs]
    fr_ci = [compute_ci(fr_p1[i], fr_std[i], fr_n[i]) for i in range(len(configs))]
    
    expel_p1 = [results['configurations'][c]['metrics']['expel']['P1']['mean'] * 100 for c in configs]
    expel_std = [results['configurations'][c]['metrics']['expel']['P1']['std'] * 100 for c in configs]
    expel_n = [results['configurations'][c]['n_seeds'] for c in configs]
    expel_ci = [compute_ci(expel_p1[i], expel_std[i], expel_n[i]) for i in range(len(configs))]
    
    # Create bars with error bars (95% CI, bounded to [0,100])
    bars1 = ax.bar(x - width, precept_p1, width, label=LABELS['precept'], 
                   color=COLORS['precept'], yerr=_cap_pct_errorbars(precept_p1, precept_ci), capsize=5, 
                   edgecolor='black', linewidth=1.2, error_kw={'linewidth': 1.5, 'capthick': 1.5})
    bars2 = ax.bar(x, fr_p1, width, label=LABELS['full_reflexion'],
                   color=COLORS['full_reflexion'], yerr=_cap_pct_errorbars(fr_p1, fr_ci), capsize=5,
                   edgecolor='black', linewidth=1.2, hatch=HATCHES['full_reflexion'],
                   error_kw={'linewidth': 1.5, 'capthick': 1.5})
    bars3 = ax.bar(x + width, expel_p1, width, label=LABELS['expel'],
                   color=COLORS['expel'], yerr=_cap_pct_errorbars(expel_p1, expel_ci), capsize=5,
                   edgecolor='black', linewidth=1.2, hatch=HATCHES['expel'],
                   error_kw={'linewidth': 1.5, 'capthick': 1.5})
    
    # Add significance markers
    significance = ['***', '***', '***', 'n.s.']
    for i, sig in enumerate(significance):
        y_pos = max(precept_p1[i] + precept_ci[i], fr_p1[i] + fr_ci[i], expel_p1[i] + expel_ci[i]) + 5
        ax.annotate(sig, xy=(x[i], y_pos),
                   ha='center', fontsize=12, fontweight='bold',
                   color='#2c3e50' if sig != 'n.s.' else '#7f8c8d')
    
    # Add value labels on PRECEPT bars
    for i, (bar, val, ci) in enumerate(zip(bars1, precept_p1, precept_ci)):
        ax.annotate(f'{val:.0f}%',
                   xy=(bar.get_x() + bar.get_width() / 2, val - 8),
                   ha='center', va='top', fontsize=10, fontweight='bold', color='white')
    
    ax.set_ylabel(r'First-Try Success Rate $P_1$ (%)', fontsize=13, fontweight='bold')
    ax.set_xlabel('Configuration', fontsize=13, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(config_labels, fontsize=11)
    ax.set_ylim(0, 120)
    ax.legend(loc='upper right', framealpha=0.95, edgecolor='black', fancybox=False)
    ax.axhline(y=100, color='#2c3e50', linestyle='--', alpha=0.5, linewidth=1.0)
    
    # Add grid
    ax.yaxis.grid(True, linestyle='--', alpha=0.4, linewidth=0.8)
    ax.set_axisbelow(True)
    
    # Add annotation for error bars
    ax.annotate('Error bars: 95% CI', xy=(0.02, 0.98), xycoords='axes fraction',
               fontsize=9, ha='left', va='top', style='italic', color='#555555')
    
    # Add annotation for significance
    ax.annotate('*** p < 0.001', xy=(0.02, 0.93), xycoords='axes fraction',
               fontsize=9, ha='left', va='top', color='#555555')
    
    plt.tight_layout()
    
    fig.savefig(output_dir / 'exp6_p1_comparison.pdf', format='pdf', dpi=600)
    fig.savefig(output_dir / 'exp6_p1_comparison.png', format='png', dpi=600)
    plt.close()
    print(f"  ✓ P1 comparison chart saved (with 95% CI)")

def create_pt_comparison_with_ci(results, output_dir):
    """Create Pt (Eventual Success) comparison bar chart with 95% CI."""
    configs = list(results['configurations'].keys())
    config_labels = ['Logistics\n2-way', 'Logistics\n3-way', 'Booking\n2-way', 'Booking\n3-way']
    
    x = np.arange(len(configs))
    width = 0.26
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Extract data with CI
    precept_pt = [results['configurations'][c]['metrics']['precept']['Pt']['mean'] * 100 for c in configs]
    precept_std = [results['configurations'][c]['metrics']['precept']['Pt']['std'] * 100 for c in configs]
    precept_n = [results['configurations'][c]['n_seeds'] for c in configs]
    precept_ci = [compute_ci(precept_pt[i], precept_std[i], precept_n[i]) for i in range(len(configs))]
    
    fr_pt = [results['configurations'][c]['metrics']['full_reflexion']['Pt']['mean'] * 100 for c in configs]
    fr_std = [results['configurations'][c]['metrics']['full_reflexion']['Pt']['std'] * 100 for c in configs]
    fr_n = [results['configurations'][c]['n_seeds'] for c in configs]
    fr_ci = [compute_ci(fr_pt[i], fr_std[i], fr_n[i]) for i in range(len(configs))]
    
    expel_pt = [results['configurations'][c]['metrics']['expel']['Pt']['mean'] * 100 for c in configs]
    expel_std = [results['configurations'][c]['metrics']['expel']['Pt']['std'] * 100 for c in configs]
    expel_n = [results['configurations'][c]['n_seeds'] for c in configs]
    expel_ci = [compute_ci(expel_pt[i], expel_std[i], expel_n[i]) for i in range(len(configs))]
    
    # Create bars with error bars (95% CI, bounded to [0,100])
    bars1 = ax.bar(x - width, precept_pt, width, label=LABELS['precept'],
                   color=COLORS['precept'], yerr=_cap_pct_errorbars(precept_pt, precept_ci), capsize=5,
                   edgecolor='black', linewidth=1.2, error_kw={'linewidth': 1.5, 'capthick': 1.5})
    bars2 = ax.bar(x, fr_pt, width, label=LABELS['full_reflexion'],
                   color=COLORS['full_reflexion'], yerr=_cap_pct_errorbars(fr_pt, fr_ci), capsize=5,
                   edgecolor='black', linewidth=1.2, hatch=HATCHES['full_reflexion'],
                   error_kw={'linewidth': 1.5, 'capthick': 1.5})
    bars3 = ax.bar(x + width, expel_pt, width, label=LABELS['expel'],
                   color=COLORS['expel'], yerr=_cap_pct_errorbars(expel_pt, expel_ci), capsize=5,
                   edgecolor='black', linewidth=1.2, hatch=HATCHES['expel'],
                   error_kw={'linewidth': 1.5, 'capthick': 1.5})
    
    ax.set_ylabel(r'Eventual Success Rate $P_t$ (%)', fontsize=13, fontweight='bold')
    ax.set_xlabel('Configuration', fontsize=13, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(config_labels, fontsize=11)
    ax.set_ylim(0, 120)
    ax.legend(loc='lower right', framealpha=0.95, edgecolor='black', fancybox=False)
    ax.axhline(y=100, color='#2c3e50', linestyle='--', alpha=0.5, linewidth=1.0)
    
    ax.yaxis.grid(True, linestyle='--', alpha=0.4, linewidth=0.8)
    ax.set_axisbelow(True)
    
    ax.annotate('Error bars: 95% CI', xy=(0.02, 0.98), xycoords='axes fraction',
               fontsize=9, ha='left', va='top', style='italic', color='#555555')
    
    plt.tight_layout()
    
    fig.savefig(output_dir / 'exp6_pt_comparison.pdf', format='pdf', dpi=600)
    fig.savefig(output_dir / 'exp6_pt_comparison.png', format='png', dpi=600)
    plt.close()
    print(f"  ✓ Pt comparison chart saved (with 95% CI)")

def create_steps_comparison_with_ci(results, output_dir):
    """Create steps efficiency comparison bar chart with 95% CI."""
    configs = list(results['configurations'].keys())
    config_labels = ['Logistics\n2-way', 'Logistics\n3-way', 'Booking\n2-way', 'Booking\n3-way']
    
    x = np.arange(len(configs))
    width = 0.26
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Extract data with CI
    precept_steps = [results['configurations'][c]['metrics']['precept']['avg_steps']['mean'] for c in configs]
    precept_std = [results['configurations'][c]['metrics']['precept']['avg_steps']['std'] for c in configs]
    precept_n = [results['configurations'][c]['n_seeds'] for c in configs]
    precept_ci = [compute_ci(precept_steps[i], precept_std[i], precept_n[i]) for i in range(len(configs))]
    
    fr_steps = [results['configurations'][c]['metrics']['full_reflexion']['avg_steps']['mean'] for c in configs]
    fr_std = [results['configurations'][c]['metrics']['full_reflexion']['avg_steps']['std'] for c in configs]
    fr_n = [results['configurations'][c]['n_seeds'] for c in configs]
    fr_ci = [compute_ci(fr_steps[i], fr_std[i], fr_n[i]) for i in range(len(configs))]
    
    expel_steps = [results['configurations'][c]['metrics']['expel']['avg_steps']['mean'] for c in configs]
    expel_std = [results['configurations'][c]['metrics']['expel']['avg_steps']['std'] for c in configs]
    expel_n = [results['configurations'][c]['n_seeds'] for c in configs]
    expel_ci = [compute_ci(expel_steps[i], expel_std[i], expel_n[i]) for i in range(len(configs))]
    
    # Create bars with error bars (95% CI)
    bars1 = ax.bar(x - width, precept_steps, width, label=LABELS['precept'],
                   color=COLORS['precept'], yerr=precept_ci, capsize=5,
                   edgecolor='black', linewidth=1.2, error_kw={'linewidth': 1.5, 'capthick': 1.5})
    bars2 = ax.bar(x, fr_steps, width, label=LABELS['full_reflexion'],
                   color=COLORS['full_reflexion'], yerr=fr_ci, capsize=5,
                   edgecolor='black', linewidth=1.2, hatch=HATCHES['full_reflexion'],
                   error_kw={'linewidth': 1.5, 'capthick': 1.5})
    bars3 = ax.bar(x + width, expel_steps, width, label=LABELS['expel'],
                   color=COLORS['expel'], yerr=expel_ci, capsize=5,
                   edgecolor='black', linewidth=1.2, hatch=HATCHES['expel'],
                   error_kw={'linewidth': 1.5, 'capthick': 1.5})
    
    # Add value labels
    for bars, values in [(bars1, precept_steps), (bars2, fr_steps), (bars3, expel_steps)]:
        for bar, val in zip(bars, values):
            ax.annotate(f'{val:.1f}',
                       xy=(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3),
                       ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    ax.set_ylabel('Average Steps per Task', fontsize=13, fontweight='bold')
    ax.set_xlabel('Configuration', fontsize=13, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(config_labels, fontsize=11)
    ax.set_ylim(0, 10)
    ax.legend(loc='upper right', framealpha=0.95, edgecolor='black', fancybox=False)
    
    ax.yaxis.grid(True, linestyle='--', alpha=0.4, linewidth=0.8)
    ax.set_axisbelow(True)
    
    ax.annotate('Error bars: 95% CI', xy=(0.02, 0.98), xycoords='axes fraction',
               fontsize=9, ha='left', va='top', style='italic', color='#555555')
    ax.annotate('Lower is better', xy=(0.02, 0.93), xycoords='axes fraction',
               fontsize=9, ha='left', va='top', style='italic', color='#555555')
    
    plt.tight_layout()
    
    fig.savefig(output_dir / 'exp6_steps_comparison.pdf', format='pdf', dpi=600)
    fig.savefig(output_dir / 'exp6_steps_comparison.png', format='png', dpi=600)
    plt.close()
    print(f"  ✓ Steps comparison chart saved (with 95% CI)")

def create_combined_figure(results, output_dir):
    """Create a high-quality combined 2x2 figure with all key metrics and CI."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    configs = list(results['configurations'].keys())
    config_labels = ['Log. 2-way', 'Log. 3-way', 'Book. 2-way', 'Book. 3-way']
    x = np.arange(len(configs))
    width = 0.26
    
    # Helper to extract data
    def get_data(metric):
        precept = [results['configurations'][c]['metrics']['precept'][metric]['mean'] for c in configs]
        precept_std = [results['configurations'][c]['metrics']['precept'][metric]['std'] for c in configs]
        fr = [results['configurations'][c]['metrics']['full_reflexion'][metric]['mean'] for c in configs]
        fr_std = [results['configurations'][c]['metrics']['full_reflexion'][metric]['std'] for c in configs]
        expel = [results['configurations'][c]['metrics']['expel'][metric]['mean'] for c in configs]
        expel_std = [results['configurations'][c]['metrics']['expel'][metric]['std'] for c in configs]
        n = [results['configurations'][c]['n_seeds'] for c in configs]
        return precept, precept_std, fr, fr_std, expel, expel_std, n
    
    # ===== (a) P1 =====
    ax = axes[0, 0]
    precept, precept_std, fr, fr_std, expel, expel_std, n = get_data('P1')
    precept = [v * 100 for v in precept]
    precept_std = [v * 100 for v in precept_std]
    fr = [v * 100 for v in fr]
    fr_std = [v * 100 for v in fr_std]
    expel = [v * 100 for v in expel]
    expel_std = [v * 100 for v in expel_std]
    precept_ci = [compute_ci(precept[i], precept_std[i], n[i]) for i in range(len(configs))]
    fr_ci = [compute_ci(fr[i], fr_std[i], n[i]) for i in range(len(configs))]
    expel_ci = [compute_ci(expel[i], expel_std[i], n[i]) for i in range(len(configs))]
    
    ax.bar(x - width, precept, width, label='PRECEPT', color=COLORS['precept'], 
           yerr=_cap_pct_errorbars(precept, precept_ci), capsize=4, edgecolor='black', linewidth=1.0,
           error_kw={'linewidth': 1.2, 'capthick': 1.2})
    ax.bar(x, fr, width, label='Full Reflexion', color=COLORS['full_reflexion'],
           yerr=_cap_pct_errorbars(fr, fr_ci), capsize=4, edgecolor='black', linewidth=1.0, hatch='//',
           error_kw={'linewidth': 1.2, 'capthick': 1.2})
    ax.bar(x + width, expel, width, label='ExpeL', color=COLORS['expel'],
           yerr=_cap_pct_errorbars(expel, expel_ci), capsize=4, edgecolor='black', linewidth=1.0, hatch='\\\\',
           error_kw={'linewidth': 1.2, 'capthick': 1.2})
    
    ax.set_ylabel(r'$P_1$ (%)', fontsize=12, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(config_labels, fontsize=10)
    ax.set_ylim(0, 120)
    ax.set_title(r'(a) First-Try Success Rate $P_1$', fontsize=13, fontweight='bold')
    ax.legend(loc='upper right', fontsize=9, framealpha=0.95)
    ax.yaxis.grid(True, linestyle='--', alpha=0.4)
    ax.set_axisbelow(True)
    ax.axhline(y=100, color='gray', linestyle='--', alpha=0.4)
    
    # ===== (b) Pt =====
    ax = axes[0, 1]
    precept, precept_std, fr, fr_std, expel, expel_std, n = get_data('Pt')
    precept = [v * 100 for v in precept]
    precept_std = [v * 100 for v in precept_std]
    fr = [v * 100 for v in fr]
    fr_std = [v * 100 for v in fr_std]
    expel = [v * 100 for v in expel]
    expel_std = [v * 100 for v in expel_std]
    precept_ci = [compute_ci(precept[i], precept_std[i], n[i]) for i in range(len(configs))]
    fr_ci = [compute_ci(fr[i], fr_std[i], n[i]) for i in range(len(configs))]
    expel_ci = [compute_ci(expel[i], expel_std[i], n[i]) for i in range(len(configs))]
    
    ax.bar(x - width, precept, width, label='PRECEPT', color=COLORS['precept'],
           yerr=_cap_pct_errorbars(precept, precept_ci), capsize=4, edgecolor='black', linewidth=1.0,
           error_kw={'linewidth': 1.2, 'capthick': 1.2})
    ax.bar(x, fr, width, label='Full Reflexion', color=COLORS['full_reflexion'],
           yerr=_cap_pct_errorbars(fr, fr_ci), capsize=4, edgecolor='black', linewidth=1.0, hatch='//',
           error_kw={'linewidth': 1.2, 'capthick': 1.2})
    ax.bar(x + width, expel, width, label='ExpeL', color=COLORS['expel'],
           yerr=_cap_pct_errorbars(expel, expel_ci), capsize=4, edgecolor='black', linewidth=1.0, hatch='\\\\',
           error_kw={'linewidth': 1.2, 'capthick': 1.2})
    
    ax.set_ylabel(r'$P_t$ (%)', fontsize=12, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(config_labels, fontsize=10)
    ax.set_ylim(0, 120)
    ax.set_title(r'(b) Eventual Success Rate $P_t$', fontsize=13, fontweight='bold')
    ax.legend(loc='lower right', fontsize=9, framealpha=0.95)
    ax.yaxis.grid(True, linestyle='--', alpha=0.4)
    ax.set_axisbelow(True)
    ax.axhline(y=100, color='gray', linestyle='--', alpha=0.4)
    
    # ===== (c) Steps =====
    ax = axes[1, 0]
    precept, precept_std, fr, fr_std, expel, expel_std, n = get_data('avg_steps')
    precept_ci = [compute_ci(precept[i], precept_std[i], n[i]) for i in range(len(configs))]
    fr_ci = [compute_ci(fr[i], fr_std[i], n[i]) for i in range(len(configs))]
    expel_ci = [compute_ci(expel[i], expel_std[i], n[i]) for i in range(len(configs))]
    
    ax.bar(x - width, precept, width, label='PRECEPT', color=COLORS['precept'],
           yerr=precept_ci, capsize=4, edgecolor='black', linewidth=1.0,
           error_kw={'linewidth': 1.2, 'capthick': 1.2})
    ax.bar(x, fr, width, label='Full Reflexion', color=COLORS['full_reflexion'],
           yerr=fr_ci, capsize=4, edgecolor='black', linewidth=1.0, hatch='//',
           error_kw={'linewidth': 1.2, 'capthick': 1.2})
    ax.bar(x + width, expel, width, label='ExpeL', color=COLORS['expel'],
           yerr=expel_ci, capsize=4, edgecolor='black', linewidth=1.0, hatch='\\\\',
           error_kw={'linewidth': 1.2, 'capthick': 1.2})
    
    ax.set_ylabel('Avg Steps', fontsize=12, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(config_labels, fontsize=10)
    ax.set_ylim(0, 10)
    ax.set_title('(c) Computational Efficiency (lower is better)', fontsize=13, fontweight='bold')
    ax.legend(loc='upper right', fontsize=9, framealpha=0.95)
    ax.yaxis.grid(True, linestyle='--', alpha=0.4)
    ax.set_axisbelow(True)
    
    # ===== (d) Advantages =====
    ax = axes[1, 1]
    
    p1_adv = [results['configurations'][c]['advantages']['P1_vs_full_reflexion'] for c in configs]
    pt_adv = [results['configurations'][c]['advantages']['Pt_vs_full_reflexion'] for c in configs]
    steps_saved = [results['configurations'][c]['advantages']['steps_saved_vs_full_reflexion'] for c in configs]
    
    x_adv = np.arange(len(configs))
    width_adv = 0.28
    
    bars1 = ax.bar(x_adv - width_adv, p1_adv, width_adv, label=r'$P_1$ Advantage (pp)',
                   color=COLORS['precept'], edgecolor='black', linewidth=1.0)
    bars2 = ax.bar(x_adv, pt_adv, width_adv, label=r'$P_t$ Advantage (pp)',
                   color='#2ecc71', edgecolor='black', linewidth=1.0)
    bars3 = ax.bar(x_adv + width_adv, steps_saved, width_adv, label='Steps Saved',
                   color='#9b59b6', edgecolor='black', linewidth=1.0)
    
    ax.set_ylabel('Advantage vs Full Reflexion', fontsize=12, fontweight='bold')
    ax.set_xticks(x_adv)
    ax.set_xticklabels(config_labels, fontsize=10)
    ax.set_ylim(-5, 85)
    ax.set_title('(d) PRECEPT Advantages', fontsize=13, fontweight='bold')
    ax.legend(loc='upper right', fontsize=9, framealpha=0.95)
    ax.yaxis.grid(True, linestyle='--', alpha=0.4)
    ax.set_axisbelow(True)
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
    
    # Add value labels
    for bar, val in zip(bars1, p1_adv):
        if val > 5:
            ax.annotate(f'+{val:.0f}', xy=(bar.get_x() + bar.get_width()/2, val + 1),
                       ha='center', fontsize=8, fontweight='bold')
    
    plt.suptitle('Experiment 6: Compositional Semantic Generalization\n(Error bars: 95% Confidence Intervals)',
                fontsize=15, fontweight='bold', y=1.02)
    
    plt.tight_layout()
    
    fig.savefig(output_dir / 'exp6_combined_figure.pdf', format='pdf', dpi=600)
    fig.savefig(output_dir / 'exp6_combined_figure.png', format='png', dpi=600)
    plt.close()
    print(f"  ✓ Combined figure saved (with 95% CI)")

def create_effect_size_heatmap(results, output_dir):
    """Create publication-quality effect size heatmap."""
    configs = ['logistics_2way', 'logistics_3way', 'booking_2way', 'booking_3way']
    config_labels = ['Logistics\n2-way', 'Logistics\n3-way', 'Booking\n2-way', 'Booking\n3-way']
    comparisons = [r'$P_1$ vs FR', r'$P_1$ vs ExpeL']
    
    # Effect sizes from statistical tests
    effect_sizes = np.array([
        [8.27, 9.57],   # logistics_2way
        [4.30, 3.02],   # logistics_3way
        [2.88, 2.19],   # booking_2way
        [0.56, 0.57],   # booking_3way
    ])
    
    fig, ax = plt.subplots(figsize=(7, 6))
    
    im = ax.imshow(effect_sizes, cmap='Blues', aspect='auto', vmin=0, vmax=10)
    
    # Add colorbar
    cbar = ax.figure.colorbar(im, ax=ax, shrink=0.85, pad=0.02)
    cbar.ax.set_ylabel(r"Cohen's $d$ Effect Size", rotation=-90, va="bottom", fontsize=12, fontweight='bold')
    
    # Add text annotations
    for i in range(len(configs)):
        for j in range(len(comparisons)):
            value = effect_sizes[i, j]
            if value >= 1.2:
                category = "huge"
            elif value >= 0.8:
                category = "large"
            elif value >= 0.5:
                category = "medium"
            else:
                category = "small"
            
            text_color = 'white' if value > 4 else 'black'
            ax.text(j, i, f'{value:.2f}\n({category})', ha='center', va='center',
                   color=text_color, fontsize=11, fontweight='bold')
    
    ax.set_xticks(np.arange(len(comparisons)))
    ax.set_yticks(np.arange(len(configs)))
    ax.set_xticklabels(comparisons, fontsize=11)
    ax.set_yticklabels(config_labels, fontsize=11)
    
    ax.set_xlabel('Comparison', fontsize=12, fontweight='bold')
    ax.set_ylabel('Configuration', fontsize=12, fontweight='bold')
    ax.set_title(r"Effect Sizes (Cohen's $d$) for $P_1$ Improvement", fontsize=14, fontweight='bold', pad=15)
    
    # Add interpretation guide
    ax.annotate('Interpretation: small (0.2), medium (0.5), large (0.8), huge (>1.2)',
               xy=(0.5, -0.12), xycoords='axes fraction',
               ha='center', fontsize=9, style='italic', color='#555555')
    
    plt.tight_layout()
    
    fig.savefig(output_dir / 'exp6_effect_sizes.pdf', format='pdf', dpi=600)
    fig.savefig(output_dir / 'exp6_effect_sizes.png', format='png', dpi=600)
    plt.close()
    print(f"  ✓ Effect size heatmap saved")

def main():
    """Generate all publication figures."""
    print("=" * 70)
    print("Generating Highest-Quality Publication Figures for Experiment 6")
    print("=" * 70)
    
    results = load_results()
    
    output_dir = Path(__file__).parent.parent / "data" / "publication_results" / "exp6_final_publication" / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\nOutput directory: {output_dir}")
    print("\nGenerating figures with 95% Confidence Intervals...")
    print("Resolution: 600 DPI\n")
    
    create_p1_comparison_with_ci(results, output_dir)
    create_pt_comparison_with_ci(results, output_dir)
    create_steps_comparison_with_ci(results, output_dir)
    create_combined_figure(results, output_dir)
    create_effect_size_heatmap(results, output_dir)
    
    print("\n" + "=" * 70)
    print("✅ All highest-quality figures generated!")
    print("=" * 70)
    print(f"\nFiles in: {output_dir}")
    print("\nFigures (600 DPI, PDF + PNG):")
    print("  - exp6_p1_comparison.pdf/png      (with 95% CI)")
    print("  - exp6_pt_comparison.pdf/png      (with 95% CI)")
    print("  - exp6_steps_comparison.pdf/png   (with 95% CI)")
    print("  - exp6_combined_figure.pdf/png    (4-panel with 95% CI)")
    print("  - exp6_effect_sizes.pdf/png       (Cohen's d heatmap)")

if __name__ == "__main__":
    main()
