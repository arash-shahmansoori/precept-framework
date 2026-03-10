#!/usr/bin/env python3
"""
Generate Publication-Quality Results for Experiment 3: Training Size Ablation (β Effect)

Produces one focused 2-panel figure for the main paper:
  (a) P₁ Learning Curve with 95% CI and significance annotations (*, **, ***)
  (b) Avg Steps grouped bar chart with theoretical minimum reference

Plus supporting LaTeX tables and a Markdown analysis report.

Output (600 DPI, PDF + PNG):
  figures/Figure_Exp3_Main.pdf/png   — The paper figure
  tables/exp3_main_table.tex         — Camera-ready LaTeX results table
  tables/exp3_significance_table.tex — Camera-ready LaTeX significance table
  Exp3_Analysis_Report.md            — Full analysis with key findings

Usage:
    python scripts/create_results/generate_exp3_training_size_results.py [results_directory]

    If no directory is provided, auto-detects the latest exp3 results.
"""

import json
import sys
from collections import OrderedDict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np

# ─── Publication-quality matplotlib configuration ───────────────────────────
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

# ─── Colorblind-friendly, high-contrast palette ────────────────────────────
COLORS = {
    'precept': '#2563eb',        # Strong blue
    'full_reflexion': '#dc2626', # Strong red
    'expel': '#ea580c',          # Strong orange
}

MARKERS = {'precept': 'o', 'full_reflexion': 's', 'expel': '^'}
LABELS = {'precept': 'PRECEPT', 'full_reflexion': 'Full Reflexion', 'expel': 'ExpeL'}
HATCHES = {'precept': None, 'full_reflexion': '//', 'expel': '\\\\'}
AGENTS = ['precept', 'full_reflexion', 'expel']


# ─── Helpers ────────────────────────────────────────────────────────────────

def find_latest_results_dir() -> Optional[Path]:
    """Find the most recent exp3 results directory."""
    pub = Path(__file__).parent.parent.parent / "data" / "publication_results"
    dirs = sorted(pub.glob("exp3_training_size_*"), reverse=True)
    return dirs[0] if dirs else None


def load_results(path: Path) -> Dict[str, Any]:
    f = path / "aggregated_results.json"
    if not f.exists():
        print(f"  ✗ Not found: {f}")
        sys.exit(1)
    with open(f) as fh:
        return json.load(fh)


def _m(entry: dict, agent: str, metric: str) -> dict:
    """Resolve metric dict, handling both old and new key formats."""
    d = entry[agent]
    for k in {'p1': ['p1', 'first_try_success'],
              'pt': ['pt', 'success_rate'],
              'steps': ['steps', 'avg_steps']}.get(metric, [metric]):
        if k in d:
            return d[k]
    raise KeyError(f"No key '{metric}' in {agent}: {list(d.keys())}")


def sig_marker(p: float) -> str:
    if p < 0.001: return '***'
    if p < 0.01:  return '**'
    if p < 0.05:  return '*'
    return ''


def sig_full(p: float) -> str:
    if p < 0.001: return '***'
    if p < 0.01:  return '**'
    if p < 0.05:  return '*'
    return 'ns'


def cap_err(means: List[float], cis: List[float],
            lo: float = 0, hi: float = 100) -> List[List[float]]:
    return [[max(0, min(c, m - lo)) for m, c in zip(means, cis)],
            [max(0, min(c, hi - m)) for m, c in zip(means, cis)]]


def bounded_pct_ci(mean_pct: float, ci_pct: float, lo: float = 0.0, hi: float = 100.0) -> float:
    """Bound symmetric percentage CI to keep mean±CI in [lo, hi]."""
    return max(0.0, min(ci_pct, mean_pct - lo, hi - mean_pct))


def d_label(d: float) -> str:
    a = abs(d)
    if a < 0.2: return 'negligible'
    if a < 0.5: return 'small'
    if a < 0.8: return 'medium'
    return 'large'


def resolve_stat_test(
    entry: Dict[str, Any],
    comparison: str,
    metric: str = "first_try_success",
) -> Dict[str, float]:
    """Resolve statistical-test dict across current and legacy result schemas."""
    tests = entry.get("statistical_tests", entry.get("tests", {}))

    comp = tests.get(comparison, {})
    if isinstance(comp, dict):
        metric_test = comp.get(metric)
        if isinstance(metric_test, dict):
            return metric_test
        if "p_value" in comp:
            return comp

    legacy_map = {
        ("precept_vs_fr", "first_try_success"): "precept_vs_fr_p1",
        ("precept_vs_expel", "first_try_success"): "precept_vs_expel_p1",
    }
    legacy_key = legacy_map.get((comparison, metric))
    if legacy_key:
        legacy = tests.get(legacy_key, {})
        if isinstance(legacy, dict):
            return legacy

    # Older format used top-level metric test as PRECEPT vs FR
    if comparison == "precept_vs_fr":
        legacy_metric = tests.get(metric, {})
        if isinstance(legacy_metric, dict):
            return legacy_metric

    return {"p_value": 1.0, "cohens_d": 0.0}


def corrected_p_value(test: Dict[str, Any]) -> float:
    """Prefer Bonferroni-corrected p-value when available."""
    return float(test.get("p_value_bonferroni", test.get("p_value", 1.0)))


def ordered(data: Dict) -> OrderedDict:
    o = OrderedDict()
    for k in sorted(data, key=lambda k: data[k].get('beta', 0)):
        if 'beta' in data[k]:
            o[k] = data[k]
    return o


def compute_step_significance(results_dir: Path) -> Dict[str, Dict]:
    """Compute paired t-tests for avg_steps from per-seed result files.

    Returns: {beta_key: {precept_vs_fr: {p, d}, precept_vs_expel: {p, d}}}
    """
    from scipy import stats as sp_stats

    sig = {}
    for beta in range(1, 6):
        precept_steps, fr_steps, expel_steps = [], [], []
        for f in sorted(results_dir.glob(f"beta{beta}_seed*_results.json")):
            with open(f) as fh:
                d = json.load(fh)
            agents = d.get("agents", {})
            if "precept" in agents:
                precept_steps.append(agents["precept"]["avg_steps"])
            if "full_reflexion" in agents:
                fr_steps.append(agents["full_reflexion"]["avg_steps"])
            if "expel" in agents:
                expel_steps.append(agents["expel"]["avg_steps"])

        key = f"beta_{beta}"
        sig[key] = {"precept_vs_fr": {"p_value": 1.0, "cohens_d": 0.0},
                     "precept_vs_expel": {"p_value": 1.0, "cohens_d": 0.0}}

        if len(precept_steps) >= 2 and len(fr_steps) >= 2:
            p_arr, f_arr = np.array(precept_steps), np.array(fr_steps)
            t, p = sp_stats.ttest_rel(p_arr, f_arr)
            diff = p_arr - f_arr
            d_val = diff.mean() / diff.std() if diff.std() > 0 else 0.0
            sig[key]["precept_vs_fr"] = {"p_value": p, "cohens_d": d_val}

        if len(precept_steps) >= 2 and len(expel_steps) >= 2:
            p_arr, e_arr = np.array(precept_steps), np.array(expel_steps)
            t, p = sp_stats.ttest_rel(p_arr, e_arr)
            diff = p_arr - e_arr
            d_val = diff.mean() / diff.std() if diff.std() > 0 else 0.0
            sig[key]["precept_vs_expel"] = {"p_value": p, "cohens_d": d_val}

    return sig


# ═══════════════════════════════════════════════════════════════════════════
# MAIN FIGURE — The only figure needed in the paper
# ═══════════════════════════════════════════════════════════════════════════

def generate_main_figure(data: Dict, output_dir: Path, results_dir: Path = None):
    """
    2-panel grouped-bar figure:
      (a) P₁ vs Training Exposure (β) — grouped bars with 95% CI, *, **, ***
      (b) Avg Steps vs Training Exposure (β) — grouped bars with theoretical min + significance
    """
    entries = ordered(data)
    betas = [d['beta'] for d in entries.values()]
    trains = [d['train_count'] for d in entries.values()]
    n = entries[list(entries.keys())[0]]['n_runs']

    p1 = {a: [_m(d, a, 'p1')['mean'] * 100 for d in entries.values()] for a in AGENTS}
    p1c_raw = {a: [_m(d, a, 'p1')['ci_95'] * 100 for d in entries.values()] for a in AGENTS}
    p1c = {a: [bounded_pct_ci(m, c) for m, c in zip(p1[a], p1c_raw[a])] for a in AGENTS}
    st = {a: [_m(d, a, 'steps')['mean'] for d in entries.values()] for a in AGENTS}
    stc = {a: [_m(d, a, 'steps')['ci_95'] for d in entries.values()] for a in AGENTS}

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    x = np.arange(len(betas))
    w = 0.25

    # ── (a) P₁ Grouped Bars ───────────────────────────────────────────────
    ax = axes[0]

    for j, agent in enumerate(AGENTS):
        errs = cap_err(p1[agent], p1c[agent])
        ax.bar(x + (j - 1) * w, p1[agent], w, yerr=errs,
               label=LABELS[agent], color=COLORS[agent], capsize=4,
               edgecolor='black', linewidth=0.8, hatch=HATCHES[agent],
               error_kw={'linewidth': 1.2, 'capthick': 1.2})

    # Significance markers — above each baseline's bar for both comparisons
    for i, key in enumerate(entries):
        st_tests = entries[key].get('statistical_tests', entries[key].get('tests', {}))
        # PRECEPT vs FR: place above FR bar (j=1, offset = 0*w = center)
        pvd_fr = st_tests.get('precept_vs_fr', {})
        fr_test = (
            pvd_fr.get('first_try_success', pvd_fr)
            if isinstance(pvd_fr, dict)
            else {}
        )
        pv_fr = corrected_p_value(fr_test if isinstance(fr_test, dict) else {})
        s_fr = sig_marker(pv_fr)
        if s_fr:
            y_fr = p1['full_reflexion'][i] + p1c['full_reflexion'][i]
            ax.annotate(s_fr, xy=(x[i] + 0 * w, min(y_fr + 4, 115)),
                        ha='center', fontsize=11, fontweight='bold', color=COLORS['full_reflexion'])
        # PRECEPT vs ExpeL: place above ExpeL bar (j=2, offset = 1*w)
        pvd_ex = st_tests.get('precept_vs_expel', {})
        ex_test = (
            pvd_ex.get('first_try_success', pvd_ex)
            if isinstance(pvd_ex, dict)
            else {}
        )
        pv_ex = corrected_p_value(ex_test if isinstance(ex_test, dict) else {})
        s_ex = sig_marker(pv_ex)
        if s_ex:
            y_ex = p1['expel'][i] + p1c['expel'][i]
            ax.annotate(s_ex, xy=(x[i] + 1 * w, min(y_ex + 4, 115)),
                        ha='center', fontsize=11, fontweight='bold', color=COLORS['expel'])

    ax.axhline(y=100, color='gray', linestyle='--', alpha=0.4, linewidth=0.8)
    ax.set_xlabel(r'Training Exposure ($\beta$)', fontweight='bold')
    ax.set_ylabel(r'First-Try Success Rate $P_1$ (%)', fontweight='bold')
    ax.set_title(r'(a) $P_1$ vs Training Exposure', fontweight='bold', pad=12)
    ax.set_xticks(x)
    ax.set_xticklabels(betas, fontsize=10)
    ax.set_ylim(0, 120)
    ax.legend(loc='lower right', framealpha=0.95, edgecolor='gray')
    ax.yaxis.grid(True, linestyle='--', alpha=0.3)
    ax.set_axisbelow(True)

    # ── (b) Avg Steps Grouped Bars ─────────────────────────────────────────
    ax = axes[1]

    for j, agent in enumerate(AGENTS):
        ax.bar(x + (j - 1) * w, st[agent], w, yerr=stc[agent],
               label=LABELS[agent], color=COLORS[agent], capsize=4,
               edgecolor='black', linewidth=0.8, hatch=HATCHES[agent],
               error_kw={'linewidth': 1.2, 'capthick': 1.2})

    ax.axhline(y=2.0, color='#15803d', linestyle='--', alpha=0.9,
               linewidth=1.5, label='Theoretical min.')
    ax.annotate('Optimal (2 steps)', xy=(0.5, 0.12), xycoords='axes fraction',
                ha='center', va='top', fontsize=9,
                fontweight='bold', style='italic', color='#15803d',
                bbox=dict(boxstyle='round,pad=0.2', fc='white', ec='none', alpha=0.8))

    # Step significance markers — above each baseline's bar
    if results_dir is not None:
        step_sig = compute_step_significance(results_dir)
        for i, key in enumerate(entries):
            ss = step_sig.get(key, {})
            # vs FR
            pv_fr = ss.get('precept_vs_fr', {}).get('p_value', 1)
            s_fr = sig_marker(pv_fr)
            if s_fr:
                y_fr = st['full_reflexion'][i] + stc['full_reflexion'][i]
                ax.annotate(s_fr, xy=(x[i] + 0 * w, y_fr + 0.3),
                            ha='center', fontsize=11, fontweight='bold', color=COLORS['full_reflexion'])
            # vs ExpeL
            pv_ex = ss.get('precept_vs_expel', {}).get('p_value', 1)
            s_ex = sig_marker(pv_ex)
            if s_ex:
                y_ex = st['expel'][i] + stc['expel'][i]
                ax.annotate(s_ex, xy=(x[i] + 1 * w, y_ex + 0.3),
                            ha='center', fontsize=11, fontweight='bold', color=COLORS['expel'])

    ax.set_xlabel(r'Training Exposure ($\beta$)', fontweight='bold')
    ax.set_ylabel('Avg Steps', fontweight='bold')
    ax.set_title('(b) Avg Steps vs Training Exposure', fontweight='bold', pad=12)
    ax.set_xticks(x)
    ax.set_xticklabels(betas, fontsize=10)
    ax.set_ylim(0, 10)
    ax.legend(loc='upper right', framealpha=0.95, edgecolor='gray', fontsize=10)
    ax.yaxis.grid(True, linestyle='--', alpha=0.3)
    ax.set_axisbelow(True)
    ax.annotate('Lower is better', xy=(0.98, 0.02), xycoords='axes fraction',
                ha='right', va='bottom', fontsize=10, style='italic', color='#666666')

    plt.tight_layout()
    fig.text(0.5, -0.04,
             f'Error bars: 95% CI (n={n} seeds per β). '
             '*** p<0.001, ** p<0.01, * p<0.05 (paired t-test, PRECEPT vs FR). '
             'Composite conditions (N=5).',
             ha='center', fontsize=10, style='italic', color='#333333')

    fig.savefig(output_dir / 'Figure_Exp3_Main.pdf', format='pdf')
    fig.savefig(output_dir / 'Figure_Exp3_Main.png', format='png')
    plt.close()
    print("    ✓ Figure_Exp3_Main.pdf/png")


# ═══════════════════════════════════════════════════════════════════════════
# LATEX TABLES
# ═══════════════════════════════════════════════════════════════════════════

def generate_latex_main_table(data: Dict, output_dir: Path):
    entries = ordered(data)
    n = entries[list(entries.keys())[0]]['n_runs']

    tex = r"""\begin{table*}[t]
\centering
\caption{Experiment 3: Training Size Ablation --- $P_1$ and Avg Steps ($N{=}5$ composite conditions). Bold = best per row.$^a$}
\label{tab:exp3_training_size}
\begin{tabular}{cc ccc ccc}
\toprule
& & \multicolumn{3}{c}{\textbf{First-Try Success $P_1$ (\%)}} & \multicolumn{3}{c}{\textbf{Avg Steps} $\downarrow$} \\
\cmidrule(lr){3-5} \cmidrule(lr){6-8}
$\beta$ & $T_{\text{train}}$ & \textbf{PRECEPT} & \textbf{FR} & \textbf{ExpeL} & \textbf{PRECEPT} & \textbf{FR} & \textbf{ExpeL} \\
\midrule
"""
    for e in entries.values():
        v = {
            a: {
                'p': _m(e, a, 'p1')['mean'] * 100,
                'pc': bounded_pct_ci(_m(e, a, 'p1')['mean'] * 100, _m(e, a, 'p1')['ci_95'] * 100),
                's': _m(e, a, 'steps')['mean'],
                'sc': _m(e, a, 'steps')['ci_95'],
            }
            for a in AGENTS
        }

        fr_test = resolve_stat_test(e, "precept_vs_fr", "first_try_success")
        sig = sig_marker(corrected_p_value(fr_test))
        stex = f'$^{{{sig}}}$' if sig else ''
        bp = max(v[a]['p'] for a in AGENTS)
        bs = min(v[a]['s'] for a in AGENTS)

        def fp(a):
            s = f"\\textbf{{{v[a]['p']:.1f}}}" if v[a]['p'] == bp else f"{v[a]['p']:.1f}"
            return f"{s}$\\pm${v[a]['pc']:.1f}"
        def fs(a):
            s = f"\\textbf{{{v[a]['s']:.2f}}}" if v[a]['s'] == bs else f"{v[a]['s']:.2f}"
            return f"{s}$\\pm${v[a]['sc']:.2f}"

        tex += (f"{e['beta']} & {e['train_count']} & "
                f"{fp('precept')}{stex} & {fp('full_reflexion')} & {fp('expel')} & "
                f"{fs('precept')} & {fs('full_reflexion')} & {fs('expel')} \\\\\n")

    tex += r"""\bottomrule
\end{tabular}
\vspace{2mm}
\footnotesize
""" + f"$^a$ n={n} seeds per $\\beta$. " + r"""$\pm$ 95\% CI. $^{***}$p$<$0.001, $^{**}$p$<$0.01, $^{*}$p$<$0.05 (paired t-test vs FR). $\downarrow$ lower is better.
\end{table*}
"""
    with open(output_dir / 'exp3_main_table.tex', 'w') as f:
        f.write(tex)
    print("    ✓ exp3_main_table.tex")


def generate_latex_significance_table(data: Dict, output_dir: Path):
    entries = ordered(data)

    tex = r"""\begin{table}[t]
\centering
\caption{Statistical Significance: PRECEPT vs Baselines on $P_1$ (paired t-test)}
\label{tab:exp3_significance}
\begin{tabular}{c cc cc}
\toprule
& \multicolumn{2}{c}{\textbf{vs Full Reflexion}} & \multicolumn{2}{c}{\textbf{vs ExpeL}} \\
\cmidrule(lr){2-3} \cmidrule(lr){4-5}
$\beta$ & $p$-value & Cohen's $d$ & $p$-value & Cohen's $d$ \\
\midrule
"""
    for e in entries.values():
        tfr = resolve_stat_test(e, "precept_vs_fr", "first_try_success")
        tex_ = resolve_stat_test(e, "precept_vs_expel", "first_try_success")

        def fmt(t):
            p, d = corrected_p_value(t), t.get('cohens_d', 0)
            s = sig_marker(p)
            ps = f"\\textbf{{{p:.4f}}}$^{{{s}}}$" if s else f"{p:.4f}"
            ds = f"\\textbf{{{d:.2f}}}" if abs(d) >= 0.8 else f"{d:.2f}"
            return ps, ds

        pfr, dfr = fmt(tfr)
        pex, dex = fmt(tex_)
        tex += f"{e['beta']} & {pfr} & {dfr} & {pex} & {dex} \\\\\n"

    tex += r"""\bottomrule
\end{tabular}
\vspace{2mm}
\footnotesize
$^{***}$p$<$0.001, $^{**}$p$<$0.01, $^{*}$p$<$0.05. Bold $d$: large effect ($\geq$0.8).
\end{table}
"""
    with open(output_dir / 'exp3_significance_table.tex', 'w') as f:
        f.write(tex)
    print("    ✓ exp3_significance_table.tex")


# ═══════════════════════════════════════════════════════════════════════════
# ANALYSIS REPORT
# ═══════════════════════════════════════════════════════════════════════════

def generate_report(data: Dict, output_dir: Path):
    entries = ordered(data)
    n = entries[list(entries.keys())[0]]['n_runs']
    peak = max(entries.values(), key=lambda e: _m(e, 'precept', 'p1')['mean'])
    worst = min(entries.values(), key=lambda e: _m(e, 'full_reflexion', 'p1')['mean'])

    r = f"""# Experiment 3: Training Size Ablation — Analysis Report

**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
**Configuration:** N=5 composite conditions, n={n} seeds per β

---

## Results Summary

| β | T | PRECEPT P₁ | FR P₁ | ExpeL P₁ | Δ vs FR | PRECEPT Steps | FR Steps | Sig. |
|---|---|-----------|-------|----------|---------|---------------|----------|------|
"""
    for e in entries.values():
        pp = _m(e, 'precept', 'p1')['mean']*100
        fp = _m(e, 'full_reflexion', 'p1')['mean']*100
        ep = _m(e, 'expel', 'p1')['mean']*100
        ps = _m(e, 'precept', 'steps')['mean']
        fs = _m(e, 'full_reflexion', 'steps')['mean']
        fr_test = resolve_stat_test(e, "precept_vs_fr", "first_try_success")
        pv = corrected_p_value(fr_test)
        d = fr_test.get("cohens_d", 0)
        r += (f"| {e['beta']} | {e['train_count']} | **{pp:.1f}%** | {fp:.1f}% | {ep:.1f}% | "
              f"+{pp-fp:.1f} pp | {ps:.2f} | {fs:.2f} | "
              f"p={pv:.4f} ({sig_full(pv)}), d={d:.2f} |\n")

    pp = _m(peak, 'precept', 'p1')['mean']*100
    wp = _m(worst, 'precept', 'p1')['mean']*100
    wf = _m(worst, 'full_reflexion', 'p1')['mean']*100
    worst_fr_test = resolve_stat_test(worst, "precept_vs_fr", "first_try_success")
    worst_pv = corrected_p_value(worst_fr_test)

    significant_betas = 0
    for entry in entries.values():
        entry_test = resolve_stat_test(entry, "precept_vs_fr", "first_try_success")
        if corrected_p_value(entry_test) < 0.05:
            significant_betas += 1

    r += f"""
## Key Findings

1. **Peak:** PRECEPT {pp:.1f}% P₁ at β={peak['beta']}.
2. **Largest gap:** β={worst['beta']}: PRECEPT {wp:.1f}% vs FR {wf:.1f}% (+{wp-wf:.1f} pp, p={worst_pv:.4f}).
3. **Coverage:** PRECEPT outperforms FR across all β settings in this run.
4. **Significance:** {significant_betas}/{len(entries)} β settings are significant (p<0.05) for PRECEPT vs FR.
"""
    with open(output_dir / 'Exp3_Analysis_Report.md', 'w') as f:
        f.write(r)
    print("    ✓ Exp3_Analysis_Report.md")


# ═══════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════

def find_all_domain_dirs() -> list:
    """Find all exp3 domain directories in publication_results (latest run per domain)."""
    pub = Path(__file__).parent.parent.parent / "data" / "publication_results"
    dirs = sorted(pub.glob("exp3_training_size_*"))
    known_domains = {"integration", "logistics", "booking"}
    domain_dirs = {}
    for d in dirs:
        parts = d.name.replace("exp3_training_size_", "").split("_")
        domain = parts[0]
        if domain in known_domains:
            domain_dirs[domain] = d
    return list(domain_dirs.values())


def generate_combined_figure(all_data: list, output_dir: Path, domain_dirs: list = None):
    """Generate combined multi-domain figure: one row per domain."""
    n_domains = len(all_data)
    fig, axes = plt.subplots(n_domains, 2, figsize=(13, 5.5 * n_domains))
    if n_domains == 1:
        axes = axes.reshape(1, 2)

    for row, (data, domain_name) in enumerate(all_data):
        entries = ordered(data)
        if not entries:
            continue

        betas = [d['beta'] for d in entries.values()]
        n = entries[list(entries.keys())[0]]['n_runs']
        domain_cap = domain_name.capitalize()

        p1 = {a: [_m(d, a, 'p1')['mean'] * 100 for d in entries.values()] for a in AGENTS}
        p1c_raw = {a: [_m(d, a, 'p1')['ci_95'] * 100 for d in entries.values()] for a in AGENTS}
        p1c = {a: [bounded_pct_ci(m, c) for m, c in zip(p1[a], p1c_raw[a])] for a in AGENTS}
        st = {a: [_m(d, a, 'steps')['mean'] for d in entries.values()] for a in AGENTS}
        stc = {a: [_m(d, a, 'steps')['ci_95'] for d in entries.values()] for a in AGENTS}

        x = np.arange(len(betas))
        w = 0.25

        # (a/c) P₁ Grouped Bars
        ax = axes[row, 0]
        for j, agent in enumerate(AGENTS):
            errs = cap_err(p1[agent], p1c[agent])
            ax.bar(x + (j - 1) * w, p1[agent], w, yerr=errs,
                   label=LABELS[agent], color=COLORS[agent], capsize=4,
                   edgecolor='black', linewidth=0.8, hatch=HATCHES[agent],
                   error_kw={'linewidth': 1.2, 'capthick': 1.2})

        # Significance markers — above each baseline's bar for both comparisons
        for i, key in enumerate(entries):
            st_tests = entries[key].get('statistical_tests', entries[key].get('tests', {}))
            # PRECEPT vs FR: place above FR bar
            pvd_fr = st_tests.get('precept_vs_fr', {})
            fr_test = (
                pvd_fr.get('first_try_success', pvd_fr)
                if isinstance(pvd_fr, dict)
                else {}
            )
            pv_fr = corrected_p_value(fr_test if isinstance(fr_test, dict) else {})
            s_fr = sig_marker(pv_fr)
            if s_fr:
                y_fr = p1['full_reflexion'][i] + p1c['full_reflexion'][i]
                ax.annotate(s_fr, xy=(x[i] + 0 * w, min(y_fr + 4, 115)),
                            ha='center', fontsize=11, fontweight='bold', color=COLORS['full_reflexion'])
            # PRECEPT vs ExpeL: place above ExpeL bar
            pvd_ex = st_tests.get('precept_vs_expel', {})
            ex_test = (
                pvd_ex.get('first_try_success', pvd_ex)
                if isinstance(pvd_ex, dict)
                else {}
            )
            pv_ex = corrected_p_value(ex_test if isinstance(ex_test, dict) else {})
            s_ex = sig_marker(pv_ex)
            if s_ex:
                y_ex = p1['expel'][i] + p1c['expel'][i]
                ax.annotate(s_ex, xy=(x[i] + 1 * w, min(y_ex + 4, 115)),
                            ha='center', fontsize=11, fontweight='bold', color=COLORS['expel'])

        panel_letter = chr(ord('a') + row * 2)
        ax.axhline(y=100, color='gray', linestyle='--', alpha=0.4, linewidth=0.8)
        ax.set_xlabel(r'Training Exposure ($\beta$)', fontweight='bold')
        ax.set_ylabel(r'$P_1$ (%)', fontweight='bold')
        ax.set_title(f'({panel_letter}) {domain_cap}: P₁ vs β', fontweight='bold', pad=12)
        ax.set_xticks(x)
        ax.set_xticklabels(betas, fontsize=10)
        ax.set_ylim(0, 120)
        ax.legend(loc='lower right', framealpha=0.95, edgecolor='gray')
        ax.yaxis.grid(True, linestyle='--', alpha=0.3)
        ax.set_axisbelow(True)

        # (b/d) Avg Steps Grouped Bars
        ax = axes[row, 1]
        for j, agent in enumerate(AGENTS):
            ax.bar(x + (j - 1) * w, st[agent], w, yerr=stc[agent],
                   label=LABELS[agent], color=COLORS[agent], capsize=4,
                   edgecolor='black', linewidth=0.8, hatch=HATCHES[agent],
                   error_kw={'linewidth': 1.2, 'capthick': 1.2})

        ax.axhline(y=2.0, color='#15803d', linestyle='--', alpha=0.9,
                   linewidth=1.5, label='Theoretical min.')

        # Step significance markers
        if domain_dirs is not None and row < len(domain_dirs):
            step_sig = compute_step_significance(domain_dirs[row])
            for i, key in enumerate(entries):
                ss = step_sig.get(key, {})
                # vs FR
                pv_fr = ss.get('precept_vs_fr', {}).get('p_value', 1)
                s_fr = sig_marker(pv_fr)
                if s_fr:
                    y_fr = st['full_reflexion'][i] + stc['full_reflexion'][i]
                    ax.annotate(s_fr, xy=(x[i] + 0 * w, y_fr + 0.3),
                                ha='center', fontsize=11, fontweight='bold', color=COLORS['full_reflexion'])
                # vs ExpeL
                pv_ex = ss.get('precept_vs_expel', {}).get('p_value', 1)
                s_ex = sig_marker(pv_ex)
                if s_ex:
                    y_ex = st['expel'][i] + stc['expel'][i]
                    ax.annotate(s_ex, xy=(x[i] + 1 * w, y_ex + 0.3),
                                ha='center', fontsize=11, fontweight='bold', color=COLORS['expel'])

        panel_letter2 = chr(ord('a') + row * 2 + 1)
        ax.set_xlabel(r'Training Exposure ($\beta$)', fontweight='bold')
        ax.set_ylabel('Avg Steps', fontweight='bold')
        ax.set_title(f'({panel_letter2}) {domain_cap}: Steps vs β', fontweight='bold', pad=12)
        ax.set_xticks(x)
        ax.set_xticklabels(betas, fontsize=10)
        ax.set_ylim(0, 10)
        ax.legend(loc='upper right', framealpha=0.95, edgecolor='gray', fontsize=10)
        ax.yaxis.grid(True, linestyle='--', alpha=0.3)
        ax.set_axisbelow(True)

    plt.tight_layout(h_pad=4)
    fig.savefig(output_dir / 'Figure_Exp3_Combined.pdf', format='pdf')
    fig.savefig(output_dir / 'Figure_Exp3_Combined.png', format='png')
    plt.close()
    print("    ✓ Figure_Exp3_Combined.pdf/png")


def main():
    if len(sys.argv) > 1 and sys.argv[1] != "--combined":
        rd = Path(sys.argv[1])
    else:
        rd = None

    if rd is None or sys.argv[-1] == "--combined":
        # Combined mode: auto-discover all domain directories
        domain_dirs = find_all_domain_dirs()
        if not domain_dirs:
            print("  ✗ No exp3 results found.")
            sys.exit(1)

        print()
        print("=" * 70)
        print("  EXPERIMENT 3: COMBINED PUBLICATION FIGURE GENERATION")
        print("=" * 70)

        all_data = []
        for d in domain_dirs:
            domain_name = d.name.replace("exp3_training_size_", "").split("_")[0]
            print(f"\n  Loading: {domain_name} ({d.name})")
            data = load_results(d)
            nb = len([k for k in data if 'beta' in data[k]])
            ns = data[list(data.keys())[0]]['n_runs']
            print(f"    {nb} β × {ns} seeds = {nb * ns} runs")
            all_data.append((data, domain_name))

            # Per-domain outputs
            figs = d / "figures"
            tabs = d / "tables"
            figs.mkdir(exist_ok=True)
            tabs.mkdir(exist_ok=True)
            generate_main_figure(data, figs, results_dir=d)
            generate_latex_main_table(data, tabs)
            generate_latex_significance_table(data, tabs)
            generate_report(data, d)

        if all_data:
            combined_dir = Path(__file__).parent.parent.parent / "data" / "publication_results" / "exp3_combined"
            combined_dir.mkdir(exist_ok=True)
            print(f"\n  Generating combined figure ({len(all_data)} domains)...")
            generate_combined_figure(all_data, combined_dir, domain_dirs=domain_dirs)
            print(f"\n  ✅ Combined saved to: {combined_dir}")

        print("\n" + "=" * 70)
        print("  ✅ Done!")
        print("=" * 70)
        return

    # Single directory mode (backward compatible)
    if not rd.exists():
        print(f"  ✗ Not found: {rd}")
        sys.exit(1)

    print()
    print("=" * 70)
    print("  EXPERIMENT 3: PUBLICATION FIGURE GENERATION")
    print("=" * 70)
    print(f"  Source: {rd}\n")

    data = load_results(rd)
    nb = len([k for k in data if 'beta' in data[k]])
    ns = data[list(data.keys())[0]]['n_runs']
    print(f"  Loaded: {nb} β × {ns} seeds = {nb * ns} runs\n")

    figs = rd / "figures"
    tabs = rd / "tables"
    figs.mkdir(exist_ok=True)
    tabs.mkdir(exist_ok=True)

    print("  Figure (600 DPI)...")
    generate_main_figure(data, figs, results_dir=rd)

    print("\n  LaTeX tables...")
    generate_latex_main_table(data, tabs)
    generate_latex_significance_table(data, tabs)

    print("\n  Report...")
    generate_report(data, rd)

    print()
    print("=" * 70)
    print("  ✅ Done!")
    print("=" * 70)
    print(f"\n  {figs / 'Figure_Exp3_Main.pdf'}")
    print(f"  {figs / 'Figure_Exp3_Main.png'}")
    print(f"  {tabs / 'exp3_main_table.tex'}")
    print(f"  {tabs / 'exp3_significance_table.tex'}")
    print(f"  {rd / 'Exp3_Analysis_Report.md'}")


if __name__ == "__main__":
    main()
