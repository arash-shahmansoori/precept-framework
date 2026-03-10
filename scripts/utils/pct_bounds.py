"""Shared helpers for bounding percentage CI so mean±CI never exceeds [0, 100]."""

from typing import List, Tuple


def bounded_pct_ci(mean_pct: float, ci_pct: float,
                   lower: float = 0.0, upper: float = 100.0) -> float:
    """Shrink symmetric CI so that mean±CI stays within [lower, upper]."""
    return max(0.0, min(ci_pct, mean_pct - lower, upper - mean_pct))


def bounded_pct_from_frac(metric: dict) -> Tuple[float, float]:
    """Convert a fractional {mean, ci_95} dict to bounded percentage (mean%, ci%)."""
    mean_pct = metric.get("mean", 0) * 100
    ci_pct = metric.get("ci_95", 0) * 100
    return mean_pct, bounded_pct_ci(mean_pct, ci_pct)


def cap_pct_errorbars(means_pct: List[float],
                      cis_pct: List[float]) -> List[List[float]]:
    """Create asymmetric [lower, upper] error-bar lists bounded to [0, 100]."""
    lower = [max(0.0, min(ci, m)) for m, ci in zip(means_pct, cis_pct)]
    upper = [max(0.0, min(ci, 100.0 - m)) for m, ci in zip(means_pct, cis_pct)]
    return [lower, upper]


def fmt_pct(metric: dict) -> str:
    """Format a fractional metric as 'XX.X% ± YY.Y%' with bounded CI."""
    m, c = bounded_pct_from_frac(metric)
    return f"{m:.1f}% ± {c:.1f}%"


def fmt_pct_bold(metric: dict) -> str:
    """Same as fmt_pct but with markdown bold on the mean."""
    m, c = bounded_pct_from_frac(metric)
    return f"**{m:.1f}%** ± {c:.1f}%"


def fmt_pct_latex(metric: dict, bold: bool = False) -> str:
    r"""Format as 'XX.X \pm YY.Y' for LaTeX with bounded CI."""
    m, c = bounded_pct_from_frac(metric)
    val = f"\\textbf{{{m:.1f}}}" if bold else f"{m:.1f}"
    return f"{val} $\\pm$ {c:.1f}"
