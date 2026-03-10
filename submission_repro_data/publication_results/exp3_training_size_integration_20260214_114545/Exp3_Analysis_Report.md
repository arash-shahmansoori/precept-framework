# Experiment 3: Training Size Ablation — Analysis Report

**Generated:** 2026-02-24 12:10:34
**Configuration:** N=5 composite conditions, n=10 seeds per β

---

## Results Summary

| β | T | PRECEPT P₁ | FR P₁ | ExpeL P₁ | Δ vs FR | PRECEPT Steps | FR Steps | Sig. |
|---|---|-----------|-------|----------|---------|---------------|----------|------|
| 1 | 6 | **63.3%** | 35.0% | 41.7% | +28.3 pp | 3.58 | 8.13 | p=0.1941 (ns), d=0.44 |
| 2 | 12 | **98.3%** | 45.0% | 36.7% | +53.3 pp | 2.08 | 7.33 | p=0.0000 (***), d=5.06 |
| 3 | 18 | **100.0%** | 38.3% | 48.3% | +61.7 pp | 2.00 | 7.63 | p=0.0000 (***), d=3.90 |
| 4 | 24 | **100.0%** | 43.3% | 46.7% | +56.7 pp | 2.00 | 7.27 | p=0.0000 (***), d=4.86 |
| 5 | 30 | **98.3%** | 46.7% | 48.3% | +51.7 pp | 2.02 | 7.07 | p=0.0000 (***), d=5.46 |

## Key Findings

1. **Peak:** PRECEPT 100.0% P₁ at β=3.
2. **Largest gap:** β=1: PRECEPT 63.3% vs FR 35.0% (+28.3 pp, p=0.1941).
3. **Coverage:** PRECEPT outperforms FR across all β settings in this run.
4. **Significance:** 4/5 β settings are significant (p<0.05) for PRECEPT vs FR.
