# Experiment 3: Training Size Ablation — Analysis Report

**Generated:** 2026-02-24 12:10:39
**Configuration:** N=5 composite conditions, n=10 seeds per β

---

## Results Summary

| β | T | PRECEPT P₁ | FR P₁ | ExpeL P₁ | Δ vs FR | PRECEPT Steps | FR Steps | Sig. |
|---|---|-----------|-------|----------|---------|---------------|----------|------|
| 1 | 4 | **77.5%** | 67.5% | 77.5% | +10.0 pp | 2.80 | 4.35 | p=0.5652 (ns), d=0.19 |
| 2 | 8 | **92.5%** | 75.0% | 72.5% | +17.5 pp | 2.33 | 4.10 | p=0.0095 (**), d=1.04 |
| 3 | 12 | **92.5%** | 60.0% | 72.5% | +32.5 pp | 2.30 | 5.00 | p=0.0002 (***), d=1.93 |
| 4 | 16 | **92.5%** | 55.0% | 60.0% | +37.5 pp | 2.25 | 5.05 | p=0.0003 (***), d=1.77 |
| 5 | 20 | **87.5%** | 72.5% | 77.5% | +15.0 pp | 2.52 | 4.75 | p=0.0811 (ns), d=0.62 |

## Key Findings

1. **Peak:** PRECEPT 92.5% P₁ at β=2.
2. **Largest gap:** β=4: PRECEPT 92.5% vs FR 55.0% (+37.5 pp, p=0.0003).
3. **Coverage:** PRECEPT outperforms FR across all β settings in this run.
4. **Significance:** 3/5 β settings are significant (p<0.05) for PRECEPT vs FR.
