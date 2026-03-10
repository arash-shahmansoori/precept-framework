# Experiment 1: Main Domain Comparison Results

Generated: 2026-02-08T15:54:26.635899

## Summary: PRECEPT vs Baselines

| Domain | PRECEPT P₁ | FR P₁ | ExpeL P₁ | Δ vs FR | Δ vs ExpeL | p-value (FR) |
|--------|-----------|-------|----------|---------|------------|--------------|
| Finance | 85.0% ± 13.1% | 70.0% ± 16.7% | 100.0% ± 0.0% | **+15.0 pp** | **+-15.0 pp** | 0.0946 |
| Logistics | 95.0% ± 5.0% | 57.5% ± 14.7% | 90.0% ± 9.2% | **+37.5 pp** | **+5.0 pp** | 0.0001*** |
| Coding | 74.3% ± 14.0% | 97.1% ± 2.9% | 100.0% ± 0.0% | **+-22.9 pp** | **+-25.7 pp** | 0.0152* |
| Devops | 100.0% ± 0.0% | 100.0% ± 0.0% | 100.0% ± 0.0% | **+0.0 pp** | **+0.0 pp** | nan |
| Integration | 76.7% ± 8.3% | 20.0% ± 9.4% | 26.7% ± 12.8% | **+56.7 pp** | **+50.0 pp** | 0.0000*** |

*Significance: \* p<0.05, \*\* p<0.01, \*\*\* p<0.001*

*N = 10 independent runs per domain*

## Detailed Results


### Finance

| Metric | PRECEPT | Full Reflexion | ExpeL | Δ vs FR | Δ vs ExpeL |
|--------|---------|----------------|-------|---------|------------|
| P₁ (First-Try) | 85.0% ± 13.1% | 70.0% ± 16.7% | 100.0% ± 0.0% | +15.0 pp | +0.0 pp |
| Pₜ (Overall) | 96.7% ± 3.3% | 98.3% ± 1.7% | 100.0% ± 0.0% | +-1.7 pp | +0.0 pp |
| Avg Steps | 2.37 ± 0.5 | 4.1 ± 0.8 | 2.0 ± 0.0 | 1.73 saved | -0.37 saved |

### Logistics

| Metric | PRECEPT | Full Reflexion | ExpeL | Δ vs FR | Δ vs ExpeL |
|--------|---------|----------------|-------|---------|------------|
| P₁ (First-Try) | 95.0% ± 5.0% | 57.5% ± 14.7% | 90.0% ± 9.2% | +37.5 pp | +0.0 pp |
| Pₜ (Overall) | 100.0% ± 0.0% | 80.0% ± 11.3% | 97.5% ± 2.5% | +20.0 pp | +0.0 pp |
| Avg Steps | 2.1 ± 0.2 | 5.25 ± 0.8 | 2.45 ± 0.5 | 3.15 saved | 0.35 saved |

### Coding

| Metric | PRECEPT | Full Reflexion | ExpeL | Δ vs FR | Δ vs ExpeL |
|--------|---------|----------------|-------|---------|------------|
| P₁ (First-Try) | 74.3% ± 14.0% | 97.1% ± 2.9% | 100.0% ± 0.0% | +-22.9 pp | +0.0 pp |
| Pₜ (Overall) | 97.1% ± 2.9% | 100.0% ± 0.0% | 100.0% ± 0.0% | +-2.9 pp | +0.0 pp |
| Avg Steps | 2.66 ± 0.4 | 3.06 ± 0.1 | 2.0 ± 0.0 | 0.40 saved | -0.66 saved |

### Devops

| Metric | PRECEPT | Full Reflexion | ExpeL | Δ vs FR | Δ vs ExpeL |
|--------|---------|----------------|-------|---------|------------|
| P₁ (First-Try) | 100.0% ± 0.0% | 100.0% ± 0.0% | 100.0% ± 0.0% | +0.0 pp | +0.0 pp |
| Pₜ (Overall) | 100.0% ± 0.0% | 100.0% ± 0.0% | 100.0% ± 0.0% | +0.0 pp | +0.0 pp |
| Avg Steps | 2.0 ± 0.0 | 3.0 ± 0.0 | 2.0 ± 0.0 | 1.00 saved | 0.00 saved |

### Integration

| Metric | PRECEPT | Full Reflexion | ExpeL | Δ vs FR | Δ vs ExpeL |
|--------|---------|----------------|-------|---------|------------|
| P₁ (First-Try) | 76.7% ± 8.3% | 20.0% ± 9.4% | 26.7% ± 12.8% | +56.7 pp | +0.0 pp |
| Pₜ (Overall) | 78.3% ± 8.0% | 26.7% ± 11.5% | 31.7% ± 8.8% | +51.7 pp | +0.0 pp |
| Avg Steps | 2.79 ± 0.3 | 9.03 ± 0.8 | 7.67 ± 0.9 | 6.25 saved | 4.88 saved |