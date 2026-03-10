# Experiment 7: Rule Drift / Non-Stationary CSPs

**Domain:** logistics
**Seeds:** [42, 123, 456, 789, 999, 2024, 3141, 1337, 8888, 7777]
**Successful Runs:** 10

## Drift Setup
- Train hash seed: `0`
- Test hash seed: `1`

## Parameters
- Training: 12 tasks (β=3)
- Testing: 16 tasks (4 encounters per key)
- Num Conditions: 5
- Max Retries: 3
- Unique Keys: 4

## P₁ by Encounter

| Encounter | PRECEPT | ExpeL | Full Reflexion |
|-----------|---------|-------|----------------|
| 1 | 28.3% | 35.7% | 40.0% |
| 2 | 72.7% | 53.3% | 43.7% |
| 3 | 75.8% | 65.8% | 64.3% |
| 4 | 83.3% | 77.1% | 43.8% |

## Learning Improvement (1st → last encounter)

**PRECEPT:** +55.0 pp (p=1.0000)

**EXPEL:** +41.4 pp (p=1.0000)

**FULL_REFLEXION:** +3.7 pp (p=1.0000)

## Statistical Significance

### Final Encounter Comparison (PRECEPT vs Baselines)

| Comparison | p-value | Cohen's d | Significance |
|------------|---------|-----------|-------------|
| PRECEPT vs ExpeL | 0.7627 | 0.11 | n.s. |
| PRECEPT vs Full Reflexion | 0.0314 | 0.95 | * |

### Per-Encounter Statistical Tests

| Encounter | PRECEPT vs ExpeL (p) | PRECEPT vs FR (p) |
|-----------|----------------------|-------------------|
| 1 | 0.1177 | 0.1602 |
| 2 | 0.0375* | 0.0009*** |
| 3 | 0.3582 | 0.1507 |
| 4 | 0.7627 | 0.0314* |

*Significance: \* p<0.05, \*\* p<0.01, \*\*\* p<0.001*

### Effect Size Interpretation (Cohen's d)

- Small: d = 0.2
- Medium: d = 0.5
- Large: d = 0.8
