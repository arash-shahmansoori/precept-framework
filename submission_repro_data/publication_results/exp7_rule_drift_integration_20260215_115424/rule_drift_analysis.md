# Experiment 7: Rule Drift / Non-Stationary CSPs

**Domain:** integration
**Seeds:** [42, 123, 456, 789, 999, 2024, 3141, 1337, 8888, 7777]
**Successful Runs:** 10

## Drift Setup
- Train hash seed: `0`
- Test hash seed: `1`

## Parameters
- Training: 18 tasks (β=3)
- Testing: 24 tasks (4 encounters per key)
- Num Conditions: 5
- Max Retries: 3
- Unique Keys: 6

## P₁ by Encounter

| Encounter | PRECEPT | ExpeL | Full Reflexion |
|-----------|---------|-------|----------------|
| 1 | 37.5% | 7.0% | 11.5% |
| 2 | 37.5% | 14.5% | 19.8% |
| 3 | 48.7% | 13.5% | 16.5% |
| 4 | 60.7% | 29.7% | 19.8% |

## Learning Improvement (1st → last encounter)

**PRECEPT:** +23.2 pp (p=0.0037**)

**EXPEL:** +22.7 pp (p=0.0015**)

**FULL_REFLEXION:** +8.3 pp (p=0.1791)

## Statistical Significance

### Final Encounter Comparison (PRECEPT vs Baselines)

| Comparison | p-value | Cohen's d | Significance |
|------------|---------|-----------|-------------|
| PRECEPT vs ExpeL | 0.0058 | 1.14 | ** |
| PRECEPT vs Full Reflexion | 0.0013 | 1.45 | ** |

### Per-Encounter Statistical Tests

| Encounter | PRECEPT vs ExpeL (p) | PRECEPT vs FR (p) |
|-----------|----------------------|-------------------|
| 1 | 0.0028** | 0.0165* |
| 2 | 0.0112* | 0.0072** |
| 3 | 0.0080** | 0.0231* |
| 4 | 0.0058** | 0.0013** |

*Significance: \* p<0.05, \*\* p<0.01, \*\*\* p<0.001*

### Effect Size Interpretation (Cohen's d)

- Small: d = 0.2
- Medium: d = 0.5
- Large: d = 0.8
