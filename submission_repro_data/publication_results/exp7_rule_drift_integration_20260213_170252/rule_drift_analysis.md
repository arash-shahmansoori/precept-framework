# Experiment 7: Rule Drift / Non-Stationary CSPs

**Domain:** integration
**Seeds:** [42, 123, 456, 789, 999, 2024, 3141, 1337, 8888]
**Successful Runs:** 9

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
| 1 | 100.0% | 33.3% | 31.9% |
| 2 | 100.0% | 46.7% | 37.4% |
| 3 | 100.0% | 28.1% | 15.2% |
| 4 | 100.0% | 46.7% | 44.8% |

## Learning Improvement (1st → last encounter)

**PRECEPT:** +0.0 pp (p=nan)

**EXPEL:** +13.3 pp (p=0.1379)

**FULL_REFLEXION:** +13.0 pp (p=0.0441*)

## Statistical Significance

### Final Encounter Comparison (PRECEPT vs Baselines)

| Comparison | p-value | Cohen's d | Significance |
|------------|---------|-----------|-------------|
| PRECEPT vs ExpeL | 0.0001 | 2.36 | *** |
| PRECEPT vs Full Reflexion | 0.0001 | 2.26 | *** |

### Per-Encounter Statistical Tests

| Encounter | PRECEPT vs ExpeL (p) | PRECEPT vs FR (p) |
|-----------|----------------------|-------------------|
| 1 | 0.0000*** | 0.0000*** |
| 2 | 0.0001*** | 0.0000*** |
| 3 | 0.0002*** | 0.0000*** |
| 4 | 0.0001*** | 0.0001*** |

*Significance: \* p<0.05, \*\* p<0.01, \*\*\* p<0.001*

### Effect Size Interpretation (Cohen's d)

- Small: d = 0.2
- Medium: d = 0.5
- Large: d = 0.8
