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
| 1 | 88.5% | 68.5% | 58.0% |
| 2 | 90.5% | 78.0% | 75.5% |
| 3 | 93.0% | 84.0% | 84.5% |
| 4 | 100.0% | 92.5% | 70.0% |

## Learning Improvement (1st → last encounter)

**PRECEPT:** +11.5 pp (p=0.0079**)

**EXPEL:** +24.0 pp (p=0.0047**)

**FULL_REFLEXION:** +12.0 pp (p=0.1966)

## Statistical Significance

### Final Encounter Comparison (PRECEPT vs Baselines)

| Comparison | p-value | Cohen's d | Significance |
|------------|---------|-----------|-------------|
| PRECEPT vs ExpeL | 0.0811 | 0.62 | n.s. |
| PRECEPT vs Full Reflexion | 0.0438 | 0.74 | * |

### Per-Encounter Statistical Tests

| Encounter | PRECEPT vs ExpeL (p) | PRECEPT vs FR (p) |
|-----------|----------------------|-------------------|
| 1 | 0.0163* | 0.0003*** |
| 2 | 0.0905 | 0.1161 |
| 3 | 0.1708 | 0.1848 |
| 4 | 0.0811 | 0.0438* |

*Significance: \* p<0.05, \*\* p<0.01, \*\*\* p<0.001*

### Effect Size Interpretation (Cohen's d)

- Small: d = 0.2
- Medium: d = 0.5
- Large: d = 0.8
