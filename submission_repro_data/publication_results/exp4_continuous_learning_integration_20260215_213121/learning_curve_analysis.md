# Experiment 4: Continuous Learning Analysis

**Domain:** integration
**Seeds:** [42, 123, 456, 789, 999, 2024, 3141, 1337, 8888]
**Successful Runs:** 9

## Parameters

- Training: 6 tasks (β=1)
- Testing: 24 tasks (4 encounters per key)
- Max Retries: 4
- Num Conditions: 5
- Unique Keys: 6

## Learning Curve: P₁ by Encounter

| Encounter | PRECEPT | ExpeL | Full Reflexion |
|-----------|---------|-------|----------------|
| 1 | 30.3% | 18.5% | 16.8% |
| 2 | 51.4% | 21.1% | 17.7% |
| 3 | 63.5% | 27.2% | 23.5% |
| 4 | 70.4% | 30.6% | 26.9% |

## Learning Improvement (1st → 4th)

**PRECEPT:** +40.1 pp (p=0.0004***)

**EXPEL:** +12.0 pp (p=0.0650)

**FULL_REFLEXION:** +10.1 pp (p=0.0911)

## Statistical Significance

### Final Encounter Comparison (PRECEPT vs Baselines)

| Comparison | p-value | Cohen's d | Significance |
|------------|---------|-----------|-------------|
| PRECEPT vs ExpeL | 0.0016 | 1.55 | ** |
| PRECEPT vs Full Reflexion | 0.0007 | 1.80 | *** |

### Per-Encounter Statistical Tests

| Encounter | PRECEPT vs ExpeL (p) | PRECEPT vs FR (p) |
|-----------|----------------------|-------------------|
| 1 | 0.0277* | 0.0141* |
| 2 | 0.0020** | 0.0028** |
| 3 | 0.0005*** | 0.0009*** |
| 4 | 0.0016** | 0.0007*** |

*Significance: \* p<0.05, \*\* p<0.01, \*\*\* p<0.001*

### Effect Size Interpretation (Cohen's d)

- Small: d = 0.2
- Medium: d = 0.5
- Large: d = 0.8
## Key Findings

✅ **PRECEPT shows significant cross-episode learning advantage**

