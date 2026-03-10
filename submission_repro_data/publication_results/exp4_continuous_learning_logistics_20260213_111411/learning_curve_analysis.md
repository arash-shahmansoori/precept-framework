# Experiment 4: Continuous Learning Analysis

**Domain:** logistics
**Seeds:** [42, 123, 456, 789, 999, 2024, 3141, 1337, 8888, 7777]
**Successful Runs:** 10

## Parameters

- Training: 4 tasks (β=1)
- Testing: 16 tasks (4 encounters per key)
- Max Retries: 2
- Num Conditions: 5
- Unique Keys: 4

## Learning Curve: P₁ by Encounter

| Encounter | PRECEPT | ExpeL | Full Reflexion |
|-----------|---------|-------|----------------|
| 1 | 44.8% | 36.6% | 38.8% |
| 2 | 82.5% | 65.0% | 62.5% |
| 3 | 87.5% | 72.5% | 60.0% |
| 4 | 100.0% | 85.0% | 67.5% |

## Learning Improvement (1st → 4th)

**PRECEPT:** +55.2 pp (p=0.0000***)

**EXPEL:** +48.4 pp (p=0.0001***)

**FULL_REFLEXION:** +28.7 pp (p=0.0013**)

## Statistical Significance

### Final Encounter Comparison (PRECEPT vs Baselines)

| Comparison | p-value | Cohen's d | Significance |
|------------|---------|-----------|-------------|
| PRECEPT vs ExpeL | 0.0239 | 0.86 | * |
| PRECEPT vs Full Reflexion | 0.0019 | 1.37 | ** |

### Per-Encounter Statistical Tests

| Encounter | PRECEPT vs ExpeL (p) | PRECEPT vs FR (p) |
|-----------|----------------------|-------------------|
| 1 | 0.0556 | 0.1634 |
| 2 | 0.0248* | 0.0031** |
| 3 | 0.0051** | 0.0032** |
| 4 | 0.0239* | 0.0019** |

*Significance: \* p<0.05, \*\* p<0.01, \*\*\* p<0.001*

### Effect Size Interpretation (Cohen's d)

- Small: d = 0.2
- Medium: d = 0.5
- Large: d = 0.8
## Key Findings

✅ **PRECEPT shows modest cross-episode learning advantage**

