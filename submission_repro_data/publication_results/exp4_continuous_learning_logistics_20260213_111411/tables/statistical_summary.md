# Statistical Summary: Experiment 4 (Continuous Learning)

## Experiment Configuration

- **Domain**: logistics
- **Beta (β)**: 1
- **Training Tasks**: 4
- **Test Tasks**: 16
- **Encounters per Key**: 4
- **Max Retries**: 2
- **Num Conditions**: 5
- **Successful Runs**: 10
- **Seeds**: [42, 123, 456, 789, 999, 2024, 3141, 1337, 8888, 7777]

## Key Findings

1. **PRECEPT achieves the largest P₁ improvement**: +55.2pp (vs ExpeL +48.4pp, Full Reflexion +28.7pp)

2. **PRECEPT converges nearest to optimal**: 2.00 avg steps at 4th encounter (optimal = 2.00 steps)

3. **Strong significance vs Full Reflexion**: p=0.0019, Cohen's d=1.37 (large effect)

4. **PRECEPT vs ExpeL**: p=0.0239, Cohen's d=0.86 (significant)

## Per-Encounter Statistical Tests

| Encounter | PRECEPT vs ExpeL | | PRECEPT vs FR | |
|-----------|----------|---------|----------|---------|
| | p-value | d | p-value | d |
| 1st | 0.0556 | 0.69 | 0.1634 | 0.48 |
| 2nd | 0.0248* | 0.85 | 0.0031** | 1.26 |
| 3rd | 0.0051** | 1.16 | 0.0032** | 1.26 |
| 4th | 0.0239* | 0.86 | 0.0019** | 1.37 |

## Improvement Significance (Paired t-tests: 1st vs 4th encounter)

- **PRECEPT**: t=11.62, p=0.000001***, d=3.68
- **Full Reflexion**: t=4.11, p=0.002649**, d=1.30
- **ExpeL**: t=5.95, p=0.000214***, d=1.88

*Significance: \* p<0.05, \*\* p<0.01, \*\*\* p<0.001*