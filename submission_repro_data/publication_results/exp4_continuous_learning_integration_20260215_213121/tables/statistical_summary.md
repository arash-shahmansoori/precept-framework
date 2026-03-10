# Statistical Summary: Experiment 4 (Continuous Learning)

## Experiment Configuration

- **Domain**: integration
- **Beta (β)**: 1
- **Training Tasks**: 6
- **Test Tasks**: 24
- **Encounters per Key**: 4
- **Max Retries**: 4
- **Num Conditions**: 5
- **Successful Runs**: 9
- **Seeds**: [42, 123, 456, 789, 999, 2024, 3141, 1337, 8888]

## Key Findings

1. **PRECEPT achieves the largest P₁ improvement**: +40.1pp (vs ExpeL +12.0pp, Full Reflexion +10.1pp)

2. **PRECEPT converges nearest to optimal**: 2.33 avg steps at 4th encounter (optimal = 2.00 steps)

3. **Strong significance vs Full Reflexion**: p=0.0007, Cohen's d=1.80 (large effect)

4. **PRECEPT vs ExpeL**: p=0.0016, Cohen's d=1.55 (significant)

## Per-Encounter Statistical Tests

| Encounter | PRECEPT vs ExpeL | | PRECEPT vs FR | |
|-----------|----------|---------|----------|---------|
| | p-value | d | p-value | d |
| 1st | 0.0277* | 0.90 | 0.0141* | 1.04 |
| 2nd | 0.0020** | 1.50 | 0.0028** | 1.42 |
| 3rd | 0.0005*** | 1.85 | 0.0009*** | 1.70 |
| 4th | 0.0016** | 1.55 | 0.0007*** | 1.80 |

## Improvement Significance (Paired t-tests: 1st vs 4th encounter)

- **PRECEPT**: t=5.25, p=0.000771***, d=1.75
- **Full Reflexion**: t=1.46, p=0.182186, d=0.49
- **ExpeL**: t=1.69, p=0.129955, d=0.56

*Significance: \* p<0.05, \*\* p<0.01, \*\*\* p<0.001*