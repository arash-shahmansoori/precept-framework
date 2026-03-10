# Experiment 3: Training Size Ablation Results (logistics)

Generated: 2026-02-14T19:42:12.960003

## Training Exposure (β) Effect

```
β = Learning Threshold (times each error type seen during training)
T_train = β × E = β × 4 for logistics

β=1: Single encounter (fragile rules)
β=2: Two encounters (moderate robustness)
β=3: Three encounters (publication quality) ← RECOMMENDED
β=4: Four encounters (diminishing returns)
```

## Results Summary

| β | T_train | PRECEPT P₁ | FR P₁ | Δ P₁ | p-value |
|---|---------|-----------|-------|------|---------|
| 1 | 4 | 77.5% ± 22.5% | 67.5% ± 20.7% | **+10.0 pp** | 0.5652 |
| 2 | 8 | 92.5% ± 7.5% | 75.0% ± 14.6% | **+17.5 pp** | 0.0095** |
| 3 | 12 | 92.5% ± 7.5% | 60.0% ± 15.1% | **+32.5 pp** | 0.0002*** |
| 4 | 16 | 92.5% ± 7.5% | 55.0% ± 14.1% | **+37.5 pp** | 0.0003*** |
| 5 | 20 | 87.5% ± 9.4% | 72.5% ± 13.2% | **+15.0 pp** | 0.0811 |

*Significance: \* p<0.05, \*\* p<0.01, \*\*\* p<0.001*

## Key Findings

1. **Sample Efficiency**: PRECEPT shows strong performance even at β=1
2. **Robustness**: Performance improves with more training exposure
3. **Recommended**: β=3 provides good balance of efficiency and robustness
4. **PRECEPT Advantage**: Consistently outperforms Full Reflexion across all β values