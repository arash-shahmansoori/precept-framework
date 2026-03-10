# Experiment 3: Static Knowledge Ablation Results

Generated: 2026-02-20T14:46:34.190562

## Summary: Impact of Static Knowledge

| Configuration | PRECEPT P₁ | FR P₁ | PRECEPT Pₜ | FR Pₜ |
|---------------|-----------|-------|-----------|-------|
| With Static KB | 80.0% ± 9.4% | 38.3% ± 8.0% | 81.7% ± 8.8% | 43.3% ± 8.3% |
| Without Static KB | 85.0% ± 8.8% | 38.3% ± 9.8% | 88.3% ± 9.8% | 48.3% ± 3.8% |

## Static Knowledge Effect

| Agent | P₁ Gain | Pₜ Gain |
|-------|---------|---------|
| PRECEPT | +-5.0 pp | +-6.7 pp |
| Full Reflexion | +-0.0 pp | +-5.0 pp |

## Key Finding

Static knowledge provides initial context that benefits both agents,
but PRECEPT maintains a significant advantage in both configurations
due to its structured rule learning and deterministic application.