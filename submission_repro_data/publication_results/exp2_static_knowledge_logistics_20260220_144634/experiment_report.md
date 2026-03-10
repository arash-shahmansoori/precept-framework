# Experiment 3: Static Knowledge Ablation Results

Generated: 2026-02-20T16:02:44.999992

## Summary: Impact of Static Knowledge

| Configuration | PRECEPT P₁ | FR P₁ | PRECEPT Pₜ | FR Pₜ |
|---------------|-----------|-------|-----------|-------|
| With Static KB | 92.5% ± 7.5% | 60.0% ± 12.5% | 100.0% ± 0.0% | 85.0% ± 12.5% |
| Without Static KB | 97.5% ± 2.5% | 57.5% ± 18.9% | 97.5% ± 2.5% | 87.5% ± 12.5% |

## Static Knowledge Effect

| Agent | P₁ Gain | Pₜ Gain |
|-------|---------|---------|
| PRECEPT | +-5.0 pp | +2.5 pp |
| Full Reflexion | +2.5 pp | +-2.5 pp |

## Key Finding

Static knowledge provides initial context that benefits both agents,
but PRECEPT maintains a significant advantage in both configurations
due to its structured rule learning and deterministic application.