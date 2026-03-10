# Experiment 8: COMPASS Prompt Evolution Ablation

**Domain:** integration | **E:** 6 | **N:** 5 | **β:** 3

## Research Question

Does COMPASS's outer-loop prompt evolution (Pareto selection, smart rollouts, GEPA mutation) contribute measurable value beyond the deterministic rule compilation that occurs independently?

## Configurations

| Config | COMPASS Evolution | Rules in Prompt |
|--------|-------------------|------------------|
| PRECEPT (full) | ✓ | ✓ |
| PRECEPT (no COMPASS) | ✗ | ✓ |
| PRECEPT (no rules in prompt) | ✓ | ✗ |
| PRECEPT (base prompt only) | ✗ | ✗ |

## Results

| Config | P₁ | Pₜ | Steps | GEPA Triggers |
|--------|-----|-----|-------|---------------|
| PRECEPT (full) | 76.7±8.3% | 80.0±7.5% | 2.82 | 10.5 |
| PRECEPT (no COMPASS) | 80.0±7.5% | 85.0±6.8% | 2.74 | 5.7 |
| PRECEPT (no rules in prompt) | 88.3±11.3% | 93.3±8.3% | 2.38 | 11.5 |
| PRECEPT (base prompt only) | 90.0±6.2% | 90.0±6.2% | 2.33 | 1.0 |

## Statistical Comparisons (vs Full PRECEPT)

| Comparison | Δ P₁ | t | p | Cohen's d |
|------------|-------|---|---|----------|
| vs no_compass | -3.3pp | -0.80 | 0.4435 | -0.30 |
| vs no_rules | -11.7pp | -2.09 | 0.0663 | -0.84 |
| vs base_only | -13.3pp | -3.21 | 0.0107 | -1.30 |
