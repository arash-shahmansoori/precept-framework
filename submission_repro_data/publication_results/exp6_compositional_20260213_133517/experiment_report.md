# Experiment 6: Compositional Generalization Results

Generated: 2026-02-13T16:51:50.880194

## Purpose

Test PRECEPT's ability to generalize to NOVEL combinations of conditions that were never seen during training. This demonstrates **Atomic Constraint Stacking**.

## Key Mechanism

```
ATOMIC CONSTRAINT STACKING:

1. DECOMPOSITION: Break A+B+C into [A, B, C]
2. RETRIEVAL: Get precept for each: A→X, B→Y, C→Z
3. STACKING: Inject all constraints into LLM context
4. SYNTHESIS: LLM composes solution satisfying all

Result: O(2^N) combinations from N atomic precepts
```

## Configurations Tested

| Config | Train Conditions | Test Conditions | Gap |
|--------|-----------------|-----------------|-----|
| logistics_2way | 1 | 2 | +1 |
| logistics_3way | 1 | 3 | +2 |
| integration_2way | 1 | 2 | +1 |
| integration_3way | 1 | 3 | +2 |

## Results Summary

| Config | PRECEPT P₁ | FR P₁ | ExpeL P₁ | Δ vs FR | Δ vs ExpeL |
|--------|-----------|-------|----------|---------|------------|
| logistics_2way | **100.0%** ± 0.0% | 57.3% ± 11.6% | 47.7% ± 14.6% | **+42.7 pp** | **+52.3 pp** |
| logistics_3way | **78.0%** ± 9.4% | 43.0% ± 13.9% | 40.0% ± 10.1% | **+35.0 pp** | **+38.0 pp** |
| integration_2way | **41.7%** ± 33.4% | 18.1% ± 16.4% | 11.2% ± 11.2% | **+23.6 pp** | **+30.5 pp** |
| integration_3way | **49.0%** ± 25.7% | 17.0% ± 11.5% | 25.0% ± 21.6% | **+32.0 pp** | **+24.0 pp** |

## Key Findings

1. **PRECEPT achieves compositional generalization**: Can solve novel combinations

2. **Baselines fail on novel combinations**: They require exact pattern matching

3. **Atomic precepts enable O(2^N) coverage**: N learned precepts → 2^N combinations


## Theoretical Claim

```
"PRECEPT achieves O(1) Compositional Adaptation.
Instead of training on the combinatorics of all possible
failure modes (A+B, A+C, B+C...), it learns the atomic
constraints once and utilizes the LLM's inherent logical
reasoning to synthesize composite solutions at runtime."
```