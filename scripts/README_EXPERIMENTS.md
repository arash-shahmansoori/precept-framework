# PRECEPT Experiment Scripts (Paper-Aligned)

This document is the script-level index for the experiments reported in the manuscript.
It supersedes older "five core experiments" wording and aligns with the current **9-experiment** paper mapping.

For reviewer-facing artifact provenance (final paper artifacts + timestamped source traces), also see:

- `submission_repro_data/paper_experiment_sources/README.md`
- `submission_repro_data/REPRODUCIBILITY_MANIFEST.md`
- Appendix mapping table (`tab:artifact_map`) in `paper/*/PRECEPT_PAPER.tex`

## Paper Experiment to Script Mapping

| Paper Exp | Study | Primary Script(s) | Notes |
|---|---|---|---|
| 1 | Main domain comparison | `scripts/run_exp1_main_comparison.py` | Uses composite-condition setup (`NUM_CONDITIONS = 5`) in current script. |
| 2 | Compositional semantic generalization | `scripts/run_exp6_compositional_generalization.py` | Evaluates atomic-to-composite transfer. |
| 3 | Training size ablation (beta effect) | `scripts/run_exp3_training_size_ablation.py` | Varies training coverage (`beta`) and compares learning quality. |
| 4 | Continuous learning | `scripts/run_exp4_continuous_learning.py` | Sequential encounter setting to measure in-test adaptation. |
| 5 | Rule persistence and retrieval fidelity | `scripts/run_exp7_rule_drift.py` | Persistence mode: same hash seed for train and test (`train_hash_seed == test_hash_seed`). |
| 6 | Static knowledge ablation (Type I conflict) | `scripts/run_exp2_static_knowledge_ablation.py` | With/without static knowledge and conflict handling comparison. |
| 7 | Rule drift adaptation (Type II conflict) | `scripts/run_exp7_rule_drift.py` | Drift mode: different hash seeds (`train_hash_seed != test_hash_seed`). |
| 8 | COMPASS ablation | `scripts/run_exp8_compass_ablation.py` | Isolates effect of COMPASS components. |
| 9 | COMPASS stress / OOD semantic follow-up | `scripts/run_exp9_compass_stress.py` | Stress-tests COMPASS under harder semantic settings. |

## Quick Usage

```bash
# Exp1
uv run scripts/run_exp1_main_comparison.py

# Exp2
uv run scripts/run_exp6_compositional_generalization.py

# Exp3
uv run scripts/run_exp3_training_size_ablation.py

# Exp4
uv run scripts/run_exp4_continuous_learning.py

# Exp5 (persistence mode)
uv run scripts/run_exp7_rule_drift.py --train-hash-seed 0 --test-hash-seed 0

# Exp6
uv run scripts/run_exp2_static_knowledge_ablation.py

# Exp7 (drift mode)
uv run scripts/run_exp7_rule_drift.py --train-hash-seed 0 --test-hash-seed 1

# Exp8
uv run scripts/run_exp8_compass_ablation.py

# Exp9
uv run scripts/run_exp9_compass_stress.py
```

## Consistency Notes

- **Exp1 condition complexity:** current `run_exp1_main_comparison.py` is configured with `NUM_CONDITIONS = 5` (not a single-condition setup).
- **Exp5/Exp7 shared driver:** both are produced by `run_exp7_rule_drift.py`; seed pairing determines whether the run is persistence (no drift) or drift.
- **Exact paper trace provenance:** use `submission_repro_data/paper_experiment_sources/README.md` for timestamped source directories and curated final artifact locations.

## One-Command Repro Check

Use the locked runner that validates environment lock + figure hashes and reconstructs a reviewer bundle:

```bash
bash scripts/run_submission_repro.sh
```
