# Paper-Numbered Reproduction Index

This folder provides a reviewer-facing mapping from paper experiment numbers to the exact artifact directories inside `submission_repro_data/publication_results/`, plus the corresponding script entry points.

Use this together with:

- `submission_repro_data/REPRODUCIBILITY_MANIFEST.md`
- `submission_repro_data/FIGURE_SHA256.txt`
- `submission_repro_data/environment/ENVIRONMENT_LOCK.md`

One-command runner:

- `bash scripts/run_submission_repro.sh`

## Exp1 - Main Domain Comparison

- Final paper artifacts:
  - `../publication_results/exp1_main_comparison_combined/`
- Source traces/logs:
  - `../publication_results/exp1_main_comparison_20260209_141144/` (Integration)
  - `../publication_results/exp1_main_comparison_20260208_163115/` (Booking)
  - `../publication_results/exp1_main_comparison_20260208_115951/` (Logistics)
- Script:
  - `scripts/create_results/generate_exp1_main_comparison_results.py`

## Exp2 - Compositional Semantic Generalization

- Final paper artifacts:
  - `../publication_results/exp6_final_publication/`
- Source traces/logs:
  - `../publication_results/exp6_compositional_20260213_133517/`
- Scripts:
  - `scripts/generate_exp6_main_figure.py`
  - `scripts/generate_exp6_figures.py`

## Exp3 - Training Size Ablation

- Final paper artifacts:
  - `../publication_results/exp3_combined/`
- Source traces/logs:
  - `../publication_results/exp3_training_size_integration_20260214_114545/`
  - `../publication_results/exp3_training_size_logistics_20260214_164544/`
- Script:
  - `scripts/create_results/generate_exp3_training_size_results.py`

## Exp4 - Continuous Learning

- Final paper artifacts:
  - `../publication_results/exp4_combined/`
- Source traces/logs:
  - `../publication_results/exp4_continuous_learning_integration_20260215_213121/`
  - `../publication_results/exp4_continuous_learning_logistics_20260213_111411/`
- Script:
  - `scripts/create_results/generate_exp4_continuous_learning_results.py`

## Exp5 - Rule Persistence and Retrieval Fidelity

- Final paper artifacts:
  - `../publication_results/exp5_persistence_combined/`
- Source traces/logs for paper values:
  - `../publication_results/exp7_rule_drift_integration_20260213_170252/`
  - `../publication_results/exp7_rule_drift_logistics_20260213_231005/`
- Verification note:
  - `../publication_results/EXP5_RULE_PERSISTENCE_VERIFICATION_REPORT.md`
- Script family:
  - `scripts/run_exp7_rule_drift.py`
  - `scripts/generate_exp7_main_figure.py`

## Exp6 - Static Knowledge Ablation (Type I)

- Final paper artifacts:
  - `../publication_results/exp2_static_knowledge_combined/`
- Source traces/logs:
  - `../publication_results/exp2_static_knowledge_integration_20260220_123951/`
  - `../publication_results/exp2_static_knowledge_logistics_20260220_144634/`
- Script:
  - `scripts/create_results/generate_exp2_static_knowledge_results.py`
- Supporting data:
  - `../static_knowledge/`

## Exp7 - Rule Drift Adaptation (Type II)

- Final paper artifacts:
  - `../publication_results/exp6_drift_combined/`
- Source traces/logs for paper drift values:
  - `../publication_results/exp7_rule_drift_integration_20260215_115424/`
  - `../publication_results/exp7_rule_drift_logistics_20260215_145340/`
- Script family:
  - `scripts/run_exp7_rule_drift.py`
  - `scripts/generate_exp7_main_figure.py`

## Exp8 - COMPASS Ablation

- Source traces/logs and aggregate output:
  - `../publication_results/exp8_compass_ablation_20260228_141040/`
- Script:
  - `scripts/run_exp8_compass_ablation.py`

## Exp9 - COMPASS Stress / OOD Semantic Follow-Up

- Source traces/logs and aggregate output:
  - `../publication_results/exp9_compass_stress_sem_only_20260305_075820/`
- Script:
  - `scripts/run_exp9_compass_stress.py`

## Reviewer Guidance

- For exact paper artifact identity, rely on the already generated figure/table files in the mapped "Final paper artifacts" directories and verify hashes in `FIGURE_SHA256.txt`.
- For trace-level provenance and reruns, use the mapped "Source traces/logs" directories with the listed scripts.
