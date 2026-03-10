# Submission Reproducibility Data Manifest

This directory is a curated, submission-ready subset of `data/` containing only artifacts used to produce the paper's reported results, tables, and figures.

## Directory Structure

- `submission_repro_data/publication_results/`
- `submission_repro_data/static_knowledge/`

## One-Command Repro

From project root:

- `bash scripts/run_submission_repro.sh`

This runs a lock-synced reproducibility check, verifies figure hashes, and writes a reconstructed bundle to:

- `submission_repro_data/regenerated_paper_artifacts/`

## Paper Artifact Mapping

### Experiment 1 (Main Domain Comparison)

- Final paper figure/table artifacts:
  - `publication_results/exp1_main_comparison_combined/`
- Source run traces/logs used to construct domain rows:
  - `publication_results/exp1_main_comparison_20260209_141144/` (Integration)
  - `publication_results/exp1_main_comparison_20260208_163115/` (Booking)
  - `publication_results/exp1_main_comparison_20260208_115951/` (Logistics source run)

### Experiment 2 (Compositional Semantic Generalization)

- Final paper figure/table artifacts:
  - `publication_results/exp6_final_publication/`
- Source run traces/logs:
  - `publication_results/exp6_compositional_20260213_133517/`

### Experiment 3 (Training Size Ablation, beta effect)

- Final paper figure artifacts:
  - `publication_results/exp3_combined/`
- Source run traces/logs:
  - `publication_results/exp3_training_size_integration_20260214_114545/`
  - `publication_results/exp3_training_size_logistics_20260214_164544/`

### Experiment 4 (Continuous Learning)

- Final paper figure artifacts:
  - `publication_results/exp4_combined/`
- Source run traces/logs:
  - `publication_results/exp4_continuous_learning_integration_20260215_213121/` (9-seed integration setting in paper)
  - `publication_results/exp4_continuous_learning_logistics_20260213_111411/`

### Experiment 5 (Rule Persistence and Retrieval Fidelity)

- Final paper figure artifacts:
  - `publication_results/exp5_persistence_combined/`
- Source run traces/logs for paper table/curve values:
  - `publication_results/exp7_rule_drift_integration_20260213_170252/` (9 runs)
  - `publication_results/exp7_rule_drift_logistics_20260213_231005/` (10 runs)
- Verification report included:
  - `publication_results/EXP5_RULE_PERSISTENCE_VERIFICATION_REPORT.md`

Note: the paper labels this as Experiment 5 persistence, while the source run directories above use the `exp7_rule_drift_*` naming convention.

### Experiment 6 (Static Knowledge Ablation, Type I Conflict)

- Final paper figure artifacts:
  - `publication_results/exp2_static_knowledge_combined/`
- Source run traces/logs:
  - `publication_results/exp2_static_knowledge_integration_20260220_123951/`
  - `publication_results/exp2_static_knowledge_logistics_20260220_144634/`
- Static knowledge data used by these runs:
  - `static_knowledge/`

### Experiment 7 (Rule Drift Adaptation, Type II Conflict)

- Final paper figure artifacts:
  - `publication_results/exp6_drift_combined/`
- Source run traces/logs for paper drift table/curve values:
  - `publication_results/exp7_rule_drift_integration_20260215_115424/`
  - `publication_results/exp7_rule_drift_logistics_20260215_145340/`

### Experiment 8 (COMPASS Ablation)

- Source run traces/logs and aggregate outputs:
  - `publication_results/exp8_compass_ablation_20260228_141040/`

### Experiment 9 (COMPASS Stress / OOD Semantic Follow-up)

- Source run traces/logs and aggregate outputs used for the OOD semantic follow-up:
  - `publication_results/exp9_compass_stress_sem_only_20260305_075820/`

## Figure Identity Check (Paper vs Data)

The following paper figures correspond exactly to copied artifacts in this package:

- `fig1_domain_comparison.png` <- `exp1_main_comparison_combined/figures/fig1_domain_comparison.png`
- `Figure_Exp6_Main.png` <- `exp6_final_publication/figures/Figure_Exp6_Main.png`
- `Figure_Exp3_Combined.png` <- `exp3_combined/Figure_Exp3_Combined.png`
- `Figure_Exp4_Combined.png` <- `exp4_combined/Figure_Exp4_Combined.png`
- `Figure_Exp7_Combined.png` <- `exp5_persistence_combined/Figure_Exp7_Combined.png`
- `fig2_sk_integration.png` <- `exp2_static_knowledge_combined/fig2_sk_integration.png`
- `fig2_sk_logistics.png` <- `exp2_static_knowledge_combined/fig2_sk_logistics.png`
- `Figure_Exp7_Drift_Bar.png` <- `exp6_drift_combined/Figure_Exp7_Drift_Bar.png`

## Scope Note

This package intentionally excludes unrelated exploratory, quick-validation, and superseded runs from `data/publication_results/`.
