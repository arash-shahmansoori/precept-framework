# Publication Result Generation Scripts

This directory contains scripts to generate publication-quality tables and figures from experiment results.

## Experiment Structure

| Exp | Script | Description |
|-----|--------|-------------|
| 1 | `run_exp1_main_comparison.py` | Main 6-domain comparison |
| 2 | `run_exp2_static_knowledge_ablation.py` | Static knowledge ablation |
| 3 | `run_exp3_training_size_ablation.py` | Training size (β) ablation |
| 4 | `run_exp4_continuous_learning.py` | Continuous/cross-episode learning |
| 5 | `run_exp5_model_ablation.py` | Model/embedding ablation |

## Output Summary (Unique, Non-Overlapping)

Each script produces **unique** outputs with no redundancy:

| Script | Tables | Figures | Key Insight |
|--------|--------|---------|-------------|
| `generate_exp1_main_comparison_results.py` | Table 1 (6-domain comparison) | Fig 1-2 (domain bars + summary) | PRECEPT wins across ALL domains |
| `generate_exp2_static_knowledge_results.py` | Table 2 (SK ablation) | Fig 2 (with/without SK bars) | SK helps both, PRECEPT still leads |
| `generate_exp3_training_size_results.py` | Table 3 (training size) | Fig 3 (learning curve) | PRECEPT sample-efficient |
| `generate_exp4_continuous_learning_results.py` | Table 4 (cross-episode) | Fig 4 (encounter curve) | PRECEPT learns during testing |
| `generate_exp5_model_ablation_results.py` | Table 5 (model ablation) | Fig 5 (model comparison) | Powerful models don't help baselines |
| `generate_consolidated_results.py` | Master summary table | Fig 6 (5-panel overview) | Publication master summary |

## Output Uniqueness Guarantee

Each figure and table provides unique value:

### Figures
- **Figure 1**: Domain-by-domain bar comparison (P₁) - shows consistency across 6 domains
- **Figure 2**: Static knowledge ablation (grouped bars) - shows KB contribution
- **Figure 3**: Learning curve (β effect) - shows sample efficiency
- **Figure 4**: Cross-episode learning (encounter curve) - shows in-testing improvement
- **Figure 5**: Model/embedding ablation (grouped bars) - shows architectural advantage
- **Figure 6**: (Consolidated) Publication-ready 5-panel overview

### Tables
- **Table 1**: Full 6-domain comparison with all metrics (P₁, Pₜ, steps, p-values)
- **Table 2**: Static knowledge impact with effect sizes
- **Table 3**: Training size (β=1,2,3,4) comparison with learning thresholds
- **Table 4**: Cross-episode learning by encounter number
- **Table 5**: Model/embedding ablation comparison

## Usage

### Generate Results for Individual Experiments

```bash
# After running experiments, generate results:
python scripts/create_results/generate_exp1_main_comparison_results.py \
    data/publication_results/exp1_main_comparison_<timestamp>

python scripts/create_results/generate_exp2_static_knowledge_results.py \
    data/publication_results/exp2_static_knowledge_<timestamp>

python scripts/create_results/generate_exp3_training_size_results.py \
    data/publication_results/exp3_training_size_<timestamp>

python scripts/create_results/generate_exp4_continuous_learning_results.py \
    data/publication_results/exp4_continuous_learning_<timestamp>

python scripts/create_results/generate_exp5_model_ablation_results.py \
    data/publication_results/exp5_model_ablation_<timestamp>
```

### Generate Consolidated Summary

```bash
# Generate master summary from all experiments
python scripts/create_results/generate_consolidated_results.py \
    --exp1 data/publication_results/exp1_main_comparison_<timestamp> \
    --exp2 data/publication_results/exp2_static_knowledge_<timestamp> \
    --exp3 data/publication_results/exp3_training_size_<timestamp> \
    --exp4 data/publication_results/exp4_continuous_learning_<timestamp> \
    --exp5 data/publication_results/exp5_model_ablation_<timestamp> \
    --output data/publication_results/consolidated_<timestamp>
```

## Output Directory Structure

```
data/publication_results/<experiment>_<timestamp>/
├── aggregated_results.json     # Raw aggregated data
├── experiment_config.json      # Experiment configuration
├── experiment_report.md        # Auto-generated markdown report
├── tables/
│   ├── *.tex                   # LaTeX tables
│   └── *.md                    # Markdown tables
└── figures/
    ├── *.png                   # High-res PNG (300 DPI)
    └── *.pdf                   # Vector PDF for LaTeX
```

## Data Cleaning (Experiment Contamination Avoidance)

**Critical**: Before each experiment run, the following data is automatically cleaned:

```
data/chroma_precept/            # PRECEPT's learned knowledge
data/chroma_static_knowledge/   # Static knowledge embeddings
data/chroma_expel/              # ExpeL's learned insights
data/precept_*.json             # All persisted JSON state files
```

This ensures:
1. **Fair comparison**: Each seed starts from clean state
2. **No contamination**: Previous runs don't affect new results
3. **Reproducibility**: Same seed produces same results

The cleaning is done automatically by `clean_data_directory()` in each experiment script, called **before every individual seed run**.

## Visual Style Guidelines

All figures follow publication standards:
- **Font**: Serif (Times New Roman / DejaVu Serif)
- **Font sizes**: 11pt body, 12pt labels, 13pt titles
- **DPI**: 300 for saved figures
- **Colors**:
  - PRECEPT: `#2E86AB` (blue)
  - Full Reflexion: `#A23B72` (magenta)
  - ExpeL: `#6B8E23` (olive green)
  - Theoretical: `#F18F01` (orange)
- **Error bars**: 95% confidence intervals

## Statistical Requirements

All comparisons include:
- **95% Confidence Intervals**: Using t-distribution (n-1 df)
- **Paired t-tests**: For PRECEPT vs baseline comparison
- **Cohen's d**: Effect size interpretation
- **p-values**: With significance markers (* p<0.05, ** p<0.01, *** p<0.001)

## Dependencies

```bash
pip install matplotlib numpy scipy
```

## Files in This Directory

```
scripts/create_results/
├── README.md                                    # This file
├── generate_exp1_main_comparison_results.py    # Exp 1: Main domain comparison
├── generate_exp2_static_knowledge_results.py   # Exp 2: Static knowledge ablation
├── generate_exp3_training_size_results.py      # Exp 3: Training size ablation
├── generate_exp4_continuous_learning_results.py # Exp 4: Continuous learning
├── generate_exp5_model_ablation_results.py     # Exp 5: Model/embedding ablation
└── generate_consolidated_results.py            # Master summary combining all
```
