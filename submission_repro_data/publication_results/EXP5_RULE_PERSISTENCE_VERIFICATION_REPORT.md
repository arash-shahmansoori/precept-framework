# Experiment 5 (Rule Persistence) Table Verification Report

**Date:** 2025-02-21  
**Paper table:** PRECEPT_PAPER.tex lines 1344-1352 (Table 13)  
**Data sources:**  
- Integration: `exp7_rule_drift_integration_20260213_170252/rule_drift_results.json`  
- Logistics: `exp7_rule_drift_logistics_20260213_231005/rule_drift_results.json`

---

## 1. Mean P₁ Values: Paper vs. Raw Data

### Integration (N=9 seeds)

| Metric | Agent | Paper | rule_drift_results.json | Match? |
|--------|-------|-------|--------------------------|---------|
| Enc.1 P₁ | PRECEPT | 100.0% | 100.0% | ✓ |
| Enc.1 P₁ | FR | 31.9% | 31.9% | ✓ |
| Enc.1 P₁ | ExpeL | 33.3% | 33.3% | ✓ |
| Enc.4 P₁ | PRECEPT | 100.0% | 100.0% | ✓ |
| Enc.4 P₁ | FR | 44.8% | 44.8% | ✓ |
| Enc.4 P₁ | ExpeL | 46.7% | 46.7% | ✓ |
| Overall Pₜ | PRECEPT | 100.0% | 100.0% | ✓ |
| Overall Pₜ | FR | 33.7% | 33.7% | ✓ |
| Overall Pₜ | ExpeL | 35.6% | 35.6% | ✓ |
| Avg Steps | PRECEPT | 2.00 | 2.00 | ✓ |
| Avg Steps | FR | 4.69 | 4.69 | ✓ |
| Avg Steps | ExpeL | 4.62 | 4.62 | ✓ |

**Overall Pₜ source:** Encounter 1 Pₜ (first-encounter eventual success rate).  
**Avg Steps source:** Encounter 1 steps for Integration.

### Logistics (N=10 seeds)

| Metric | Agent | Paper | rule_drift_results.json | Match? |
|--------|-------|-------|--------------------------|---------|
| Enc.1 P₁ | PRECEPT | 88.5% | 88.5% | ✓ |
| Enc.1 P₁ | FR | 58.0% | 58.0% | ✓ |
| Enc.1 P₁ | ExpeL | 68.5% | 68.5% | ✓ |
| Enc.4 P₁ | PRECEPT | 100.0% | 100.0% | ✓ |
| Enc.4 P₁ | FR | 70.0% | 70.0% | ✓ |
| Enc.4 P₁ | ExpeL | 92.5% | 92.5% | ✓ |
| Overall Pₜ | PRECEPT | 100.0% | 100.0% (enc4) | ✓ |
| Overall Pₜ | FR | 85.0% | 85.0% (enc4) | ✓ |
| Overall Pₜ | ExpeL | 97.5% | 97.5% (enc4) | ✓ |
| Avg Steps | PRECEPT | 2.00 | 2.00 | ✓ |
| Avg Steps | FR | 2.90 | 2.90 | ✓ |
| Avg Steps | ExpeL | 2.20 | 2.20 | ✓ |

**Overall Pₜ source:** Encounter 4 Pₜ for Logistics (steady state).  
**Avg Steps source:** Encounter 4 steps for Logistics.

**Verdict:** All mean values in the paper match the raw data.

---

## 2. CI Bounds: Paper vs. Raw 95% CI

The paper reports "mean ± 95% CI". The raw data provides both `p1_std` and `p1_ci_95`. The paper uses `p1_ci_95` (not std).

### Integration (N=9, t≈2.306, CI ≈ std × 0.769)

| Metric | Agent | Paper ± | Raw p1_ci_95 | Match? |
|--------|-------|---------|--------------|--------|
| Enc.1 P₁ | FR | ±9.5 | 9.5 | ✓ |
| Enc.1 P₁ | ExpeL | ±11.9 | 11.9 | ✓ |
| Enc.4 P₁ | FR | ±18.8 | 18.8 | ✓ |
| Enc.4 P₁ | ExpeL | ±17.4 | 17.4 | ✓ |
| Overall Pₜ | FR | ±10.6 | 10.6 | ✓ |
| Overall Pₜ | ExpeL | ±11.4 | 11.4 | ✓ |
| Avg Steps | FR | ±0.39 | 0.39 | ✓ |
| Avg Steps | ExpeL | ±0.46 | 0.46 | ✓ |

### Logistics (N=10, t≈2.262, CI ≈ std × 0.715)

| Metric | Agent | Paper ± | Raw (bounded) | Match? |
|--------|-------|---------|---------------|--------|
| Enc.1 P₁ | PRECEPT | ±8.8 | 8.8 | ✓ |
| Enc.1 P₁ | FR | ±16.3 | 16.3 | ✓ |
| Enc.1 P₁ | ExpeL | ±14.6 | 14.6 | ✓ |
| Enc.4 P₁ | FR | ±29.0 | 29.0 | ✓ |
| Enc.4 P₁ | ExpeL | ±7.5 | 8.6→7.5 (bounded) | ✓ |
| Overall Pₜ | FR | ±15.0 | 24.1→15.0 (bounded) | ✓ |
| Overall Pₜ | ExpeL | ±2.5 | 5.7→2.5 (bounded) | ✓ |
| Avg Steps | FR | ±0.97 | 0.97 | ✓ |
| Avg Steps | ExpeL | ±0.25 | 0.25 | ✓ |

**Bounded CI:** The paper uses `bounded_pct_ci(mean, ci)` so that mean±CI stays in [0, 100]. For ExpeL Overall Pₜ at 97.5%, raw CI 5.7 is capped to 2.5 (100−97.5). For FR Overall Pₜ at 85%, raw CI 24.1 is capped to 15 (100−85).

**Verdict:** All CI values in the paper are consistent with the raw data (with bounded CI where applicable).

---

## 3. Generated Tables vs. Paper

The generated `Table_Exp7_Statistics.tex` files differ from the paper in several ways:

### 3.1 Column order
- **Generated:** PRECEPT | ExpeL | Full Reflexion  
- **Paper:** PRECEPT | Full Reflexion | ExpeL  

### 3.2 Error bars
- **Generated:** Uses `p1_std` (standard deviation) passed through `bounded_pct_ci`.  
- **Paper:** Uses `p1_ci_95` (95% confidence interval).  

So the generated tables show ±std, while the paper shows ±95% CI.

### 3.3 Metrics
- **Generated:** Only P₁ by encounter and Improvement (1st→4th).  
- **Paper:** Also includes Overall Pₜ and Avg Steps.  

Overall Pₜ and Avg Steps are not produced by `generate_exp7_main_figure.py`; they are taken from `rule_drift_results.json` (enc1 pt/steps for Integration, enc4 for Logistics) and added manually to the paper table.

### 3.4 Integration table N
- **Generated table caption:** "N=10 seeds"  
- **Integration data:** `n_runs=9`  
- **Logistics data:** `n_runs=10`  

The Integration `Table_Exp7_Statistics.tex` caption says N=10 but the integration run has 9 seeds. This is a caption error in the generated table.

### 3.5 Mean differences (Integration)
The Integration `Table_Exp7_Statistics.tex` shows:
- 1st Encounter: PRECEPT 100±0, ExpeL 35.0±15.7, FR 30.8±12.8  

From `rule_drift_results.json`:
- ExpeL enc1: 33.3±15.6 (std)  
- FR enc1: 31.9±12.4 (std)  

So the generated table has ExpeL 35.0 vs raw 33.3, and FR 30.8 vs raw 31.9. These may come from an older run or a different table-generation path.

---

## 4. Summary

| Check | Result |
|-------|--------|
| Mean P₁ (Enc.1, Enc.4) | ✓ All match |
| Mean Overall Pₜ | ✓ All match |
| Mean Avg Steps | ✓ All match |
| 95% CI values | ✓ All consistent with raw data |
| Bounded CI (Logistics) | ✓ Correctly applied |
| Generated table vs paper | ⚠ Different: uses std not CI, different column order, no Overall Pₜ/Avg Steps, Integration N=10 vs 9 |

---

## 5. Data Source Summary

- **Enc.1 P₁, Enc.4 P₁:** `learning_curves.{agent}.encounter_{1|4}.p1_mean`, `p1_ci_95`  
- **Overall Pₜ (Integration):** `learning_curves.{agent}.encounter_1.pt_mean`, `pt_ci_95`  
- **Overall Pₜ (Logistics):** `learning_curves.{agent}.encounter_4.pt_mean`, `pt_ci_95`  
- **Avg Steps (Integration):** `learning_curves.{agent}.encounter_1.steps_mean`, `steps_ci_95`  
- **Avg Steps (Logistics):** `learning_curves.{agent}.encounter_4.steps_mean`, `steps_ci_95`  

No `experiment_report.md` was found in the exp7 rule drift experiment directories.
