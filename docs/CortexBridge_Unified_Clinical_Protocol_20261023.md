# CortexBridge Unified MRI–Clinical Comparison Protocol

## Locked scope

- Task: ADNI NC/MCI/DE three-class development classification.
- Folds: subject-level `cross0`–`cross2`; seed `3407`.
- Model selection: validation MCC, up to 30 epochs, patience 4.
- Held-out test: never loaded (`runtime.evaluate_test=false`).
- Clinical features: the same 29 variables in every arm; MMSE is excluded.
- Missingness: training-fold mean imputation, training-fold standardization and one missing flag per variable.
- Student classifier: global MRI feature only; Student ROI and graph classification residuals are disabled.
- All arms are filtered to the Teacher-available imaging cohort, including CE-only arms.

## Arms

| ID | Training supervision | Student inference input | Locked question |
|---|---|---|---|
| E0 | Logistic regression | Clinical only | Scale-only reference |
| E1 | Ground-truth CE | MRI only | Imaging reference |
| E2 | Ground-truth CE | MRI + clinical | Ordinary fusion reference |
| E3 | CE + logit KD from MRI+clinical Teacher | MRI + clinical | Ordinary KD increment |
| E4 | CE + logit KD from MRI+surface+ribbon+clinical Teacher | MRI + clinical | Full CortexBridge |
| E5 | E4, but Student clinical vectors are subject-shuffled | MRI + shuffled clinical | Correct-pairing control |
| E6 | E4, but Teacher hard cross-geometry loss is disabled | MRI + clinical | Cross-geometry increment |
| E7 | No retraining; synthetic clinical masking of E4 | MRI + partial clinical | Missingness robustness |
| E8 | No retraining; E4 stratified by observed clinical fraction | MRI + real partial clinical | Real missingness audit |

E4 versus E6 isolates the cross-geometry module/loss while retaining the same surface, legacy volume and DKT graph Teacher branches. E4 versus E5 tests whether the deployed Student benefits from correctly paired clinical input. E3 versus E2 measures logit-KD beyond ordinary MRI–clinical fusion.

## Clinical variables

`age`, `education`, `apoe`, `animal`, `gds`, `lm_imm`, `lm_del`, ten `faq_*` items and twelve `npiq_*` items. These diagnosis-proximal variables make this a clinical-assisted classification setting and require a clinical-only comparator; the results must not be compared as MRI-only performance.

## Completion status (2026-10-26)

- E0–E8 are complete on development folds 0–2.
- Every neural run records `held_out_test_evaluated=false`; the held-out test was not loaded.
- The full repository test suite passes (`82 passed`).
- The primary performance configuration is E3 (MRI+clinical Student with ordinary MRI+clinical Teacher logit KD): MCC `0.69535`, AUC `0.93203`, ACC `0.80708`.
- E4 full CortexBridge obtains MCC `0.69226`, AUC `0.93199`, ACC `0.80555`. It does not outperform E3, and its E4−E6 cross-geometry increment is small with a confidence interval crossing zero.
- E5 subject-shuffle falls to MCC `0.29103`, supporting the necessity of correct patient–clinical pairing.
- Added clinical missingness causes strong degradation: E4 MCC changes from `0.69226` at 0% synthetic masking to `0.54637`, `0.36114`, `0.20103`, and `0.00000` at 25%, 50%, 75%, and 100% masking.

The detailed results and paired confidence intervals are in `outputs/vnext_unified_clinical_20261023/UNIFIED_CLINICAL_COMPARISON_REPORT.md`; the missingness analysis is in `outputs/vnext_unified_clinical_20261023/missingness_audit/MISSINGNESS_AUDIT.md`.
