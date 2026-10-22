# CortexBridge S4 Clinical-Assisted Ablation Report

Date: 2026-10-20  
Status: **COMPLETE — three development folds, no held-out test access**  
Task: three-class NC / MCI / DE classification  
Primary selection metric: multiclass MCC

## Decision

The method should now be reported as two explicitly separated inference settings:

1. **CortexBridge S4-MRI** remains the strict MRI-only core model. It is the appropriate main result when the scientific claim is privileged cross-geometry/graph distillation into an image-only Student.
2. **CortexBridge S4-C** is a clinical-assisted extension for settings where clinical variables are available at inference. The selected development candidate is **C2: S4 + demographics/APOE + GDS/NPI-Q gated fusion**.

C2 is selected instead of the numerically highest C4 because it excludes MMSE, memory scores, verbal fluency and FAQ. It improves on both S4-MRI and the exactly matched clinical-only comparator, giving evidence that MRI and the lower-risk clinical variables contribute complementary information.

C3 and C4 are retained only as **diagnosis-proximal clinical upper bounds**. Their high absolute scores cannot be presented as evidence that the imaging method improved: C3 is slightly worse than its clinical-only comparator, and C4 has no clear MCC/accuracy advantage while its AUC is lower than clinical-only.

No experiment in this report contains MMSE.

## Architecture evaluated

The Teacher and distillation targets are unchanged from locked S4:

```text
Teacher (training only):
MRI legacy ROI path
+ auxiliary hard Surface↔Ribbon correspondence
+ true-DKT DiseaseAwareROIGraph

Student imaging path:
MRI -> shared 3D encoder -> legacy ROI tokens
    -> topology-seeded Student ROI graph -> MRI global feature
    -> CE + logit KD + ROI graph-state KD
```

For C1–C4 only, the deployed Student additionally receives the selected clinical vector. Each clinical variable is represented by its training-fold-normalized value and a missingness indicator. The gated fusion is:

```text
z_clin = ClinicalEncoder([normalized values, missing flags])
gate   = sigmoid(W [z_mri ; z_clin])
z_fuse = LayerNorm(z_mri + gate ⊙ z_clin)
logits = Classifier(z_fuse)
```

The Teacher remains clinical-off (`model.use_clinical=false`). Only the Student uses clinical data (`model.student_use_clinical=true`). Therefore C1–C4 require their clinical inputs at inference and must not be called MRI-only models.

## Experimental arms

| ID | Student inference input | Clinical variables | Risk interpretation |
|---|---|---:|---|
| C0 | MRI only | 0 | locked S4-MRI reference |
| C1 | MRI + gated clinical | 4 | age, gender, education, APOE; low-risk background |
| **C2** | **MRI + gated clinical** | **17** | **C1 + GDS + 12 NPI-Q items; selected S4-C** |
| C3 | MRI + gated clinical | 13 | animal fluency, LM immediate/delayed, 10 FAQ items; diagnosis-proximal upper bound |
| C4 | MRI + gated clinical | 29 | all available non-MMSE variables; diagnosis-proximal upper bound |

All arms use the same folds, subjects, MRI preprocessing, S4 architecture, Teacher checkpoints, initialization policy, optimizer, seed, MCC selection rule, `warmup_epochs=0`, encoder/inherited learning rate `1e-4`, and KD weights (`logit=0.5`, `ROI-state=1.0`, graph-relation/prototype=0). C1–C4 change only the Student clinical input and gated fusion.

## Three-fold classification results

All values are mean ± sample SD over development folds 0/1/2. SEN is macro one-vs-rest sensitivity and is numerically identical to macro Recall/balanced accuracy; it is shown once as an independent quantity. SPE is macro one-vs-rest specificity, and AUC is macro one-vs-rest AUC from probabilities.

| ID | MCC | ACC | SEN / Recall | SPE | AUC | Macro-F1 |
|---|---:|---:|---:|---:|---:|---:|
| C0 S4-MRI | 0.31227 ± 0.04766 | 0.53872 ± 0.03044 | 0.55227 ± 0.03053 | 0.75617 ± 0.01551 | 0.73894 ± 0.02805 | 0.51942 ± 0.04210 |
| C1 background | 0.31072 ± 0.06202 | 0.55625 ± 0.04137 | 0.57449 ± 0.06187 | 0.76351 ± 0.02232 | 0.73241 ± 0.03756 | 0.55518 ± 0.05359 |
| **C2 safe-neuro** | **0.44281 ± 0.02970** | **0.64965 ± 0.02969** | **0.61660 ± 0.00245** | **0.80842 ± 0.00995** | **0.80494 ± 0.02215** | **0.61906 ± 0.01124** |
| C3 cognition/FAQ | 0.66540 ± 0.01653 | 0.78743 ± 0.01077 | 0.79262 ± 0.01730 | 0.88670 ± 0.00453 | 0.92079 ± 0.00632 | 0.78105 ± 0.01211 |
| C4 full29, no MMSE | 0.69241 ± 0.02588 | 0.80208 ± 0.02002 | 0.80593 ± 0.01266 | 0.89617 ± 0.00738 | 0.92944 ± 0.00770 | 0.79499 ± 0.02215 |

## Paired comparison with S4-MRI

The fold delta is the mean of three paired fold differences. Confidence intervals are 2,000-replicate subject-level paired bootstrap intervals on the pooled 1,378 development predictions. McNemar tests use paired correctness.

| Candidate vs C0 | Mean fold ΔMCC | Pooled ΔMCC (95% CI) | Pooled ΔAUC (95% CI) | Pooled ΔACC (95% CI) | McNemar p |
|---|---:|---:|---:|---:|---:|
| C1 | -0.00156 | -0.00114 [-0.04715, 0.04411] | -0.00572 [-0.02597, 0.01348] | +0.01742 [-0.01234, 0.04644] | 0.2793 |
| **C2** | **+0.13054** | **+0.13140 [0.08592, 0.17725]** | **+0.06803 [0.04904, 0.08759]** | **+0.11103 [0.08200, 0.14078]** | **3.86×10⁻13** |
| C3 | +0.35313 | +0.35216 [0.30306, 0.39976] | +0.17955 [0.15868, 0.20001] | +0.24891 [0.21626, 0.27939] | 1.45×10⁻47 |
| C4 | +0.38014 | +0.37980 [0.33391, 0.42649] | +0.19187 [0.17209, 0.21128] | +0.26343 [0.23367, 0.29318] | 1.54×10⁻60 |

C1 supplies no stable gain over MRI-only S4. C2 improves every reported macro metric and its paired MCC/AUC/ACC intervals exclude zero. The much larger C3/C4 gains require comparison with clinical-only controls before interpretation.

## Matched clinical-only controls

Clinical-only logistic regression is fit independently in each training fold with fold-local imputation/scaling and evaluated on the exact model validation IDs. The source audit contained four extra fold-1 and three extra fold-2 clinical rows without a usable model imaging/cache sample; these seven rows are excluded from the matched comparisons, leaving exactly the same 454/457/467 samples as the neural models.

| Clinical-only input | MCC | ACC | SEN / Recall | SPE | AUC | Macro-F1 |
|---|---:|---:|---:|---:|---:|---:|
| Background | 0.20362 ± 0.02792 | 0.44642 ± 0.01477 | 0.49533 ± 0.02639 | 0.73054 ± 0.00893 | 0.65094 ± 0.02091 | 0.43820 ± 0.01358 |
| Safe-neuro | 0.34029 ± 0.07669 | 0.56058 ± 0.04937 | 0.57935 ± 0.05227 | 0.77666 ± 0.02619 | 0.75419 ± 0.02749 | 0.54503 ± 0.04491 |
| Cognition/FAQ | 0.67988 ± 0.01228 | 0.79398 ± 0.00795 | 0.80754 ± 0.01382 | 0.89316 ± 0.00392 | 0.93319 ± 0.00261 | 0.78887 ± 0.00988 |
| Full29, no MMSE | 0.67749 ± 0.01663 | 0.79316 ± 0.01168 | 0.80456 ± 0.02554 | 0.89265 ± 0.00629 | 0.93781 ± 0.00423 | 0.78686 ± 0.01379 |

| Fusion vs matched clinical-only | Pooled ΔMCC (95% CI) | Pooled ΔAUC (95% CI) | Pooled ΔACC (95% CI) | McNemar p |
|---|---:|---:|---:|---:|
| C1 vs background | +0.10741 [0.05986, 0.15416] | +0.08058 [0.05720, 0.10165] | +0.10958 [0.07910, 0.14006] | 5.36×10⁻12 |
| **C2 vs safe-neuro** | **+0.10350 [0.06346, 0.14313]** | **+0.05009 [0.03378, 0.06581]** | **+0.08926 [0.06241, 0.11468]** | **6.98×10⁻11** |
| C3 vs cognition/FAQ | -0.01651 [-0.04284, 0.00838] | -0.01813 [-0.02440, -0.01237] | -0.00653 [-0.02395, 0.00943] | 0.4976 |
| C4 vs full29 | +0.01370 [-0.01161, 0.03818] | -0.01061 [-0.01590, -0.00488] | +0.00871 [-0.00728, 0.02540] | 0.3491 |

This control changes the interpretation:

- C1 combines weak background clinical data with MRI, but does not improve on S4-MRI.
- **C2 beats both constituent baselines**, supporting complementary MRI + safe-neuro information under the current development protocol.
- C3 does not beat cognition/FAQ alone and has lower AUC; its absolute performance is predominantly explained by diagnosis-proximal scales.
- C4 has no clear MCC or ACC advantage over full29 clinical-only and has lower AUC; it is not selected as the proposed multimodal method.

## Per-class behavior of the selected C2

| Model | Class | Recall / SEN | SPE | AUC | F1 |
|---|---|---:|---:|---:|---:|
| C0 | NC | 0.88377 ± 0.06206 | 0.53082 ± 0.06887 | 0.77256 ± 0.01078 | 0.65582 ± 0.00700 |
| C0 | MCI | 0.27406 ± 0.05755 | 0.81599 ± 0.00829 | 0.58725 ± 0.04267 | 0.36541 ± 0.06242 |
| C0 | DE | 0.49897 ± 0.10878 | 0.92170 ± 0.05981 | 0.85702 ± 0.03393 | 0.53702 ± 0.07047 |
| **C2** | NC | **0.85191 ± 0.01867** | **0.72418 ± 0.02469** | **0.85286 ± 0.01431** | **0.73154 ± 0.01996** |
| **C2** | MCI | **0.56926 ± 0.07743** | **0.74986 ± 0.01892** | **0.69295 ± 0.03266** | **0.60839 ± 0.05741** |
| **C2** | DE | **0.42864 ± 0.08291** | **0.95122 ± 0.02323** | **0.86901 ± 0.02372** | **0.51724 ± 0.04804** |

C2's main class-level benefit is the large MCI recall/F1 increase and improved NC specificity. It is not uniformly superior: DE recall falls from 0.4990 to 0.4286. This limitation should remain visible in the paper and motivates class-sensitive tuning only in a future, separately predeclared development study; it must not be tuned against held-out test data.

## Final method naming and paper placement

Recommended terminology:

- **CortexBridge S4-MRI**: primary MRI-only method and main mechanistic contribution.
- **CortexBridge S4-C**: selected clinical-assisted extension using C2.
- **Clinical upper bounds**: C3 and C4; report as leakage/proximity controls, not proposed models.

The journal method diagram should show the clinical gate as an optional inference branch after the Student MRI/global graph representation. The main table should contain S4-MRI and S4-C in separate input-setting blocks. C3/C4 belong in an ablation or clinical-label-proximity table. They should not be compared as if they had the same deployment inputs as MRI-only baselines.

The selected standalone development config is:

`configs/runtime_vnext_graph_student_locked_s4c_safe_neuro.yaml`

## Verification and data boundary

- Four clinical-fusion arms × three folds completed with prediction CSV, development metrics and best checkpoint.
- Each arm used 454/457/467 validation samples; IDs and labels match C0 fold by fold.
- All 12 formal runs record `held_out_test_evaluated=false`; all formal fold logs contain `test=NOT_LOADED`.
- Teacher clinical input remained disabled in every run.
- Parsed clinical feature lists contain no MMSE.
- The interrupted pre-epoch C4 fold-2 directory is preserved as `fold_2_interrupted_20261020` and is excluded from the explicit analysis registry.
- Full regression suite after locking S4-C: `64 passed`.

## Reproducible artifacts

- `analysis/s4_clinical_metrics_by_fold.csv`
- `analysis/s4_clinical_metrics_summary.csv`
- `analysis/s4_clinical_per_class_by_fold.csv`
- `analysis/s4_clinical_per_class_summary.csv`
- `analysis/paired_vs_s4_mri_only.csv`
- `analysis/matched_clinical_only_by_fold.csv`
- `analysis/fusion_vs_matched_clinical_only.csv`
- `analysis/s4_clinical_config_manifest.csv`
- `analysis/s4_clinical_analysis.json`
- matched clinical audit: `outputs/vnext_clinical_leakage_audit_s4c_20261020/`
- analysis command: `python analyze_s4_clinical_ablation.py --bootstrap-replicates 2000`

## Material Passport

- **Input:** locked S4-MRI predictions; 12 formal C1–C4 development prediction files; fold-local clinical-only audit predictions.
- **Transformation:** exact-ID validation; probability-based metric recomputation; three-fold mean ± sample SD; subject-level paired bootstrap; paired McNemar test.
- **Output:** report, fold/summary/per-class tables, paired comparisons, config manifest and machine-readable JSON.
- **Exclusions:** held-out test, dry-runs, interrupted partial fold, seven clinical rows without matched imaging sample, MMSE.
- **Scope:** internal development evidence only; not external generalization or a final test estimate.

## Epistemic / fallacy scan

1. Development selection is not described as held-out test performance.
2. C1–C4 are not described as MRI-only because clinical variables are required at inference.
3. High C3/C4 scores are not attributed to imaging without clinical-only controls.
4. MMSE exclusion is checked on parsed feature lists rather than comments or filenames.
5. AUC is computed from probabilities, not hard predictions.
6. SEN and Recall are not counted as independent evidence.
7. Fold means and pooled paired effects are labeled separately.
8. Bootstrap intervals do not replace external validation.
9. The C2 choice incorporates variable-risk tier and multimodal complementarity, not post-hoc maximum score alone.
10. The DE recall decline under C2 is not hidden by macro averages.
11. The interrupted fold artifact is preserved but excluded from formal analysis.

