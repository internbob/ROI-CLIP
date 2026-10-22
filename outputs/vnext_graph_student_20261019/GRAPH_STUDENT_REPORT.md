# CortexBridge MRI-only Graph Student

- Baseline: `s0_scratch_corrected`.
- Full graph-state distillation: `s6_full_kd_corrected`.
- Clinical input: `off`.
- Subject-specific surface/PET/cognition input at Student inference: `none`.
- Held-out test loaded/evaluated: `false`.

## Three-fold development results

| Variant | MCC | AUC | Accuracy | Macro-F1 | Balanced accuracy |
|---|---:|---:|---:|---:|---:|
| s0_scratch_corrected | 0.27704 ± 0.03173 | 0.71830 ± 0.01595 | 0.49560 ± 0.02572 | 0.45080 ± 0.06271 | 0.53218 ± 0.05446 |
| s6_full_kd_corrected | 0.29141 ± 0.04274 | 0.72518 ± 0.01846 | 0.53331 ± 0.02600 | 0.52528 ± 0.01171 | 0.55677 ± 0.01168 |

## Candidate − baseline fold deltas

| Metric | Fold 0 | Fold 1 | Fold 2 | Mean |
|---|---:|---:|---:|---:|
| mcc | +0.03117 | -0.05738 | +0.06930 | +0.01436 |
| auc | +0.01390 | -0.01915 | +0.02590 | +0.00688 |
| acc | +0.03524 | +0.03720 | +0.04069 | +0.03771 |
| f1_macro | +0.00683 | +0.11537 | +0.10122 | +0.07448 |
| balanced_acc | -0.02813 | +0.01512 | +0.08677 | +0.02459 |

## Paired pooled-subject inference

- mcc: delta `+0.03028`, paired bootstrap 95% CI `[-0.02058, +0.08179]`.
- auc: delta `+0.02899`, paired bootstrap 95% CI `[+0.00559, +0.05246]`.
- balanced_acc: delta `+0.01944`, paired bootstrap 95% CI `[-0.01245, +0.05278]`.
- McNemar: candidate-only `277`, baseline-only `225`, exact p `0.022737`.

## Gate

- Predeclared development gate passed: `true`.
- Component ablations authorized: `true`.

The gate is intentionally based on matched architecture, matched subjects, and development validation only. Passing it authorized component ablations; it is not itself a claim of statistically stable or externally generalizable benefit.

## Post-gate component decision

The authorized S3–S5 progressive ablation is complete. S4 (CE + logit KD + ROI graph-state KD) has the highest matched three-fold development MCC (`0.31227 ± 0.04766`) and is locked for the next complete-CV stage. Graph-relation and prototype-posterior KD are dropped. See `GRAPH_STUDENT_COMPONENT_ABLATION_REPORT.md` for paired inference and the claim boundary. The held-out test remains unopened.
