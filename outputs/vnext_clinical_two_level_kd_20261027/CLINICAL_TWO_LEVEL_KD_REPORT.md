# Clinical Two-Level Distillation — Development Experiment

## Decision

- **The requested Teacher+Student clinical Logit+ROI-state experiment is complete, but D1 is not selected as the performance-primary method.**
- Against the matched Logit-only arm, D1−D0 pooled delta MCC is +0.00295 (95% CI [−0.01705, +0.02208]); the interval includes zero.
- D1 does not exceed the earlier E3 performance configuration: delta MCC −0.00283, delta AUC −0.00518 (95% CI [−0.00954, −0.00089]).
- Correct ROI correspondence is not supported as the source of improvement: D1−D3 delta MCC is −0.00232; the shuffled-ROI control has a slightly higher mean MCC.
- E3 should remain the classification-performance configuration. D1/D3 belong in the ablation section as evidence that the current ROI-state KD objective behaves mainly as a weak regularizer, not as validated anatomical knowledge transfer.

## Protocol

- Task: ADNI NC/MCI/DE three-class classification.
- Both Teacher and Student use the same 29 clinical variables; MMSE is excluded.
- Teacher: E4 MRI + surface + ribbon correspondence + anatomical ROI graph + clinical fusion.
- Student: MRI + clinical global classifier; ROI classification residual remains disabled.
- Two-level supervision: temperature-scaled class-logit KD plus cosine ROI-state KD.
- Model selection and all comparisons use subject-level development folds 0–2 only.
- Held-out test accessed: `false`.

## Fold-0 ROI-state weight selection

| lambda_roi | MCC | AUC | ACC | Macro-F1 |
|---:|---:|---:|---:|---:|
| 0.10 | 0.70986 | 0.93513 | 0.81498 | 0.80945 |
| 0.50 | 0.70866 | 0.93460 | 0.81498 | 0.80838 |
| 1.00 | 0.71079 | 0.93265 | 0.81278 | 0.80865 |

Selected `lambda_roi=1.00` by the locked development rule: highest fold-0 MCC, then AUC, then the smaller weight.

## Three-fold results

| Arm | Configuration | MCC | AUC | ACC | Macro-F1 | Recall | SPE | MCI recall |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| E3 | Earlier MRI+clinical Student + MRI+clinical Teacher Logit KD | 0.69535 ± 0.01743 | 0.93203 ± 0.00214 | 0.80708 ± 0.01159 | 0.79758 ± 0.01891 | 0.80618 ± 0.01988 | 0.89807 ± 0.00474 | 0.75383 ± 0.02793 |
| E4 | Earlier full privileged Teacher Logit KD | 0.69226 ± 0.01562 | 0.93199 ± 0.00224 | 0.80555 ± 0.00844 | 0.79401 ± 0.01618 | 0.80015 ± 0.01994 | 0.89661 ± 0.00472 | 0.75912 ± 0.03373 |
| D0 | Matched graph Student, Logit KD only | 0.68984 ± 0.02493 | 0.93090 ± 0.00631 | 0.80424 ± 0.01777 | 0.79305 ± 0.02205 | 0.79885 ± 0.02199 | 0.89595 ± 0.00735 | 0.75675 ± 0.03798 |
| D1 | Final clinical Logit + ROI-state KD | 0.69413 ± 0.01842 | 0.93082 ± 0.00453 | 0.80706 ± 0.00909 | 0.79338 ± 0.02137 | 0.79753 ± 0.03607 | 0.89665 ± 0.00791 | 0.77428 ± 0.04673 |
| D2 | D1 with cross-geometry-disabled Teacher | 0.68781 ± 0.02397 | 0.93182 ± 0.00458 | 0.80133 ± 0.01751 | 0.79242 ± 0.02105 | 0.80323 ± 0.01717 | 0.89579 ± 0.00682 | 0.73766 ± 0.03799 |
| D3 | D1 with Teacher ROI correspondence shuffled | 0.69504 ± 0.02334 | 0.93272 ± 0.00582 | 0.80641 ± 0.01702 | 0.79609 ± 0.02139 | 0.80524 ± 0.01424 | 0.89836 ± 0.00600 | 0.75340 ± 0.03772 |

## Paired subject-bootstrap contrasts

| Contrast | Delta MCC [95% CI] | Delta AUC [95% CI] | Delta ACC [95% CI] | McNemar p |
|---|---:|---:|---:|---:|
| D1 − D0 | +0.00295 [-0.01705, +0.02208] | -0.00371 [-0.00795, +0.00020] | +0.00290 [-0.00943, +0.01524] | 0.727547 |
| D1 − E3 | -0.00283 [-0.02521, +0.01978] | -0.00518 [-0.00954, -0.00089] | +0.00000 [-0.01379, +0.01451] | 1 |
| D1 − E4 | +0.00070 [-0.02095, +0.02160] | -0.00316 [-0.00650, +0.00032] | +0.00145 [-0.01163, +0.01451] | 0.914212 |
| D1 − D2 | +0.00477 [-0.01439, +0.02366] | -0.00467 [-0.00917, -0.00055] | +0.00581 [-0.00653, +0.01814] | 0.415985 |
| D1 − D3 | -0.00232 [-0.02359, +0.01865] | -0.00527 [-0.01008, -0.00048] | +0.00073 [-0.01306, +0.01451] | 1 |

## Conclusion

- The matched incremental effect of adding ROI-state KD is D1−D0 delta MCC +0.00295 (95% CI [-0.01705, +0.02208]).
- Correct ROI identity contributes only if D1 exceeds D3; observed delta MCC is -0.00232 (95% CI [-0.02359, +0.01865]).
- The cross-geometry-trained Teacher contribution is D1−D2 delta MCC +0.00477 (95% CI [-0.01439, +0.02366]).
- These development results do not authorize a held-out-test claim.
