# CortexBridge Student-only Med3D ResNet10 M0–M4 Ablation

Date: 2026-10-21

## Decision

**Retain M0 (the locked SFCN S4 + ROI-state KD Student) as the CortexBridge mainline.**

The best Med3D candidate is **M1**, but it does not exceed M0 on three-fold development MCC. The predeclared trigger for considering a Med3D cross-geometry teacher—M2 significantly outperforming M4—is `false`. Therefore this stage does not justify retraining the teacher with Med3D.

No held-out test split was loaded or evaluated. All conclusions below are development-only.

## Experimental matrix

| ID | Student encoder/objective | Teacher during training | Inference inputs |
|---|---|---|---|
| M0 | SFCN + CE + logit KD + ROI-state KD | Locked G3 SFCN cross-geometry teacher | MRI only |
| M1 | MedicalNet ResNet10, random initialization + CE | None | MRI only |
| M2 | MedicalNet/Med3D ResNet10 pretrained + CE | None | MRI only |
| M3 | M2 + logit KD | Locked G3 SFCN teacher | MRI only |
| M4 | M3 + ROI graph-state KD | Locked G3 SFCN teacher | MRI only |

Shared settings for M1–M4: identical folds, topology graph/query and downstream classifier; batch size 8; seed 3407; 30-epoch cap; patience 4; MCC checkpoint selection; no clinical/MMSE, PET, surface, or ribbon input at Student inference. M2–M4 use encoder LR `1e-5`; M1 uses `1e-4`; inherited and head LR are `1e-4`. M3/M4 use logit KD weight `0.5`; M4 adds ROI-state KD weight `1.0`.

## Three-fold development results

| ID | MCC | AUC | ACC | Macro-F1 | Recall / sensitivity | Specificity |
|---|---:|---:|---:|---:|---:|---:|
| M0 | 0.31227 ± 0.04766 | 0.73894 ± 0.02805 | 0.53872 ± 0.03044 | 0.51942 ± 0.04210 | 0.55227 ± 0.03053 | 0.75617 ± 0.01551 |
| M1 | 0.26318 ± 0.04678 | 0.69495 ± 0.01693 | 0.49112 ± 0.01790 | 0.45372 ± 0.02681 | 0.52159 ± 0.03022 | 0.73920 ± 0.01555 |
| M2 | 0.14511 ± 0.04697 | 0.62395 ± 0.01841 | 0.45715 ± 0.01171 | 0.41899 ± 0.00748 | 0.43629 ± 0.03092 | 0.71016 ± 0.01428 |
| M3 | 0.22263 ± 0.06020 | 0.67722 ± 0.03190 | 0.50899 ± 0.03339 | 0.49144 ± 0.04945 | 0.50169 ± 0.04228 | 0.73426 ± 0.02087 |
| M4 | 0.21078 ± 0.02745 | 0.66854 ± 0.02095 | 0.50447 ± 0.01251 | 0.46321 ± 0.03978 | 0.48077 ± 0.04573 | 0.72931 ± 0.00969 |

Recall is macro recall and equals the reported multiclass sensitivity/balanced accuracy. Specificity is macro one-vs-rest specificity. Values are fold mean ± sample SD.

## Fold-level MCC

| ID | Fold 0 | Fold 1 | Fold 2 | Mean |
|---|---:|---:|---:|---:|
| M0 | 0.36705 | 0.28941 | 0.28035 | 0.31227 |
| M1 | 0.29442 | 0.20941 | 0.28572 | 0.26318 |
| M2 | 0.19808 | 0.12871 | 0.10854 | 0.14511 |
| M3 | 0.28518 | 0.16510 | 0.21760 | 0.22263 |
| M4 | 0.17937 | 0.23015 | 0.22283 | 0.21078 |

## Progressive paired comparisons

The interval is a paired subject-level bootstrap on pooled validation predictions. Because only three folds are available, fold mean/SD and fold direction are descriptive; no low-powered fold-level t-test is used.

| Contrast | Fold ΔMCC | Mean fold ΔMCC | Positive folds | Pooled ΔMCC (95% CI) | McNemar p |
|---|---|---:|---:|---:|---:|
| M1 minus M0 | -0.07263, -0.08001, +0.00538 | -0.04909 | 1/3 | -0.04708 [-0.08691, -0.00931] | 0.000089 |
| M2 minus M1 | -0.09634, -0.08069, -0.17718 | -0.11807 | 0/3 | -0.12174 [-0.16760, -0.07330] | 0.028016 |
| M3 minus M2 | +0.08710, +0.03638, +0.10905 | +0.07751 | 3/3 | +0.07855 [+0.03222, +0.12591] | 0.000839 |
| M4 minus M3 | -0.10582, +0.06505, +0.00523 | -0.01185 | 2/3 | -0.01824 [-0.05894, +0.02375] | 0.781234 |
| M4 minus M2 | -0.01871, +0.10143, +0.11428 | +0.06567 | 2/3 | +0.06032 [+0.00842, +0.11243] | 0.003616 |

## Requested M2 versus M4 gate

- Fold mean ΔMCC (M2 − M4): `-0.06567`.
- Pooled paired ΔMCC (M2 − M4): `-0.06032` (95% CI `[-0.11219, -0.01078]`).
- Pooled paired ΔAUC (M2 − M4): `-0.04322` (95% CI `[-0.06755, -0.01995]`).
- M2 significantly better than M4 under the predeclared rule: `false`.
- Decision: do not retrain a Med3D cross-geometry teacher in this stage.

## Interpretation

1. **The locked CortexBridge mainline remains M0.** Replacing its SFCN Student with the tested MedicalNet ResNet10 recipes does not improve development MCC.
2. **Off-the-shelf Med3D pretraining is not beneficial under the locked recipe.** M2 should be interpreted against M1 to isolate initialization; the direction is adverse rather than beneficial.
3. **Logit KD partially recovers the weak pretrained Student.** M3 should be interpreted against M2.
4. **ROI-state KD is not a stable additional gain for the Med3D Student.** M4 should be interpreted against M3, with fold directions and the paired interval reported rather than claiming a benefit.
5. **A Med3D teacher is not currently justified.** The requested trigger is absent, and the best Med3D Student remains below M0. Further teacher retraining would expand compute without positive evidence that Med3D initialization is the bottleneck.

## Integrity and verification

- Official MedicalNet ResNet10 checkpoint SHA256: `afa8055f3e47f4a18239495d92a7abc587902c69c31c743de2b2784653b72605`.
- Each pretrained Student run loaded 72 backbone tensors.
- All 15 included fold JSON files record `held_out_test_evaluated: false`.
- The three checkpoints of the best Med3D variant M1 passed the MRI-only audit: no privileged state keys, zero logit change from an arbitrary clinical argument, zero logit change after invalidating subject-specific cache paths, and held-out flags false.
- Interrupted/incomplete directories and dry-runs are excluded by explicit variant paths.
- Selection metric and early stopping were unchanged across the requested ablation.
