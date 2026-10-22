# CortexBridge vNext — Disease-Aware Graph Stage A

## Protocol

- Development folds: `[0, 1, 2]`.
- Clinical input: `off` for every arm, based on the frozen leakage audit.
- Held-out test loaded/evaluated: `false`.
- G0: hard cross-geometry baseline; G1: unbiased graph; G2: boundary bias; G3: boundary + homology bias.

## Three-fold classification

| Variant | MCC | AUC | ACC | Macro-F1 | Balanced ACC |
|---|---:|---:|---:|---:|---:|
| G0 | 0.27153 ± 0.04199 | 0.71620 ± 0.03300 | 0.53228 ± 0.03521 | 0.52452 ± 0.04474 | 0.54089 ± 0.02617 |
| G1 | 0.28308 ± 0.04855 | 0.72546 ± 0.01855 | 0.54378 ± 0.02434 | 0.52315 ± 0.06278 | 0.54839 ± 0.06680 |
| G2 | 0.27526 ± 0.04186 | 0.72605 ± 0.03109 | 0.54389 ± 0.03617 | 0.51794 ± 0.06952 | 0.52578 ± 0.04659 |
| G3 | 0.29236 ± 0.01713 | 0.72346 ± 0.01842 | 0.54215 ± 0.00912 | 0.53517 ± 0.02896 | 0.55718 ± 0.02183 |

## Graph diagnostics

| Variant | Anatomy mass | Homology mass | Remote mass | Uniform KL | Boundary corr. | Gate | Output delta L2 |
|---|---:|---:|---:|---:|---:|---:|---:|
| G1 | 0.08242 | 0.01720 | 0.88360 | 0.07141 | 0.00012 | 0.10252 | 1.68262 |
| G2 | 0.10127 | 0.01655 | 0.86507 | 0.05788 | 0.35837 | 0.10252 | 1.80110 |
| G3 | 0.09839 | 0.04280 | 0.84203 | 0.05066 | 0.24399 | 0.10274 | 2.05269 |

## Paired validation comparisons

| Variant − G0 | Metric | Difference | 95% subject-bootstrap CI |
|---|---|---:|---:|
| G1 − G0 | mcc | 0.00888 | [-0.03453, 0.05009] |
| G1 − G0 | auc | 0.00006 | [-0.01419, 0.01414] |
| G1 − G0 | balanced_acc | 0.00866 | [-0.01906, 0.03773] |
| G2 − G0 | mcc | -0.00306 | [-0.04455, 0.03841] |
| G2 − G0 | auc | -0.00511 | [-0.02196, 0.01129] |
| G2 − G0 | balanced_acc | -0.01350 | [-0.04080, 0.01367] |
| G3 − G0 | mcc | 0.02021 | [-0.02188, 0.06107] |
| G3 − G0 | auc | 0.00046 | [-0.01591, 0.01589] |
| G3 − G0 | balanced_acc | 0.02167 | [-0.00553, 0.04879] |

Exact McNemar tests use paired validation predictions and Holm correction:

- G1 vs G0: G0-only `174`, candidate-only `190`, exact p `0.431783`, Holm p `1`.
- G2 vs G0: G0-only `161`, candidate-only `177`, exact p `0.414601`, Holm p `1`.
- G3 vs G0: G0-only `160`, candidate-only `174`, exact p `0.476936`, Holm p `1`.

## Gate decision

- Graph Teacher gate passed: `true`.
- Selected topology variant: `g3`.
- Next required control: `G2-R/G3-R topology randomization`.
- Student stage authorized: `true`.

This report does not implement or start Student KD. A topology-randomization control remains required before treating an anatomical bias as causal evidence.
