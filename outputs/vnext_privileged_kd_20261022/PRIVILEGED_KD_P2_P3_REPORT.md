# CortexBridge P2/P3 Privileged-KD Development Report

- Completed: `2026-10-23`
- Protocol: matched three-fold development validation; selection metric `MCC`.
- Student inference input: `MRI only` for P0, P2, and P3.
- Held-out test loaded/evaluated: `false`.
- Student architecture: locked B1 global classifier; Student ROI residual and graph are disabled.

## Three-fold results

| Variant | MCC | AUC | ACC | Macro-F1 | Macro recall | Macro specificity | MCI recall |
|---|---:|---:|---:|---:|---:|---:|---:|
| P0: B1: existing G3 logit-KD baseline | 0.32870 ± 0.03922 | 0.74623 ± 0.02770 | 0.56490 ± 0.04083 | 0.55806 ± 0.04940 | 0.57938 ± 0.03251 | 0.76816 ± 0.01586 | 0.40578 ± 0.14243 |
| P2: clinical-privileged Teacher -> MRI-only B1 Student | 0.30428 ± 0.03344 | 0.74544 ± 0.02420 | 0.56704 ± 0.03753 | 0.54478 ± 0.04745 | 0.54170 ± 0.02814 | 0.76082 ± 0.01208 | 0.50360 ± 0.15271 |
| P3: joint clinical+geometry Teacher -> MRI-only B1 Student | 0.25893 ± 0.04889 | 0.69661 ± 0.01913 | 0.50617 ± 0.04166 | 0.49548 ± 0.04380 | 0.54774 ± 0.03606 | 0.74414 ± 0.01799 | 0.46096 ± 0.20743 |

## Privileged Teacher diagnostics

These values describe the training-only privileged Teachers, not deployable MRI-only performance.

| Teacher | MCC | AUC | ACC | Macro-F1 | MCI recall | ROI Acc@1 | ROI Acc@5 | Diagonal gap |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| P2: clinical privileged | 0.42363 ± 0.08202 | 0.79602 ± 0.04325 | 0.63523 ± 0.05012 | 0.61553 ± 0.04240 | 0.56421 ± 0.01824 | — | — | — |
| P3: joint clinical+geometry privileged | 0.41999 ± 0.03122 | 0.80601 ± 0.02099 | 0.63878 ± 0.01963 | 0.61603 ± 0.02682 | 0.59276 ± 0.01119 | 0.20053 ± 0.04190 | 0.54813 ± 0.07020 | 0.24489 ± 0.04371 |

## Fold-stratified matched comparisons

### P2_minus_P0

- clinical privileged knowledge over B1.
- Fold MCC deltas: `-0.02258, -0.03143, -0.01925`; mean `-0.02442`.
- Fold-stratified paired MCC delta `-0.02442` (95% CI `[-0.06129, +0.01311]`).
- Fold-stratified paired AUC delta `-0.00079` (95% CI `[-0.01270, +0.01064]`).
- Fold-stratified paired Macro-F1 delta `-0.01327`; MCI recall delta `+0.09781`.
- Exact McNemar p `0.901130`.

### P3_minus_P0

- joint privileged knowledge over B1.
- Fold MCC deltas: `-0.12043, -0.11247, +0.02360`; mean `-0.06977`.
- Fold-stratified paired MCC delta `-0.06977` (95% CI `[-0.11427, -0.02421]`).
- Fold-stratified paired AUC delta `-0.04961` (95% CI `[-0.06553, -0.03342]`).
- Fold-stratified paired Macro-F1 delta `-0.06257`; MCI recall delta `+0.05518`.
- Exact McNemar p `0.000369`.

### P3_minus_P2

- geometry-branch increment within the privileged-Teacher design.
- Fold MCC deltas: `-0.09784, -0.08104, +0.04284`; mean `-0.04535`.
- Fold-stratified paired MCC delta `-0.04535` (95% CI `[-0.09432, +0.00367]`).
- Fold-stratified paired AUC delta `-0.04883` (95% CI `[-0.06613, -0.03198]`).
- Fold-stratified paired Macro-F1 delta `-0.04930`; MCI recall delta `-0.04264`.
- Exact McNemar p `0.000286`.

## Final development decision

- P2 advance: `false`. Its MCC delta is negative in 3/3 folds; mean MCC/AUC/Macro-F1 deltas are `-0.02442` / `-0.00079` / `-0.01327`. MCI recall changes by `+0.09781`, which is a sensitivity trade-off, not an overall advancement.
- P3 advance: `false`.
- P3 mean MCC improved over P0: `false` (1/3 folds positive).
- P3 mean AUC improved over P0: `false` (0/3 folds positive).
- Macro-F1 / MCI recall did not worsen in mean: `false` / `true`.
- Paired MCC / AUC bootstrap intervals exclude zero on the positive side: `false` / `false`.
- Locked MRI-only development method: `P0`.
- Do not run P4/P5 or evaluate the held-out test in this stage. P4/P5 would only become necessary for causal attribution after a joint privileged method first passes the performance gate.

The P3 Teacher learned non-random cross-geometry correspondence, but that privileged signal did not transfer successfully through class-logit KD to the MRI-only B1 Student. P3-P2 remains evidence about the complete geometry-privileged Teacher branch, not a clean causal estimate of the hard surface-ribbon correspondence loss alone.
