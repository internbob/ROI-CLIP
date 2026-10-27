# CortexBridge Unified Clinical Missingness Audit

Development folds only; held-out test accessed: `false`.

## E7 synthetic masking

| Missing | MCC | AUC | ACC | Macro-F1 | MCI recall |
|---:|---:|---:|---:|---:|---:|
| 0% | 0.69226 ± 0.01562 | 0.93199 ± 0.00224 | 0.80555 ± 0.00844 | 0.79401 ± 0.01618 | 0.75912 ± 0.03373 |
| 25% | 0.54637 ± 0.02873 | 0.89451 ± 0.00191 | 0.71466 ± 0.01828 | 0.67464 ± 0.00653 | 0.89097 ± 0.03718 |
| 50% | 0.36114 ± 0.03198 | 0.84986 ± 0.02778 | 0.58786 ± 0.03325 | 0.46241 ± 0.04570 | 0.96536 ± 0.02304 |
| 75% | 0.20103 ± 0.01893 | 0.73980 ± 0.04272 | 0.50224 ± 0.02749 | 0.29474 ± 0.02279 | 0.99066 ± 0.00918 |
| 100% | 0.00000 ± 0.00000 | 0.53449 ± 0.03865 | 0.45654 ± 0.01800 | 0.20891 ± 0.00562 | 1.00000 ± 0.00000 |

## E8 real missingness strata

| Scope | Observed fraction stratum | N | Classes | Reportable | MCC | AUC | ACC | Macro-F1 |
|---|---|---:|---:|---|---:|---:|---:|---:|
| pooled | 50_to_lt75pct | 1 | 1 | false | 0.00000 | NA | 1.00000 | 1.00000 |
| pooled | 75_to_lt100pct | 92 | 3 | true | 0.55458 | 0.89135 | 0.73913 | 0.73485 |
| pooled | complete | 1274 | 3 | true | 0.70043 | 0.93125 | 0.81005 | 0.79887 |
| pooled | gt0_to_lt50pct | 11 | 3 | false | 0.70276 | 0.97778 | 0.81818 | 0.57407 |
| fold_0 | 75_to_lt100pct | 32 | 3 | true | 0.59017 | 0.93251 | 0.78125 | 0.74134 |
| fold_0 | complete | 420 | 3 | true | 0.71468 | 0.93039 | 0.81667 | 0.81455 |
| fold_0 | gt0_to_lt50pct | 2 | 2 | false | 1.00000 | NA | 1.00000 | 1.00000 |
| fold_1 | 75_to_lt100pct | 33 | 3 | true | 0.47785 | 0.86884 | 0.69697 | 0.68740 |
| fold_1 | complete | 419 | 3 | true | 0.69878 | 0.94036 | 0.80907 | 0.79819 |
| fold_1 | gt0_to_lt50pct | 5 | 3 | false | 0.44194 | 0.94444 | 0.60000 | 0.44444 |
| fold_2 | 50_to_lt75pct | 1 | 1 | false | 0.00000 | NA | 1.00000 | 1.00000 |
| fold_2 | 75_to_lt100pct | 27 | 3 | false | 0.63748 | 0.93774 | 0.74074 | 0.74755 |
| fold_2 | complete | 435 | 3 | true | 0.68944 | 0.93163 | 0.80460 | 0.78116 |
| fold_2 | gt0_to_lt50pct | 4 | 2 | false | 1.00000 | NA | 1.00000 | 1.00000 |

Synthetic masks are applied only to originally observed variables. Existing missing values remain missing. A real-missingness stratum is marked reportable only when pooled/fold N >= 30 and all three classes are present; AUC is `NA` when fewer than three classes are present. Tiny strata are retained for provenance but must not be interpreted. At 100% synthetic missingness the model predicts MCI for every subject, so MCI recall=1.0 reflects single-class collapse, not useful sensitivity. No model is retrained and no test split is loaded.
