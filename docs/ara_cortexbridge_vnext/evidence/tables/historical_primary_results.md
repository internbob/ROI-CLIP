# Historical primary ADNI three-class results

**Source**: `outputs/tmi_complete_20260924/RESULTS_REPORT.md`, Primary ADNI three-class results  
**Caption**: Audited historical evidence package; these models precede the planned vNext graph method  
**Extraction type**: raw_table

| Model | ACC | AUC | MCC | Macro-F1 |
|---|---:|---:|---:|---:|
| Strong MRI-only scratch baseline | 0.5444 ± 0.0412 | 0.7444 ± 0.0260 | 0.3104 ± 0.0315 | 0.5301 ± 0.0567 |
| MRI+tabular CE baseline | 0.7910 ± 0.0125 | 0.9356 ± 0.0035 | 0.6741 ± 0.0213 | 0.7813 ± 0.0120 |
| Privileged teacher, affine-v3 final | 0.7967 ± 0.0138 | 0.9321 ± 0.0046 | 0.6818 ± 0.0195 | 0.7864 ± 0.0131 |
| MRI+tabular KD student, affine-v3 final | 0.7997 ± 0.0127 | 0.9327 ± 0.0053 | 0.6876 ± 0.0179 | 0.7900 ± 0.0106 |
| True MRI-only KD student, affine-v3 final | 0.4853 ± 0.0237 | 0.6318 ± 0.0287 | 0.1252 ± 0.0436 | 0.3814 ± 0.0405 |

