# P2/P3 Privileged-KD Protocol Audit

- Held-out test read: `false`
- Student inference input: `MRI only`
- Clinical features: `16` clean numeric variables
- P2: MRI + safe clinical Teacher; B1 Student logit KD
- P3: joint geometry + safe clinical Teacher; B1 Student logit KD

| Fold | Train rows | Valid rows | Train patients | Valid patients | Patient overlap |
|---:|---:|---:|---:|---:|---:|
| 0 | 1433 | 454 | 1433 | 454 | 0 |
| 1 | 1428 | 461 | 1428 | 461 | 0 |
| 2 | 1408 | 470 | 1408 | 470 | 0 |

Clinical imputation and standardization are fitted on each training fold and reused unchanged for validation. MRI, surface and clinical values are read from the same visit row. The audit intentionally never opens a test CSV.
