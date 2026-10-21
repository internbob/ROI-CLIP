# Phase 2.1.5 subject-specific similarity and shuffle audit

**Source**: `outputs/vnext_phase2_1_5_20261003/PHASE2_1_5_CORRESPONDENCE_AUDIT.md`, Sections 1–2  
**Caption**: Four-cell cosine audit, descriptive effect shares, and deterministic subject-shuffle retrieval  
**Extraction type**: raw_table

| Pair type | Fold 0 | Fold 1 | Fold 2 | Three-fold mean ± SD |
|---|---:|---:|---:|---:|
| same subject, same ROI | 0.33221 | 0.33965 | 0.32470 | 0.33219 ± 0.00747 |
| different subject, same ROI | 0.33065 | 0.33738 | 0.32234 | 0.33012 ± 0.00753 |
| same subject, different ROI | 0.02644 | 0.01930 | 0.03565 | 0.02713 ± 0.00819 |
| different subject, different ROI | 0.02653 | 0.01922 | 0.03563 | 0.02713 ± 0.00822 |
| G_subject | 0.00156 | 0.00226 | 0.00236 | 0.00206 ± 0.00043 |

| Fold | ROI-identity component | Subject component | ROI-identity share | Subject share |
|---:|---:|---:|---:|---:|
| 0 | 0.30412 | 0.00156 | 99.49% | 0.51% |
| 1 | 0.31816 | 0.00226 | 99.29% | 0.71% |
| 2 | 0.28671 | 0.00236 | 99.18% | 0.82% |

| Fold | Condition | Acc@1 | Acc@5 | MRR | Diagonal gap |
|---:|---|---:|---:|---:|---:|
| 0 | normal | 0.27560 | 0.65456 | 0.44540 | 0.30577 |
| 0 | subject-shuffled | 0.27134 | 0.65028 | 0.44124 | 0.30413 |
| 1 | normal | 0.26597 | 0.64288 | 0.43560 | 0.32034 |
| 1 | subject-shuffled | 0.26330 | 0.63681 | 0.43182 | 0.31808 |
| 2 | normal | 0.24323 | 0.62173 | 0.41310 | 0.28905 |
| 2 | subject-shuffled | 0.23856 | 0.61446 | 0.40805 | 0.28670 |

Across-fold normal Acc@1: `0.26160 ± 0.01662`; shuffled Acc@1: `0.25773 ± 0.01709`.

