# Phase 2.1 fold-0 lambda sweep

**Source**: `outputs/vnext_phase2_1_20261002/PHASE2_1_REPORT.md`, Fold-0 lambda sweep  
**Caption**: Development-only cross-geometry weight sweep  
**Extraction type**: raw_table

| lambda_cg | Best epoch | MCC | AUC | Acc@1 | Acc@5 | MRR | Gap | Spatially tolerant Acc |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.01 | 6 | 0.71169 | 0.93669 | 0.27560 | 0.65456 | 0.44540 | 0.30577 | 0.50554 |
| 0.05 | 11 | 0.70907 | 0.93348 | 0.35242 | 0.73515 | 0.52092 | 0.41198 | 0.58022 |
| 0.10 | 6 | 0.70673 | 0.93597 | 0.34812 | 0.72923 | 0.51645 | 0.38026 | 0.58043 |

The last column is renamed from the original report's topology label because the audit established it uses centroid spatial tolerance rather than true mesh anatomy.

