# Phase 2.1.5 top-one mismatch decomposition

**Source**: `outputs/vnext_phase2_1_5_20261003/PHASE2_1_5_CORRESPONDENCE_AUDIT.md`, Section 5  
**Caption**: Sequential categories: mesh neighbour, homologue, spatial-only, remote  
**Extraction type**: raw_table

| Fold | Mesh neighbour | Homologue | Spatial-only | Remote | Mismatches |
|---:|---:|---:|---:|---:|---:|
| 0 | 38.763% | 3.428% | 1.025% | 56.784% | 40781 |
| 1 | 38.819% | 3.282% | 1.103% | 56.796% | 41596 |
| 2 | 35.894% | 3.117% | 0.968% | 60.021% | 43823 |
| All | 37.785% | 3.272% | 1.031% | 57.912% | 126200 |

Exact aggregate counts from `audit_metrics.json`:

| Category | Count | Fraction |
|---|---:|---:|
| true_mesh_neighbor | 47685 | 0.3778526148969889 |
| bilateral_homolog | 4129 | 0.032717908082408875 |
| spatially_close_not_mesh_adjacent | 1301 | 0.010309033280507132 |
| remote_roi | 73085 | 0.5791204437400951 |

