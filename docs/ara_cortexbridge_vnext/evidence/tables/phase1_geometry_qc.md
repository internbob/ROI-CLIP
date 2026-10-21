# Phase 1 geometry and cache QC

**Source**: `outputs/vnext_phase1_20260930/PHASE1_GATE_REPORT.md`, Cache and coordinate QC  
**Caption**: Reported full-cache and five-subject ribbon geometry checks  
**Extraction type**: raw_table

| Item | Exact reported value |
|---|---|
| Cache bundles written | 2,343/2,343 |
| Cache failures | 0 |
| Cache size | 1.7 GiB |
| Depths | [0.0, 0.25, 0.5, 0.75, 1.0] |
| Vertices | 5,124 |
| DKT ROIs | 62 |
| Mean valid DKT vertex fraction | 0.921741 |
| Mean white-to-pial thickness | 2.23--2.56 voxels |
| Five-subject grid check | all grids finite; all sampled depths inside cropped MRI |
| Visual inspection | no global translation or axis-swap pattern |

Additional affine audit from `outputs/tmi_complete_20260924/RESULTS_REPORT.md`:

| Item | Before | After / change |
|---|---:|---:|
| MRI–ribbon foreground overlap | 0.83585 | 0.88706 |
| QC subjects improved | — | 18/20 |

