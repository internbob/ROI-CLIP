# Phase 1 matched fold-0 legacy versus ribbon

**Source**: `outputs/vnext_phase1_20260930/PHASE1_GATE_REPORT.md`, Matched fold-0 comparison  
**Caption**: Same split, seed, optimization, clinical inputs and selection; only volume pooling differs  
**Extraction type**: raw_table

| Metric | legacy_mask | ribbon | ribbon - legacy |
|---|---:|---:|---:|
| Best validation MCC | 0.710976 | 0.724350 | +0.013374 |
| Test ACC | 0.807440 | 0.789934 | -0.017505 |
| Test AUC | 0.930568 | 0.928009 | -0.002560 |
| Test MCC | 0.685984 | 0.661553 | -0.024431 |
| Test macro-F1 | 0.788576 | 0.774144 | -0.014432 |
| Peak allocated GPU memory (MiB) | 17,304.2 | 17,331.7 | +27.4 |

