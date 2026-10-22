# Phase 1 gate report (fold 0, seed 3407)

## Gate decision

**STOP before Phase 2.** The Phase 1 implementation and coordinate checks pass, but the
predefined empirical gate `ROI-CLIP_legacy < ROI-CLIP_ribbon` does not hold on the
held-out test set. Cross-Geometry Bridge was therefore not started.

## Cache and coordinate QC

- Ribbon cache: 2,343/2,343 bundles written, 0 failures, 1.7 GiB.
- Geometry: 5 depths `[0.0, 0.25, 0.5, 0.75, 1.0]`, 5,124 vertices, 62 DKT ROIs.
- Five-subject seeded QC: all grids finite; all sampled depths inside the cropped MRI.
- Mean valid DKT vertex fraction: 0.921741.
- Mean white-to-pial thickness across subjects: 2.23--2.56 voxels.
- Visual inspection: white/mid/pial overlays follow cortical anatomy with no global
  translation or axis-swap pattern.

## Matched fold-0 comparison

Both runs use the same split (1,432 train / 454 validation / 457 test), seed 3407,
batch size 8, optimizer, augmentation, clinical inputs, loss weights, selection
metric (MCC), and early-stopping rule. Only volume pooling differs.

| Metric | legacy_mask | ribbon | ribbon - legacy |
|---|---:|---:|---:|
| Best validation MCC | 0.710976 | 0.724350 | +0.013374 |
| Test ACC | 0.807440 | 0.789934 | -0.017505 |
| Test AUC | 0.930568 | 0.928009 | -0.002560 |
| Test MCC | 0.685984 | 0.661553 | -0.024431 |
| Test macro-F1 | 0.788576 | 0.774144 | -0.014432 |
| Peak allocated GPU memory (MiB) | 17,304.2 | 17,331.7 | +27.4 |

The ribbon run selected epoch 3, before ROI-CLIP auxiliary losses were enabled at
epoch 6. Its held-out test metrics are lower than the legacy baseline even though
its best validation MCC is higher. One fold/seed is not enough to estimate
statistical significance, but it is enough to fail the implementation guide's
strict progression gate for this run.

## Artifacts

- Cache manifest: `/home_data/home/zhouwl2023/Data/CortexWeave_ROIClip_cache/adni_dktatlas_5124_ribbon_v1/ribbon_manifest.json`
- QC summary and overlays: `outputs/vnext_phase1_20260930/qc_5subjects/`
- Legacy dry-run: `outputs/vnext_phase1_20260930/dryrun_legacy/`
- Ribbon dry-run: `outputs/vnext_phase1_20260930/dryrun_ribbon/`
- Legacy full run: `outputs/vnext_phase1_20260930/full_legacy/`
- Ribbon full run: `outputs/vnext_phase1_20260930/full_ribbon/`
