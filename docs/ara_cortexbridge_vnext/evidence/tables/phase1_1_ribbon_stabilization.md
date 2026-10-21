# Phase 1.1 ribbon stabilization and hybridization

**Source**: `outputs/vnext_phase1_1_20261001/PHASE1_1_REPORT.md`, Geometry diagnostics and development experiments  
**Caption**: Exact tables reported for high-resolution inner ribbon and hybrid ablations  
**Extraction type**: raw_table

| Check | Result |
|---|---:|
| Spatial augmentation | only zero-fill voxel shift; configured shift is 0, so training is intensity-only |
| Coordinate convention | align_corners=False, 2*(voxel+0.5)/shape-1, grid order x=W,y=H,z=D |
| Existing ribbon feature stage | [B,128,20,24,20], effective stride 8 |
| High-resolution candidate | [B,32,80,96,80], effective stride 2 |
| Existing adjacent-depth distance | mean 0.0739, median 0.0736 feature voxels; 100% < 0.5 |
| Existing adjacent-depth cosine | mean 0.999319 |
| High-res inner-depth distance | mean 0.1774, median 0.1767; 99.86% < 0.5 |
| High-res inner-depth cosine | mean 0.997458 |
| Per-ROI valid ratio | minimum ROI mean 0.98625; median and maximum 1.0 |

| Fold | E0 legacy MCC | E2 inner-uniform MCC | Paired delta | E0 AUC | E2 AUC |
|---:|---:|---:|---:|---:|---:|
| 0 | 0.710976 | 0.725100 | +0.014124 | 0.936606 | 0.936955 |
| 1 | 0.704449 | 0.701963 | -0.002487 | 0.934464 | 0.936114 |
| 2 | 0.674381 | 0.674730 | +0.000349 | 0.928942 | 0.928813 |
| Mean | 0.696602 | 0.700597 | +0.003995 | 0.933337 | 0.933960 |

| Experiment | Validation MCC | Validation AUC | Outcome |
|---|---:|---:|---|
| E0 legacy | 0.710976 | 0.936606 | baseline |
| E1 old deep ribbon attention | 0.724350 | 0.935403 | old replacement; depth features nearly identical |
| E2 high-res inner-5 uniform | 0.725100 | 0.936955 | best fold-0 result, not stable across 3 folds |
| E3 hybrid residual gate | 0.709173 | 0.937528 | did not pass E0 MCC; gate instability after auxiliary-loss activation |

Paired MCC delta sample SD: `0.008885`.

