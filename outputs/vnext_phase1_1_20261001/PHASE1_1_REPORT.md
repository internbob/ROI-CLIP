# CortexBridge Phase 1.1 — Ribbon Stabilization & Hybridization

Date: 2026-10-01

## Decision

Phase 1.1 was implemented and evaluated without consulting a new held-out test result. The implementation and geometry checks pass, but the current experiments do **not** establish a stable Phase 1 improvement:

- E2 (high-resolution, inner-5, uniform pure ribbon) improved the mean validation MCC by `+0.0040` over three paired folds, but one fold was negative and the paired standard deviation was `0.0089`.
- E3 (hybrid residual gate) did not beat E0 on fold 0 (`0.7092` versus `0.7110` validation MCC), and its mean gate jumped from about `0.08` to `0.59` when auxiliary losses became active.
- E4 was therefore not started, and Cross-Geometry Bridge remains out of scope.

The supported conclusion is that high-resolution inner ribbon pooling removes most of the original replacement penalty and may carry useful signal, but neither pure E2 nor the current E3 fusion has shown a stable gain yet.

## Implementation

- Preserved all existing uncommitted work before editing.
- Kept `legacy_mask` and pure `ribbon` modes and added `hybrid_ribbon`.
- Added dynamic reconstruction of requested ribbon depths from cached `white_voxel` and `pial_voxel` coordinates, so the existing full cache can be reused.
- Added high-resolution sampling from the first MRI encoder stage (stride 2).
- Added inner depths `[0.2, 0.35, 0.5, 0.65, 0.8]` with uniform depth mean.
- Added subject/ROI validity `q_roi`, separate legacy/ribbon normalization, a zero-initialized ribbon adapter, and a feature-conditioned scalar gate initialized to `0.1`.
- The residual output is `V_legacy + q_roi * g_roi * Adapter(V_ribbon)`. A final LayerNorm was intentionally not placed after the sum because it would break exact functional equality with the existing legacy branch at initialization.
- Preserved the RNG state around optional ribbon-module initialization, making same-seed common parameters identical across E0/E2/E3.
- Added a development-only training path (`runtime.evaluate_test: false`) and diagnostics for validity, depth entropy, gate value, and token norms.

## Geometry and feature diagnostics

Diagnostics used 32 fold-0 validation subjects and the frozen E0/E1 checkpoints.

| Check | Result |
|---|---:|
| Spatial augmentation | only zero-fill voxel shift; configured shift is `0`, so training is intensity-only |
| Coordinate convention | `align_corners=False`, `2*(voxel+0.5)/shape-1`, grid order `x=W,y=H,z=D` |
| Existing ribbon feature stage | `[B,128,20,24,20]`, effective stride 8 |
| High-resolution candidate | `[B,32,80,96,80]`, effective stride 2 |
| Existing adjacent-depth distance | mean `0.0739`, median `0.0736` feature voxels; `100% < 0.5` |
| Existing adjacent-depth cosine | mean `0.999319` |
| High-res inner-depth distance | mean `0.1774`, median `0.1767`; `99.86% < 0.5` |
| High-res inner-depth cosine | mean `0.997458` |
| Per-ROI valid ratio | minimum ROI mean `0.98625`; median and maximum `1.0` |

The old deep-stage five-depth sampler is therefore spatially near-degenerate. High resolution improves separation but does not fully resolve it. The previous overall valid rate near 92% was mainly caused by vertices without a DKT assignment, not poor coverage inside assigned ROIs.

Token distributions also differ materially before the new normalization/projection:

| Token source | Mean | Std | Mean L2 | Inter-subject variance | Inter-ROI variance |
|---|---:|---:|---:|---:|---:|
| Legacy | `0.0376` | `0.6430` | `7.2832` | `0.06785` | `0.10809` |
| Old ribbon | `0.6734` | `0.6836` | `10.7715` | `0.08692` | `0.09418` |

Full diagnostics are in `diagnostics_e1/diagnostics.json`; per-ROI results are in `diagnostics_e1/roi_validity.csv`.

## Development experiments

No new held-out test set was evaluated for these runs.

| Fold | E0 legacy MCC | E2 inner-uniform MCC | Paired delta | E0 AUC | E2 AUC |
|---:|---:|---:|---:|---:|---:|
| 0 | `0.710976` | `0.725100` | `+0.014124` | `0.936606` | `0.936955` |
| 1 | `0.704449` | `0.701963` | `-0.002487` | `0.934464` | `0.936114` |
| 2 | `0.674381` | `0.674730` | `+0.000349` | `0.928942` | `0.928813` |
| Mean | `0.696602` | `0.700597` | `+0.003995` | `0.933337` | `0.933960` |

Paired MCC delta sample SD: `0.008885`. Mean changes for E2 versus E0 were: AUC `+0.000624`, accuracy `+0.002207`, macro-F1 `+0.002396`, and balanced accuracy `+0.005010`.

Fold-0 ablation context:

| Experiment | Validation MCC | Validation AUC | Outcome |
|---|---:|---:|---|
| E0 legacy | `0.710976` | `0.936606` | baseline |
| E1 old deep ribbon attention | `0.724350` | `0.935403` | old replacement; depth features nearly identical |
| E2 high-res inner-5 uniform | `0.725100` | `0.936955` | best fold-0 result, not stable across 3 folds |
| E3 hybrid residual gate | `0.709173` | `0.937528` | did not pass E0 MCC; gate instability after auxiliary-loss activation |

## Verification

- Full unit suite: `29 passed`.
- E2 and E3 fold-0 dry-runs completed on A100 GPUs.
- Full E2/E3 fold-0 runs and paired E0/E2 fold-1/fold-2 development runs completed.
- Peak allocated GPU memory was about `17.3 GiB`.
- Every new development metrics file records `held_out_test_evaluated: false`.

## Recommended next decision

Do not enter Cross-Geometry Bridge on the claim that Phase 1 is already validated. If continuing Phase 1.1, the next bounded experiment should stabilize E3 around the auxiliary-loss transition (for example, gate-specific optimization/regularization or delayed ribbon residual activation) and repeat the same paired internal-CV protocol. E4 should remain gated on E3 first beating E0 consistently.
