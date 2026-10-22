# CortexBridge Phase 2.1.5 — Correspondence Audit

Date: 2026-10-03

## Scope and safeguards

This is a frozen-checkpoint, validation-only audit. No model was trained, no optimizer was created, `backward()` was not called, topology-soft targets were not implemented, and no held-out test split was read. The optional gradient analysis used `torch.autograd.grad` on diagnostic validation batches without updating parameters.

## 1. Subject-specific 2×2 similarity analysis

All entries are cosine similarities after the learned projectors and L2 normalization. The aggregate is the mean ± sample SD over the three development folds.

| Pair type | Fold 0 | Fold 1 | Fold 2 | Three-fold mean ± SD |
|---|---:|---:|---:|---:|
| same subject, same ROI | 0.33221 | 0.33965 | 0.32470 | 0.33219 ± 0.00747 |
| different subject, same ROI | 0.33065 | 0.33738 | 0.32234 | 0.33012 ± 0.00753 |
| same subject, different ROI | 0.02644 | 0.01930 | 0.03565 | 0.02713 ± 0.00819 |
| different subject, different ROI | 0.02653 | 0.01922 | 0.03563 | 0.02713 ± 0.00822 |
| G_subject | 0.00156 | 0.00226 | 0.00236 | 0.00206 ± 0.00043 |

$G_{subject}=sim(S[b,i],R[b,i])-sim(S[b,i],R[b',i])$.

`G_subject` is positive on all folds, so the result supports a subject-specific increment beyond atlas ROI identity.

For an interpretable effect decomposition, the ROI-identity component is `different-subject/same-ROI − different-subject/different-ROI`; the subject component is `G_subject`. Their sum is the total same-subject/same-ROI excess over the different-subject/different-ROI baseline.

| Fold | ROI-identity component | Subject component | ROI-identity share | Subject share |
|---:|---:|---:|---:|---:|
| 0 | 0.30412 | 0.00156 | 99.49% | 0.51% |
| 1 | 0.31816 | 0.00226 | 99.29% | 0.71% |
| 2 | 0.28671 | 0.00236 | 99.18% | 0.82% |

Across folds, the effect-share means are 99.32% for ROI identity and 0.68% for the subject-specific increment. These are descriptive contrast shares, not causal variance explained.

## 2. Normal versus subject-shuffled retrieval

Ribbon tokens were shuffled with 20 deterministic subject derangements per fold; every shuffled pair therefore uses a different subject. Shuffled values are means across permutations.

| Fold | Condition | Acc@1 | Acc@5 | MRR | Diagonal gap |
|---:|---|---:|---:|---:|---:|
| 0 | normal | 0.27560 | 0.65456 | 0.44540 | 0.30577 |
| 0 | subject-shuffled | 0.27134 | 0.65028 | 0.44124 | 0.30413 |
| 1 | normal | 0.26597 | 0.64288 | 0.43560 | 0.32034 |
| 1 | subject-shuffled | 0.26330 | 0.63681 | 0.43182 | 0.31808 |
| 2 | normal | 0.24323 | 0.62173 | 0.41310 | 0.28905 |
| 2 | subject-shuffled | 0.23856 | 0.61446 | 0.40805 | 0.28670 |

Across folds, normal Acc@1 is 0.26160 ± 0.01662 and subject-shuffled Acc@1 is 0.25773 ± 0.01709. The shuffle test preserves ROI identity while removing matched-subject information.

## 3. True DKT mesh topology

The graph was constructed from fsaverage triangular mesh edges. Because `mris_ca_label` could not run on the compute-node system libraries, FreeSurfer's bundled `bert` DKTatlas40 annotation was transferred to fsaverage by nearest-neighbour matching in registered `sphere.reg`/fsaverage spherical coordinates. Only the 31 retained project DKT parcels per hemisphere were valid projection sources; non-cortex vertices were excluded.

- ROI count: `62` (31 left + 31 right)
- Undirected anatomical edges: `160`
- Degree min / median / mean / max: `3` / `5.0` / `5.16` / `9`
- Shared boundary mesh edges: `19563`
- Symmetric raw adjacency and boundary counts: `True`
- Zero diagonal: `True`
- Invalid cortical DKT labels: `0`
- Exact one-to-one left/right homologues: `True`

The exported `true_dkt_topology.npz` contains `A_anat`, `B_boundary`, row-normalized `A_boundary`, and the separate `A_hom` matrix in exact model ROI order.

## 4. Spatial top-3 versus mesh neighbours

| Fold | Mean overlap count (of 3) | Mean Jaccard | ROIs with zero overlap |
|---:|---:|---:|---:|
| 0 | 2.726 | 0.540 | 0 / 62 |
| 1 | 2.710 | 0.534 | 0 / 62 |
| 2 | 2.742 | 0.544 | 0 / 62 |

Per-ROI mismatches are in `roi_topology_comparison.csv`. The Phase 2.1 metric is therefore renamed **spatially tolerant retrieval accuracy**; it is not treated as anatomical topology accuracy.

## 5. Top-1 mismatch decomposition

Categories are applied sequentially: mesh neighbour → bilateral homologue → spatial top-3 but not mesh-adjacent → remote.

| Fold | Mesh neighbour | Homologue | Spatial-only | Remote | Mismatches |
|---:|---:|---:|---:|---:|---:|
| 0 | 38.763% | 3.428% | 1.025% | 56.784% | 40781 |
| 1 | 38.819% | 3.282% | 1.103% | 56.796% | 41596 |
| 2 | 35.894% | 3.117% | 0.968% | 60.021% | 43823 |
| All | 37.785% | 3.272% | 1.031% | 57.912% | 126200 |

## 6. True graph-distance metrics

The primary finite graph is `A_anat ∪ A_hom`: mesh edges are anatomical within a hemisphere and the separately validated homology edges connect hemispheres. Pure `A_anat` is disconnected across hemispheres, so its cross-hemisphere top-1 unreachable fraction and within-hemisphere TDE are reported as diagnostics instead of silently replacing infinity with an arbitrary number.

| Fold | Direction | TDE | ETD | Mesh-only unreachable | Mesh-only within-hemi TDE |
|---:|---|---:|---:|---:|---:|
| 0 | surface→ribbon | 1.41189 | 2.21616 | 17.902% | 1.05552 |
| 0 | ribbon→surface | 1.59923 | 2.25234 | 24.581% | 1.09148 |
| 0 | bidirectional | 1.50556 | 2.23425 | 21.241% | 1.07350 |
| 1 | surface→ribbon | 1.44127 | 2.20156 | 17.597% | 1.08163 |
| 1 | ribbon→surface | 1.62187 | 2.25131 | 24.183% | 1.12825 |
| 1 | bidirectional | 1.53157 | 2.22643 | 20.890% | 1.10494 |
| 2 | surface→ribbon | 1.50729 | 2.25117 | 16.526% | 1.19302 |
| 2 | ribbon→surface | 1.72076 | 2.29260 | 26.103% | 1.20565 |
| 2 | bidirectional | 1.61403 | 2.27189 | 21.315% | 1.19933 |

## 7. Gradient-conflict diagnostic

No optimizer step was performed. Cosines compare CE classification gradients with unweighted cross-geometry gradients on the shared MRI encoder.

| Fold | Stage | Status | Gradient cosine mean ± SD | Mean ‖g_cls‖ | Mean ‖g_cg‖ |
|---:|---|---|---:|---:|---:|
| 0 | stem | measured | -0.04243 ± 0.21388 | 2.258 | 32.129 |
| 0 | block1 | not_reached_by_cross_geometry_loss | n/a | 2.4564 | 0 |
| 0 | block2 | not_reached_by_cross_geometry_loss | n/a | 1.3707 | 0 |
| 0 | block3 | not_reached_by_cross_geometry_loss | n/a | 0.98572 | 0 |
| 0 | block4 | not_reached_by_cross_geometry_loss | n/a | 1.2687 | 0 |
| 0 | block5 | not_reached_by_cross_geometry_loss | n/a | 2.2049 | 0 |
| 0 | global_projection | not_reached_by_cross_geometry_loss | n/a | 1.6431 | 0 |
| 1 | stem | measured | -0.18473 ± 0.05414 | 2.3184 | 40.438 |
| 1 | block1 | not_reached_by_cross_geometry_loss | n/a | 1.4833 | 0 |
| 1 | block2 | not_reached_by_cross_geometry_loss | n/a | 0.94277 | 0 |
| 1 | block3 | not_reached_by_cross_geometry_loss | n/a | 0.67572 | 0 |
| 1 | block4 | not_reached_by_cross_geometry_loss | n/a | 0.94896 | 0 |
| 1 | block5 | not_reached_by_cross_geometry_loss | n/a | 1.4687 | 0 |
| 1 | global_projection | not_reached_by_cross_geometry_loss | n/a | 1.1706 | 0 |
| 2 | stem | measured | 0.30906 ± 0.07052 | 1.0445 | 25.771 |
| 2 | block1 | not_reached_by_cross_geometry_loss | n/a | 0.95409 | 0 |
| 2 | block2 | not_reached_by_cross_geometry_loss | n/a | 1.0178 | 0 |
| 2 | block3 | not_reached_by_cross_geometry_loss | n/a | 0.78575 | 0 |
| 2 | block4 | not_reached_by_cross_geometry_loss | n/a | 1.2063 | 0 |
| 2 | block5 | not_reached_by_cross_geometry_loss | n/a | 1.5947 | 0 |
| 2 | global_projection | not_reached_by_cross_geometry_loss | n/a | 1.1506 | 0 |

The high-resolution ribbon anchor branches from the stem output. Consequently, `L_CG` reaches the MRI stem but not later SFCN blocks; later-stage `not_reached` entries are architectural, not zero-valued conflict estimates.

## 8. Conclusion

- ROI identity correspondence: **supported**. Normal Acc@1 is 0.2616 versus the 62-way random value 0.0161.
- Subject-specific correspondence: **supported, but weak relative to ROI identity**. `G_subject` is positive on all folds, and subject shuffling changes mean Acc@1 from 0.2616 to 0.2577 (relative drop 1.48%). The effect decomposition should be used to judge its magnitude.
- Anatomical neighbourhood evidence: **present**. Mesh-neighbour errors account for 37.79% of mismatches versus an approximate uniform wrong-ROI expectation of 8.46% from mean graph degree.
- Phase 2.2 was not implemented. These findings only determine whether a future topology-soft experiment has real mesh-based anatomical motivation; they do not show that topology-soft supervision will improve classification.

## Reproducibility artifacts

- `audit_metrics.json`: all fold-level and aggregate numerical results
- `true_dkt_topology.npz`: mesh adjacency, boundary counts/strength, homology, names, and degrees
- `roi_topology_comparison.csv`: per-fold, per-ROI spatial/mesh neighbourhood mismatches
- `mismatch_decomposition.csv`: direction-specific and bidirectional mismatch counts/fractions

`held_out_test_accessed: false`
