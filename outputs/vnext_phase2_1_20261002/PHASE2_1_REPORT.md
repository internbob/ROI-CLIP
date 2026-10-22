# CortexBridge Phase 2.1 — Cross-Geometry Alignment Only

Date: 2026-10-03

## Decision

Phase 2.1 passes the stated gate for considering Phase 2.2, but this run stops here and does not implement topology-soft targets.

- Hard surface↔ribbon ROI correspondence is consistently learned on all three development folds.
- The same-ROI cosine gap is positive on every fold.
- Classification does not show a stable decline: paired mean MCC changes by `+0.00170`; two folds improve and one declines slightly (`-0.00311`).
- Mean AUC, accuracy, macro-F1, and balanced accuracy changes are all non-negative, providing no evidence of a material adverse effect on the shared MRI representation at `lambda_cg=0.01`.
- Graph, PET, cognitive modulation, Student KD, hybrid ribbon fusion, topology-soft targets, and direct ribbon classification are not implemented.

## Architecture implemented

The existing classification path remains:

```text
MRI -> shared MRI encoder -> deep feature
    -> MaskedVolumeTokenPooling -> V_context
    -> existing ROI-CLIP / prototypes / classifier
```

The new supervision-only path is:

```text
shared MRI encoder -> high-resolution stride-2 feature
    -> uniform inner-5 RibbonSampler -> R_anchor -> W_r -> normalize

SurfaceEncoder -> SurfaceROIPooling -> S_surface -> W_s -> normalize

S_surface <-> R_anchor: hard same-ROI correspondence loss only
```

`R_anchor` is never assigned to `volume_tokens`, never enters the classifier, and no fusion gate exists in P2-B. Same-seed tests show that enabling the cross-geometry branch leaves the legacy classification logits exactly unchanged before optimization.

The inner ribbon depths are `[0.2, 0.35, 0.5, 0.65, 0.8]`, with uniform depth averaging. The two projectors map the surface and high-resolution MRI anchor tokens into a 128-dimensional alignment space.

## Objective

For each subject independently, the implementation forms a `K x K` cosine matrix:

```text
sim[b,i,j] = normalize(W_s(S))[b,i] @ normalize(W_r(R))[b,j] / temperature
```

It uses hard ROI index targets in both directions:

```text
L_CG = 0.5 * (CE(surface->ribbon) + CE(ribbon->surface))
L = existing total loss + lambda_cg * L_CG
```

This is deliberately separate from the existing ROI-CLIP loss, which flattens subject and ROI dimensions. Cross-geometry supervision is subject-local and cannot use another subject's matching ROI as the positive target.

Gradients from `L_CG` update the surface encoder/projector, ribbon projector, and shared MRI encoder. There is no hybrid fusion gate to update.

## Retrieval metrics

All retrieval metrics are bidirectional averages. Similarity diagnostics use cosine values before temperature scaling.

- ROI retrieval Acc@1 and Acc@5 use the exact ROI index.
- MRR ranks the exact ROI index.
- Same-ROI similarity is the diagonal cosine mean.
- Off-diagonal similarity is the non-diagonal cosine mean.
- The diagonal gap is same-ROI minus off-diagonal similarity.
- Topology-aware accuracy is an evaluation-only top-1 tolerance metric: exact ROI, fixed top-3 same-hemisphere spatial neighbors, or the bilateral homologue counts as correct. It does not alter the hard training target.

For `K=62`, random expectations are Acc@1 `0.01613`, Acc@5 `0.08065`, and MRR `0.07601`. The approximate topology-aware random expectation is about `0.072` for the fixed evaluation neighborhoods.

## Fold-0 lambda sweep

All choices use development validation only; no held-out test was evaluated.

| lambda_cg | Best epoch | MCC | AUC | Acc@1 | Acc@5 | MRR | Gap | Topology Acc |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `0.01` | 6 | `0.71169` | `0.93669` | `0.27560` | `0.65456` | `0.44540` | `0.30577` | `0.50554` |
| `0.05` | 11 | `0.70907` | `0.93348` | `0.35242` | `0.73515` | `0.52092` | `0.41198` | `0.58022` |
| `0.10` | 6 | `0.70673` | `0.93597` | `0.34812` | `0.72923` | `0.51645` | `0.38026` | `0.58043` |

Larger weights strengthen retrieval but perturb classification more. The predeclared selection rule was to choose the smallest weight that learns clear correspondence while preserving legacy classification, so `lambda_cg=0.01` was selected for fold-1/2.

## Three-fold P2-A versus P2-B

P2-A is the locked three-fold legacy development baseline from Phase 1.1. It is reusable because P2-A is the identical legacy configuration, and regression/dry-run checks prove the disabled optional branch preserves common initialization and logits exactly.

| Fold | P2-A MCC | P2-B MCC | Delta MCC | P2-B AUC | Acc@1 | Acc@5 | MRR | Diag sim | Off-diag sim | Gap | Topology Acc |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | `0.71098` | `0.71169` | `+0.00071` | `0.93669` | `0.27560` | `0.65456` | `0.44540` | `0.33221` | `0.02644` | `0.30577` | `0.50554` |
| 1 | `0.70445` | `0.71195` | `+0.00750` | `0.93574` | `0.26597` | `0.64288` | `0.43560` | `0.33965` | `0.01930` | `0.32034` | `0.49465` |
| 2 | `0.67438` | `0.67127` | `-0.00311` | `0.92891` | `0.24323` | `0.62173` | `0.41310` | `0.32470` | `0.03565` | `0.28905` | `0.46273` |
| Mean | `0.69660` | `0.69830` | `+0.00170` | `0.93378` | `0.26160` | `0.63972` | `0.43137` | `0.33219` | `0.02713` | `0.30506` | `0.48764` |

Across-fold sample SDs are `0.00537` for paired MCC delta, `0.01662` for Acc@1, `0.01664` for Acc@5, `0.01656` for MRR, and `0.01566` for the diagonal gap.

Mean paired classification changes for P2-B versus P2-A:

| Metric | P2-A mean | P2-B mean | Mean delta |
|---|---:|---:|---:|
| MCC | `0.69660` | `0.69830` | `+0.00170` |
| AUC | `0.93334` | `0.93378` | `+0.00045` |
| Accuracy | `0.80854` | `0.80928` | `+0.00075` |
| Macro-F1 | `0.79787` | `0.79880` | `+0.00094` |
| Balanced accuracy | `0.80238` | `0.80681` | `+0.00443` |

## Gate assessment

| Phase 2.2 entry condition | Evidence | Result |
|---|---|---|
| Correspondence clearly above random | Mean Acc@1 `0.2616` vs `0.0161`; Acc@5 `0.6397` vs `0.0806`; MRR `0.4314` vs `0.0760` | PASS |
| Same-ROI gap stably positive | Fold gaps `0.3058`, `0.3203`, `0.2891` | PASS |
| No stable three-fold classification decline | MCC deltas `+0.0007`, `+0.0075`, `-0.0031`; mean `+0.0017` | PASS |
| No clear adverse shared-MRI effect | All five reported classification metrics have non-negative mean deltas | PASS |

This gate means Phase 2.2 can be considered in a separate, explicitly authorized stage. It is not evidence that topology-soft training itself will help, and Phase 2.2 is not started here.

## Verification

- Unit suite before full training: `34 passed`.
- P2-A/P2-B dry-runs completed and had identical initial classification CE/metrics.
- P2-B peak allocated memory was about `17.31 GiB` on an A100 80 GB GPU.
- Every full P2-B development result records `held_out_test_evaluated: false`.
- No Graph, PET, cognitive modulation, Student KD, hybrid fusion, or topology-soft target was added.
