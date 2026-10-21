# Falsifiable Claims

## C01: Affine-aware cortical ribbon geometry is valid
- **Statement**: The implemented white–pial sampling grid is spatially aligned with the cropped network MRI and valid for cortical sampling.
- **Status**: supported
- **Falsification criteria**: Systematic axis swap, global translation, non-finite grids, or samples outside the intended cortical ribbon in independent QC.
- **Proof**: [E01]
- **Evidence basis**: Full cache completion, five-subject overlays, valid-grid statistics, and affine-overlap audit.
- **Interpretation**: Geometry validity does not imply classification benefit.
- **Dependencies**: none
- **Tags**: geometry, affine, ribbon, QC

## C02: Pure ribbon pooling is not a validated replacement for legacy pooling
- **Statement**: Current experiments do not establish a stable classification advantage for pure ribbon or hybrid ribbon pooling over `legacy_mask`.
- **Status**: supported
- **Falsification criteria**: A locked, paired multi-fold experiment shows consistent improvement with acceptable variance and no test-driven tuning.
- **Proof**: [E02]
- **Evidence basis**: Original gate failure, inconsistent three-fold deltas, and fold-level hybrid-gate failure.
- **Interpretation**: Ribbon may still be useful as auxiliary supervision.
- **Dependencies**: C01
- **Tags**: negative-result, pooling, ablation

## C03: Hard surface–ribbon correspondence is learned
- **Statement**: Subject-local hard same-ROI supervision produces retrieval substantially above random and a positive diagonal similarity gap on every development fold.
- **Status**: supported
- **Falsification criteria**: Replication yields retrieval near random or a non-positive diagonal gap on multiple folds.
- **Proof**: [E03]
- **Evidence basis**: Three-fold Acc@1/5, MRR, diagonal/off-diagonal similarities and gap.
- **Interpretation**: This validates auxiliary correspondence, not direct ribbon classification.
- **Dependencies**: C01
- **Tags**: cross-geometry, retrieval, correspondence

## C04: Subject-specific correspondence is positive but weak
- **Statement**: Matched-subject similarity and retrieval exceed subject-shuffled controls, but the subject component is a small minority of total correspondence excess.
- **Status**: supported
- **Falsification criteria**: `G_subject` becomes non-positive or shuffle controls match/exceed normal retrieval under replicated evaluation.
- **Proof**: [E04]
- **Evidence basis**: Four-cell similarity audit, effect decomposition, and deterministic subject permutations.
- **Interpretation**: Atlas ROI identity is the dominant learned signal.
- **Dependencies**: C03
- **Tags**: subject-specific, identity, shuffle-control

## C05: Retrieval errors are anatomically structured
- **Statement**: Incorrect top-one retrievals are enriched for true DKT mesh neighbours beyond a uniform wrong-ROI reference.
- **Status**: supported
- **Falsification criteria**: A replicated mesh audit shows neighbour-error frequency no greater than a degree-matched null.
- **Proof**: [E04]
- **Evidence basis**: True mesh adjacency, mismatch decomposition, TDE/ETD, and spatial-versus-mesh comparison.
- **Interpretation**: This motivates a graph experiment but does not establish that graph training improves diagnosis.
- **Dependencies**: C01, C03
- **Tags**: topology, anatomy, mismatch

## C06: Small hard correspondence regularization does not stably harm classification
- **Statement**: At the selected small weight, three-fold classification does not show a stable decline relative to the matched legacy baseline.
- **Status**: supported
- **Falsification criteria**: Additional locked folds/seeds show consistent degradation in the primary classification endpoint.
- **Proof**: [E03]
- **Evidence basis**: Paired fold deltas and non-negative mean changes across reported classification metrics.
- **Interpretation**: The observed mean gain is too small to claim material classification improvement.
- **Dependencies**: C03
- **Tags**: safety, classification, regularization

## C07: Anatomy-prior ROI graph improves the Teacher
- **Statement**: A residual subject-adaptive ROI graph using real DKT boundary and homology priors will improve or preserve classification while producing non-degenerate graph states.
- **Status**: hypothesis
- **Falsification criteria**: The graph consistently reduces development performance, copies the prior, collapses to identity/uniform attention, or contributes no classifier signal.
- **Proof**: [E05]
- **Evidence basis**: No direct result yet; C05 supplies motivation only.
- **Interpretation**: This is the next core experiment, not a completed contribution.
- **Dependencies**: C05, C06
- **Tags**: graph, topology, teacher

## C08: Graph-state distillation improves an MRI-only Student
- **Statement**: Topology-seeded queries and distillation of Teacher graph states will outperform a matched MRI-only scratch Student without inference-time privileged inputs.
- **Status**: hypothesis
- **Falsification criteria**: The proposed Student fails to improve the primary endpoint or requires any patient-specific non-MRI input at inference.
- **Proof**: [E06]
- **Evidence basis**: No positive direct evidence; the historical method is a negative baseline.
- **Interpretation**: This claim must not enter the abstract until verified.
- **Dependencies**: C07, C10
- **Tags**: student, MRI-only, distillation

## C09: ROI-wise cognition or PET adds transferable privilege
- **Statement**: Properly audited cognition/PET could improve the Teacher and transfer useful information to the MRI-only Student.
- **Status**: hypothesis
- **Falsification criteria**: Gains disappear after leakage controls, are confined to privileged inference, or do not transfer to MRI-only predictions.
- **Proof**: [E07]
- **Evidence basis**: Not available from current vNext experiments.
- **Interpretation**: These are optional extensions, not core modules.
- **Dependencies**: C08
- **Tags**: cognition, PET, privilege, leakage

## C10: Existing MRI-only KD is unsuccessful
- **Statement**: The historical strict MRI-only KD configuration performs worse than matched MRI-only scratch.
- **Status**: supported
- **Falsification criteria**: Reanalysis of the same paired predictions reverses the reported delta or reveals a protocol mismatch.
- **Proof**: [E08]
- **Evidence basis**: Five-fold summary, paired OOF counts, bootstrap interval, and exact McNemar test.
- **Interpretation**: This negative result motivates a teacher-specific graph-state target.
- **Dependencies**: none
- **Tags**: negative-result, MRI-only, KD

