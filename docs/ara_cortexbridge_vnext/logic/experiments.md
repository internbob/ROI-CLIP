# Experiment Registry and Plans

Exact completed results are stored only in `evidence/`; expected outcomes below are directional.

## E01: Ribbon cache and coordinate QC
- **Verifies**: C01
- **Setup**: ADNI DKT ribbon cache, seeded subject overlays, affine-aware cropped MRI geometry.
- **Procedure**:
  1. Precompute white–pial grids and validity fields.
  2. Validate finite coordinates, in-volume sampling, thickness, and visual overlays.
  3. Compare foreground overlap before and after affine correction.
- **Metrics**: cache success rate, valid fraction, thickness, foreground overlap, visual failure count.
- **Expected outcome**: Correct affine-aware geometry improves overlap and shows no global coordinate error.
- **Baselines**: pre-affine cache and invalid direct-coordinate normalization.
- **Dependencies**: none

## E02: Legacy versus ribbon pooling
- **Verifies**: C02
- **Setup**: Matched Teacher configurations differing only in volume pooling or hybridization.
- **Procedure**:
  1. Compare legacy, deep ribbon, high-resolution inner ribbon, and hybrid residual variants.
  2. Use paired development folds and preserve common initialization.
  3. Audit depth separability and gate dynamics.
- **Metrics**: MCC, AUC, accuracy, macro-F1, depth cosine/distance, gate value, memory.
- **Expected outcome**: A replacement is accepted only if it improves consistently without unstable gating.
- **Baselines**: legacy mask pooling.
- **Dependencies**: E01

## E03: Hard cross-geometry alignment
- **Verifies**: C03, C06
- **Setup**: Legacy classification path plus auxiliary high-resolution surface–ribbon correspondence.
- **Procedure**:
  1. Sweep a small set of correspondence weights on development validation.
  2. Select the smallest weight that learns correspondence while preserving classification.
  3. Repeat the selected setting across development folds.
- **Metrics**: Acc@1/5, MRR, diagonal/off-diagonal similarity, similarity gap, MCC, AUC, accuracy, macro-F1.
- **Expected outcome**: Retrieval exceeds random, the gap is positive, and classification does not decline consistently.
- **Baselines**: identical legacy Teacher without cross-geometry loss.
- **Dependencies**: E02

## E04: Correspondence and topology audit
- **Verifies**: C04, C05
- **Setup**: Frozen Phase 2.1 checkpoints, validation only, no optimizer.
- **Procedure**:
  1. Compute four subject/ROI similarity cells.
  2. Run deterministic subject derangements.
  3. Build true DKT mesh adjacency/boundary/homology.
  4. Decompose mismatches and compute graph distances.
  5. Diagnose MRI encoder gradient cosine without updates.
- **Metrics**: identity/subject contrast, shuffled retrieval, adjacency statistics, mismatch fractions, TDE, ETD, gradient cosine.
- **Expected outcome**: The audit distinguishes atlas identity, subject-specific information, and anatomical-neighbour structure.
- **Baselines**: random retrieval and centroid spatial tolerance.
- **Dependencies**: E03

## E05: Anatomy-prior Teacher graph ablation
- **Verifies**: C07
- **Setup**: Frozen P2-B protocol extended by a residual ROI graph on fused Teacher tokens.
- **Procedure**:
  1. Compare no graph, graph without prior, boundary/anatomy prior, and anatomy plus homology.
  2. Use identical development folds and selection protocol.
  3. Inspect attention entropy, anatomical mass, homology mass, and classifier contribution.
- **Metrics**: classification endpoints, graph diagnostics, fold consistency, compute/memory.
- **Expected outcome**: The proposed topology-biased graph preserves or improves diagnosis and remains subject-adaptive rather than copying the prior.
- **Baselines**: P2-B and graph without topology.
- **Dependencies**: E04

## E06: MRI-only topology-query and graph-state distillation
- **Verifies**: C08
- **Setup**: Locked graph Teacher and strict image-only Student dataloader.
- **Procedure**:
  1. Re-run matched MRI-only scratch.
  2. Add topology-seeded queries.
  3. Add logit, graph-state, graph-distribution, and prototype KD incrementally.
  4. Audit that all privileged inputs can be physically removed at inference.
- **Metrics**: MRI-only classification, calibration, paired OOF delta, deployment latency, inference contract tests.
- **Expected outcome**: Graph-state targets outperform scratch and old raw-volume-token KD without privileged inference inputs.
- **Baselines**: MRI-only scratch and historical/current-query KD.
- **Dependencies**: E05

## E07: Clinical leakage and optional privileged modalities
- **Verifies**: C09
- **Setup**: Label provenance audit, clinical-only/imaging-only controls, and optional real PET cohort.
- **Procedure**:
  1. Identify variables used directly or indirectly in target diagnosis.
  2. Compare safe-demographic and full-clinical settings.
  3. Add ROI-wise cognition only after exclusions are locked.
  4. Add PET only with real registered images and fold-local statistics.
- **Metrics**: classification, transfer to MRI-only Student, missing-modality robustness, modality gates.
- **Expected outcome**: Any retained privileged modality adds transferable information after leakage controls.
- **Baselines**: clinical-only, MRI-only, surface+MRI without clinical.
- **Dependencies**: E06

## E08: Historical MRI-only KD comparison
- **Verifies**: C10
- **Setup**: Historical five-fold strict MRI-only scratch and KD predictions.
- **Procedure**:
  1. Aggregate out-of-fold predictions.
  2. Compute paired correctness and bootstrap accuracy interval.
  3. Apply an exact paired classification test.
- **Metrics**: ACC, AUC, MCC, macro-F1, paired delta, confidence interval, exact p-value.
- **Expected outcome**: The comparison determines whether the existing KD implementation transfers useful privilege.
- **Baselines**: MRI-only scratch.
- **Dependencies**: none

