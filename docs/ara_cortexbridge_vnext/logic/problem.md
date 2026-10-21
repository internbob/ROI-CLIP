# Problem Specification

## Observations

### O1: Ribbon geometry is technically valid
- **Statement**: The full ribbon cache and five-subject coordinate QC passed, with no global translation or axis-swap pattern.
- **Evidence**: Phase 1 gate report; E01.
- **Implication**: A negative ribbon classification result cannot be dismissed as a coordinate-system failure.

### O2: Ribbon replacement is not stably better than legacy pooling
- **Statement**: Pure ribbon pooling failed the original fold-level gate and later showed only a small, inconsistent development-fold mean improvement.
- **Evidence**: Phase 1 and Phase 1.1 tables; E02.
- **Implication**: Ribbon should not replace `MaskedVolumeTokenPooling` in the classification path.

### O3: Hard cross-geometry correspondence is learnable
- **Statement**: All development folds show retrieval far above random, a positive diagonal cosine gap, and no stable classification decline at the selected small loss weight.
- **Evidence**: Phase 2.1 tables; E03.
- **Implication**: Ribbon has a defensible role as an auxiliary MRI-side cortical anchor.

### O4: Correspondence is dominated by atlas identity
- **Statement**: The subject-specific similarity increment and shuffle effect are positive but small relative to the ROI-identity component.
- **Evidence**: Phase 2.1.5 similarity and shuffle tables; E04.
- **Implication**: The method must not overclaim individualized cortical matching.

### O5: Errors contain real anatomical structure
- **Statement**: Mesh-neighbour top-one errors are strongly enriched relative to a uniform wrong-ROI expectation, while centroid-only spatial mismatches are uncommon.
- **Evidence**: Phase 2.1.5 topology/mismatch tables; E04.
- **Implication**: Real DKT topology is more naturally used for downstream graph reasoning than as a replacement for exact correspondence.

### O6: Existing MRI-only distillation is unsuccessful
- **Statement**: Historical true MRI-only KD underperforms matched MRI-only scratch with a statistically significant paired accuracy decrease.
- **Evidence**: TMI evidence package; E08.
- **Implication**: A journal paper cannot yet claim successful privileged-to-MRI-only transfer.

## Gaps

### G1: No disease-aware graph reasoning
- **Statement**: Current relation matrices are cosine diagnostics/losses, not message-passing graph modules.
- **Caused by**: O5.
- **Existing attempts**: Spatial/local relation weighting in Student losses.
- **Why they fail**: They do not produce graph-enhanced ROI states or subject-adaptive graph attention.

### G2: No graph-specific Student representation
- **Statement**: Student uses free learnable queries and distills raw Teacher volume tokens.
- **Caused by**: O6.
- **Existing attempts**: Logit, token, relation, prototype, and optional spatial KD.
- **Why they fail**: The target is not a distinctly privileged cortical graph state.

### G3: Clinical contribution is scientifically ambiguous
- **Statement**: Cognitive scales may overlap with the definition of the diagnosis target.
- **Caused by**: Current clinical-enabled baseline configuration.
- **Existing attempts**: Gated, cross-transformer, and atlas-prompt clinical fusion.
- **Why they fail**: No label-provenance/leakage audit establishes a defensible causal or deployment interpretation.

### G4: Journal-level validation is incomplete
- **Statement**: vNext evidence is mainly three development folds and lacks locked final internal/external evaluation.
- **Caused by**: Stage-wise implementation protocol.
- **Existing attempts**: Historical five-fold evidence package.
- **Why they fail**: Historical models do not include the planned graph/state-distillation method.

## Key Insight

- **Insight**: Separate the roles of MRI volume context, cortical geometry, anatomical topology, and privileged transfer. Keep legacy pooling for classification, use ribbon only to regularize ROI identity, apply real topology to a residual ROI graph, and distill the resulting privileged graph state to an MRI-only Student.
- **Derived from**: O1–O6.
- **Enables**: A compact three-module journal method with falsifiable ablations and restrained claims.

## Assumptions

- A1: DKT ROI names and ordering are consistent across modalities and folds.
- A2: The fsaverage-projected DKT mesh is an acceptable template anatomical prior when provenance is disclosed.
- A3: Surface and clinical inputs are training-time privileges; final Student inference is T1-only.
- A4: Hyperparameters are selected only on development data after the consumed fold-level test is retired from tuning.
- A5: Disease-aware means label-supervised subject-adaptive representation, not access to the ground-truth diagnosis inside graph construction.

