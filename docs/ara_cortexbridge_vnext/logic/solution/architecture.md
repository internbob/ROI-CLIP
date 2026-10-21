# Architecture

## Current Validated Teacher Backbone

**Purpose**: Preserve strong MRI context classification while exposing ROI-level privileged outputs.

**Inputs**: T1 MRI, cortical surface features/masks, aligned DKT volume masks, optional clinical vector; ribbon grids only when cross-geometry is enabled.

**Outputs**: classification logits, projected surface/volume ROI tokens, fused Teacher tokens, prototype logits, relation matrices, optional correspondence tokens.

```text
MRI → SFCN ─┬→ global feature → optional clinical fusion → global logits
            ├→ deep feature → MaskedVolumeTokenPooling → V_context
            └→ stem feature → RibbonSampler → R_anchor (auxiliary only)

Surface → SurfaceEncoder → SurfaceROIPooling → S_surface

S_surface + V_context → ROI-CLIP → teacher_fusion → z_teacher → ROI logits
S_surface ↔ R_anchor → hard subject-local L_CG
```

**Key design choices**:

- `legacy_mask` remains the classification path.
- Ribbon anchor never replaces `V_context` and never enters the classifier.
- Cross-geometry loss is small and hard same-ROI.

## True DKT Topology Asset

**Purpose**: Supply a real anatomical prior rather than centroid-neighbour semantics.

**Inputs**: fsaverage triangular mesh and DKT annotation transferred in registered spherical space.

**Outputs**: `A_anat`, `B_boundary`, `A_boundary`, `A_hom`, homolog indices and ROI names.

**Key design choices**:

- Anatomy and bilateral homology remain separate.
- Raw boundary counts are symmetric; normalized strength is row-stochastic.
- Centroid top-three remains a diagnostic only.

## Planned Anatomy-Prior Disease-Aware Graph

**Purpose**: Model inter-regional organization on the fused privileged Teacher state.

**Inputs**: `z_teacher`, `A_anat`, `A_boundary`, `A_hom`.

**Outputs**: `z_teacher_graph`, graph attention, graph diagnostics.

```text
z_teacher → Pre-LN graph attention + additive topology bias
          → small residual gate
          → FFN residual
          → z_teacher_graph
```

**Interactions**: ROI classifier and prototype bank move to `z_teacher_graph`; global MRI classifier remains unchanged.

**Key design choices**:

- Subject tokens learn edges; topology is a bias, not a label-dependent graph.
- Remote edges remain possible.
- Start with one graph layer and no explicit graph regularizer.

## Current MRI-Only Student

**Purpose**: Predict from T1 only.

**Inputs**: `image`; clinical data only in legacy opt-in mode.

**Outputs**: logits, learnable-query ROI tokens, spatial attention, prototype and relation outputs.

```text
MRI → SFCN → feature tokens
learnable ROI queries → cross-attention → z_student → classifier
```

## Planned Topology-Query Student

**Purpose**: Recover the Teacher graph state without patient-specific privileged inputs.

**Inputs**: T1 MRI and fixed model buffers containing template graph positional encoding.

**Outputs**: `z_student_graph`, Student graph, prototype posterior and diagnosis.

```text
MRI → feature pyramid
learnable ROI identity + fixed DKT graph PE → ROI queries
ROI queries → MRI cross-attention → ROI state
ROI state → same graph architecture → z_student_graph
```

**Key design choices**:

- Forward contract is `student(image)`.
- No surface, subject atlas, cognition, PET, or ribbon grid at inference.

