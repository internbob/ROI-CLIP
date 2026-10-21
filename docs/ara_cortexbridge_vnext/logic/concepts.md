# Concepts

## Legacy Volume Context Token
- **Notation**: \(V_{context}\in\mathbb{R}^{B\times K\times C}\)
- **Definition**: ROI token produced by pooling deep MRI features with aligned local-ribbon and full-ROI volume masks followed by MLP fusion.
- **Boundary conditions**: Used in the classification path; it is not a white–pial sampled token.
- **Related concepts**: Ribbon Anchor, ROI-CLIP

## Ribbon Anchor
- **Notation**: \(R_{anchor}\in\mathbb{R}^{B\times K\times C_r}\)
- **Definition**: ROI token obtained by sampling a high-resolution MRI feature map at multiple inner cortical depths and uniformly aggregating depths and vertices.
- **Boundary conditions**: Auxiliary supervision only in Phase 2.1; never substituted for `V_context` or passed to the classifier.
- **Related concepts**: Cross-Geometry Loss, Cortical Ribbon

## Subject-Local Cross-Geometry Loss
- **Notation**: \(L_{CG}\)
- **Definition**: Symmetric cross-entropy over each subject's ROI-by-ROI surface/ribbon similarity matrix with exact ROI index as target.
- **Boundary conditions**: Positives and negatives are formed within each subject; it is distinct from flattened ROI-CLIP.
- **Related concepts**: Ribbon Anchor, ROI Identity

## ROI Identity Component
- **Notation**: \(G_{ROI}\)
- **Definition**: Different-subject/same-ROI similarity minus different-subject/different-ROI similarity.
- **Boundary conditions**: A descriptive contrast, not causal variance explained.
- **Related concepts**: Subject-Specific Increment

## Subject-Specific Increment
- **Notation**: \(G_{subject}\)
- **Definition**: Same-subject/same-ROI similarity minus different-subject/same-ROI similarity.
- **Boundary conditions**: Positive values indicate information beyond atlas identity, but magnitude must be interpreted relative to total correspondence excess.
- **Related concepts**: ROI Identity Component, Subject Shuffle

## Anatomical DKT Adjacency
- **Notation**: \(A_{anat}\)
- **Definition**: Binary parcel adjacency induced by triangular mesh edges whose endpoint vertices carry different valid DKT labels.
- **Boundary conditions**: Within-hemisphere and zero-diagonal; it does not include bilateral homology.
- **Related concepts**: Boundary Strength, Homology Matrix

## Boundary Strength
- **Notation**: \(A_{boundary}\)
- **Definition**: Row-normalized count of mesh edges shared by each adjacent parcel pair.
- **Boundary conditions**: Directional after row normalization although raw counts are symmetric.
- **Related concepts**: Anatomical DKT Adjacency

## Bilateral Homology Matrix
- **Notation**: \(A_{hom}\)
- **Definition**: Validated one-to-one mapping between left and right DKT parcels with the same canonical base name.
- **Boundary conditions**: Stored separately from cortical mesh adjacency.
- **Related concepts**: Anatomical DKT Adjacency

## Disease-Aware ROI Graph
- **Notation**: \(Z_G=\mathrm{Graph}(Z,A)\)
- **Definition**: Planned subject-adaptive attention/message-passing module whose logits receive additive template anatomy/homology bias and whose weights are learned from ROI tokens under diagnosis supervision.
- **Boundary conditions**: Must not select edges using the ground-truth label and must allow remote learned interactions.
- **Related concepts**: Graph-State Distillation

## Graph-State Distillation
- **Notation**: \(L_{ROI}+L_{graphKD}+L_{protoKD}\)
- **Definition**: Planned transfer from privileged Teacher graph tokens/relations/prototypes to corresponding MRI-only Student graph outputs.
- **Boundary conditions**: Teacher targets are detached; Student inference receives only T1 MRI and fixed template buffers.
- **Related concepts**: Disease-Aware ROI Graph, Topology-Seeded Query

## Topology-Seeded Query
- **Notation**: \(Q=E_{ROI}+W_{PE}U_{DKT}\)
- **Definition**: Planned Student ROI queries combining learnable ROI identity with fixed template graph positional encoding.
- **Boundary conditions**: No subject-specific atlas or surface may be read at inference.
- **Related concepts**: Graph-State Distillation

