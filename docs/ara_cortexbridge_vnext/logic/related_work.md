# Related Work and Technical Dependencies

The provided inputs do not contain a bibliography. DOI metadata is therefore marked unavailable rather than inferred. A submission-ready literature review remains required.

## RW01: ROI-CLIP repository baseline
- **DOI**: Not available from provided input
- **Type**: baseline
- **Delta**:
  - What changed: CortexBridge retains ROI projection/prototype foundations and adds affine-aware cortical ribbon anchoring plus planned topology graph/state distillation.
  - Why: Existing surface/volume interaction is projection alignment plus concatenation rather than explicit cortical geometry and graph reasoning.
- **Claims affected**: C02, C03, C07, C08
- **Adopted elements**: ROI tokens, ROI-CLIP projectors, prototype bank, Teacher–Student structure

## RW02: FreeSurfer DKT atlas and surface geometry
- **DOI**: Not available from provided input
- **Type**: imports
- **Delta**:
  - What changed: DKT annotation, white/pial geometry, and fsaverage mesh are converted into network-aligned ribbon grids and parcel topology.
  - Why: Surface coordinates and MRI tensor coordinates cannot be equated directly.
- **Claims affected**: C01, C05
- **Adopted elements**: cortical surfaces, DKT parcel identities, registered spherical coordinates

## RW03: SFCN-style volumetric encoding
- **DOI**: Not available from provided input
- **Type**: imports
- **Delta**:
  - What changed: Intermediate high-resolution features support ribbon anchors; deep features retain legacy ROI context.
  - Why: Deep features lack sufficient depth-axis resolution for direct cortical ribbon replacement.
- **Claims affected**: C02, C03
- **Adopted elements**: staged 3D convolutional encoder and global projection

## RW04: Knowledge distillation baselines
- **DOI**: Not available from provided input
- **Type**: bounds
- **Delta**:
  - What changed: Planned KD target moves from raw volume tokens to a privileged graph-enhanced cortical state.
  - Why: Existing strict MRI-only KD underperforms scratch.
- **Claims affected**: C08, C10
- **Adopted elements**: logit, token, relation, and prototype distillation primitives

## Literature gap for submission

A formal search must add verified citations for cortical surface learning, surface-volume correspondence, graph neural networks in neuroimaging, privileged information/distillation, MRI-only cognitive diagnosis, DKT/FreeSurfer, SFCN, and calibration/external validation. No citation should be generated from memory without verification.

