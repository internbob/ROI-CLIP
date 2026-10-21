# Constraints and Limitations

## Scientific constraints

- Ribbon geometry validity is established; ribbon classification superiority is not.
- Hard correspondence is dominated by atlas identity and must not be described as mainly individualized.
- Real topology motivates graph modelling but does not establish a graph benefit before E05.
- Disease-aware graph weights must not read ground-truth labels as graph inputs.
- Clinical scales that define or closely proxy diagnosis require exclusion/leakage analysis.
- PET may be used only when real registered PET exists; MRI cannot serve as a placeholder.

## Data constraints

- All normalization, prototype initialization, covariance, or population priors with learned statistics must be training-fold local.
- The previously inspected fold-level test is unavailable for future model selection.
- Internal cross-validation and external zero-shot evaluation are distinct regimes.
- The current true topology annotation is transferred from bundled FreeSurfer `bert` to fsaverage because node libraries cannot run `mris_ca_label`; provenance must be disclosed.

## Deployment constraints

- Final Student inference input is T1 MRI only.
- Fixed template topology/graph positional encodings may be model buffers.
- Subject-specific surface, ROI masks, ribbon grids, cognition, PET, and clinical vectors are forbidden at Student inference.

## Method limitations

- Cross-geometry MRI gradients reach only the stem at the current anchor location.
- Adjacent ribbon-depth features remain highly correlated even at the high-resolution stage.
- Current relation loss is not graph reasoning.
- Current Student uses unconstrained learnable queries and raw-volume-token KD.
- Current vNext evidence covers three development folds rather than a locked final evaluation.

## Claim limitations

- Do not claim improved diagnosis from Phase 2.1 based on its small mean delta.
- Do not call centroid tolerance anatomical topology accuracy.
- Do not claim successful MRI-only privileged transfer from historical KD.
- Do not claim AD propagation mechanisms from learned attention without independent biological validation.

## Missing journal metadata

- Cohort inclusion/exclusion, class counts, repeat-scan handling, diagnostic-label provenance, scanner/site distributions, ethics, and split-level subject isolation are not fully specified in the provided inputs.
- These items must be supplied from authoritative study records; they cannot be reconstructed from model code.
