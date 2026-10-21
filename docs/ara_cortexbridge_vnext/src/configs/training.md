# Training Configuration

## Teacher batch size
- **Value**: `8`
- **Rationale**: Established memory-safe batch for the current volumetric Teacher.
- **Search range**: Not systematically searched in provided evidence.
- **Sensitivity**: medium
- **Source**: `configs/teacher.yaml`

## Teacher learning rate
- **Value**: `1e-4`
- **Rationale**: Locked matched baseline setting.
- **Search range**: Not specified.
- **Sensitivity**: high
- **Source**: `configs/teacher.yaml`

## Teacher weight decay
- **Value**: `1e-4`
- **Rationale**: Existing AdamW regularization.
- **Search range**: Not specified.
- **Sensitivity**: medium
- **Source**: `configs/teacher.yaml`

## Teacher selection metric
- **Value**: `MCC`
- **Rationale**: Robust primary endpoint under class imbalance.
- **Search range**: accuracy or macro-F1 are implemented alternatives.
- **Sensitivity**: high
- **Source**: `configs/teacher.yaml`

## Teacher seed
- **Value**: `3407`
- **Rationale**: Locked seed used by completed matched experiments.
- **Search range**: Journal final models require additional seed assessment.
- **Sensitivity**: high
- **Source**: Phase reports and configuration

## CE-only warmup
- **Value**: `5 epochs`
- **Rationale**: Stabilize classification before auxiliary losses.
- **Search range**: Not specified.
- **Sensitivity**: high
- **Source**: `configs/teacher.yaml`; hybrid-gate instability occurred around auxiliary activation

## ROI-CLIP loss weight
- **Value**: `0.01`
- **Rationale**: Existing legacy auxiliary setting.
- **Search range**: Not specified.
- **Sensitivity**: medium
- **Source**: `configs/teacher.yaml`

## Prototype loss weight
- **Value**: `0.01`
- **Rationale**: Existing Teacher setting.
- **Search range**: Not specified.
- **Sensitivity**: medium
- **Source**: `configs/teacher.yaml`

## Relation loss weight
- **Value**: `0.005`
- **Rationale**: Existing weak surface-volume relation consistency.
- **Search range**: Not specified.
- **Sensitivity**: medium
- **Source**: `configs/teacher.yaml`

## Cross-geometry loss weight
- **Value**: `0.01`
- **Rationale**: Smallest tested value that learned clear correspondence while preserving classification.
- **Search range**: `[0.01, 0.05, 0.10]` in completed fold sweep.
- **Sensitivity**: high
- **Source**: Phase 2.1 lambda sweep

## Cross-geometry temperature
- **Value**: `0.1`
- **Rationale**: Locked Phase 2.1 contrastive scale.
- **Search range**: Not specified.
- **Sensitivity**: medium
- **Source**: Phase 2.1 configuration

## Student current base learning rate
- **Value**: `1e-4`
- **Rationale**: Existing new-head learning rate.
- **Search range**: Not specified.
- **Sensitivity**: high
- **Source**: `configs/student.yaml`

## Student inherited/encoder learning rate
- **Value**: `2e-6`
- **Rationale**: Low-rate fine-tuning of Teacher-initialized modules.
- **Search range**: Not specified.
- **Sensitivity**: high
- **Source**: `configs/student.yaml`

## Planned graph-state KD weights
- **Value**: `CE=1.0, logit=0.5, ROI=1.0, graph=0.2, prototype=0.2, attention=0.0`
- **Rationale**: Initial implementation-guide proposal emphasizing ROI state and disabling attention KD.
- **Search range**: Must be selected on development data; no optimum is established.
- **Sensitivity**: high
- **Source**: Implementation guide Section 16.6

