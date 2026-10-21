# Model Configuration

## DKT ROI count
- **Value**: `62`
- **Rationale**: Thirty-one retained cortical DKT parcels per hemisphere.
- **Search range**: Fixed by atlas.
- **Sensitivity**: low
- **Source**: data/model configuration and topology audit

## MRI input shape
- **Value**: `[160, 192, 160]`
- **Rationale**: Existing aligned network grid.
- **Search range**: Not searched.
- **Sensitivity**: high
- **Source**: `configs/data.yaml`

## MRI base channels
- **Value**: `32`
- **Rationale**: Existing SFCN encoder capacity.
- **Search range**: Not specified.
- **Sensitivity**: medium
- **Source**: `configs/teacher.yaml`

## Feature/projection dimension
- **Value**: `128 / 128`
- **Rationale**: Existing shared ROI representation sizes.
- **Search range**: Not specified.
- **Sensitivity**: medium
- **Source**: Teacher configuration

## Classification pooling mode
- **Value**: `legacy_mask`
- **Rationale**: Pure/hybrid ribbon has no stable advantage.
- **Search range**: `legacy_mask`, `ribbon`, `hybrid_ribbon` are implemented.
- **Sensitivity**: high
- **Source**: Phase 1/1.1 decisions

## Ribbon depths
- **Value**: `[0.2, 0.35, 0.5, 0.65, 0.8]`
- **Rationale**: Avoid white/pial boundary contamination and use the validated inner-five anchor.
- **Search range**: Alternative depth counts are supported; no full sweep establishes an optimum.
- **Sensitivity**: medium
- **Source**: Phase 1.1/2.1 configuration

## Ribbon depth aggregation
- **Value**: uniform mean
- **Rationale**: Depth features remain highly correlated; attention was not needed for the auxiliary anchor.
- **Search range**: uniform or learned depth attention.
- **Sensitivity**: medium
- **Source**: Phase 1.1 diagnostics

## Cross-geometry projection dimension
- **Value**: `128`
- **Rationale**: Matches the existing ROI projection space.
- **Search range**: Not specified.
- **Sensitivity**: low
- **Source**: Phase 2.1 configuration

## Planned graph depth
- **Value**: `1 layer` initially
- **Rationale**: Isolate the minimal graph contribution and avoid oversmoothing.
- **Search range**: one layer first; two layers only after positive evidence.
- **Sensitivity**: high
- **Source**: Journal roadmap recommendation

## Planned graph heads
- **Value**: `4`
- **Rationale**: Compact multi-head subject-adaptive relations at fixed small ROI count.
- **Search range**: Not validated.
- **Sensitivity**: medium
- **Source**: Implementation guide / roadmap proposal

## Planned graph residual gate
- **Value**: initialized near `0.1`
- **Rationale**: Preserve the locked baseline at initialization as closely as practical.
- **Search range**: small positive initialization only.
- **Sensitivity**: high
- **Source**: Implementation guide residual principle

## Student inference clinical flag
- **Value**: `false`
- **Rationale**: Enforce strict MRI-only inference.
- **Search range**: No search; deployment constraint.
- **Sensitivity**: high
- **Source**: `configs/student.yaml` and MRI-only contract

