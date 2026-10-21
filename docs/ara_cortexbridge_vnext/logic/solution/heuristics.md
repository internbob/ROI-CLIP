# Design Heuristics

## H01: Keep ribbon supervision auxiliary
- **Rationale**: Pure and hybrid ribbon classification lack stable gains, while correspondence is reliable.
- **Sensitivity**: high
- **Bounds**: Ribbon anchor must not replace legacy volume tokens or directly enter the classifier in the locked baseline.
- **Code ref**: [src/execution/method_spec.py]
- **Source**: Phase 1, Phase 1.1, and Phase 2.1 reports

## H02: Use the smallest effective correspondence weight
- **Rationale**: Larger weights strengthen retrieval but perturb classification more.
- **Sensitivity**: high
- **Bounds**: Keep the selected small weight for matched graph ablations; any new sweep uses development data only.
- **Code ref**: [src/execution/method_spec.py]
- **Source**: Phase 2.1 lambda sweep

## H03: Preserve exact ROI identity
- **Rationale**: ROI identity explains nearly all correspondence excess.
- **Sensitivity**: high
- **Bounds**: Any topology-soft component remains a minority mixture with hard identity.
- **Code ref**: [src/execution/method_spec.py]
- **Source**: Phase 2.1.5 similarity decomposition

## H04: Use true boundary topology before centroid proximity
- **Rationale**: Centroid top-three mostly overlaps mesh neighbours, and spatial-only mismatch is small.
- **Sensitivity**: medium
- **Bounds**: `A_anat/A_boundary` are primary; centroid spatial affinity is optional ablation only.
- **Code ref**: [src/execution/method_spec.py]
- **Source**: Phase 2.1.5 topology comparison

## H05: Add topology as bias, not a hard mask
- **Rationale**: Remote inter-regional disease relationships may be discriminative.
- **Sensitivity**: medium
- **Bounds**: Preserve finite logits for remote ROI pairs; initialize graph residual contribution small.
- **Code ref**: [src/execution/method_spec.py]
- **Source**: Implementation guide Sections 7 and 29

## H06: Isolate graph architecture before adding graph losses
- **Rationale**: Strong KL/entropy regularization can make attention copy the prior or collapse.
- **Sensitivity**: medium
- **Bounds**: First graph experiment uses classification/prototype objectives and diagnostics without strong explicit graph regularization.
- **Code ref**: [src/execution/method_spec.py]
- **Source**: Implementation guide Section 15.3

## H07: Gate Student work on Teacher graph validity
- **Rationale**: Distilling an unvalidated representation repeats the historical weak-target failure.
- **Sensitivity**: high
- **Bounds**: E06 starts only after E05 passes its classification and non-degeneracy gates.
- **Code ref**: [src/execution/method_spec.py]
- **Source**: Historical MRI-only KD evidence and implementation roadmap

## H08: Keep Student inference physically image-only
- **Rationale**: Configuration flags alone are insufficient to prove deployment constraints.
- **Sensitivity**: high
- **Bounds**: Remove privileged tensors from the inference dataloader and test `forward(image)`.
- **Code ref**: [src/execution/method_spec.py]
- **Source**: Implementation guide constraint 0.1 and Student contract tests

