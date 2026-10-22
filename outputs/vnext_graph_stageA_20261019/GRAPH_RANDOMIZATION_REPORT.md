# CortexBridge Graph Topology-Randomization Control

- True topology variant: `g3`.
- Shuffled control: `g3r`.
- Clinical input: `off`.
- Held-out test loaded/evaluated: `false`.

## Fold-level true − randomized deltas

| Metric | Fold deltas | Mean delta |
|---|---:|---:|
| mcc | -0.00825, -0.01003, 0.06002 | 0.01391 |
| auc | -0.00844, -0.02519, 0.01890 | -0.00491 |
| acc | -0.00881, 0.00875, 0.04283 | 0.01426 |
| f1_macro | -0.00478, 0.02966, 0.08643 | 0.03711 |
| balanced_acc | 0.00614, 0.02389, 0.07346 | 0.03450 |
| graph_anatomical_mass | 0.01408, 0.01429, 0.01345 | 0.01394 |
| graph_homology_mass | 0.02572, 0.02728, 0.02570 | 0.02623 |
| graph_boundary_correlation | 0.26055, 0.25124, 0.20087 | 0.23755 |
| graph_remote_mass | -0.04120, -0.03869, -0.04034 | -0.04008 |

## Gate

- No material shuffled-topology advantage: `true`.
- Anatomy-specific evidence: `false`.
- Student stage authorized: `true`.

Anatomy-specific evidence requires true topology to improve MCC in at least two folds and to increase true boundary/anatomy attention diagnostics. If this condition fails, the manuscript must not attribute a graph benefit specifically to the DKT prior.
