# CortexBridge Graph Student Component Ablation

- Completed: `2026-10-20`.
- Protocol: matched three-fold development validation.
- Clinical input: `off`.
- Held-out test loaded/evaluated: `false`.
- Development-selected Student: **S4** (highest mean MCC across the matched three-fold development ablation).

## Variant results

| ID | Cumulative objective | MCC | AUC | Accuracy | Macro-F1 | Balanced accuracy |
|---|---|---:|---:|---:|---:|---:|
| S0 | CE only | 0.27704 ± 0.03173 | 0.71830 ± 0.01595 | 0.49560 ± 0.02572 | 0.45080 ± 0.06271 | 0.53218 ± 0.05446 |
| S3 | + logit KD | 0.30505 ± 0.01183 | 0.72422 ± 0.02664 | 0.52768 ± 0.01331 | 0.50789 ± 0.01245 | 0.55300 ± 0.01117 |
| S4 | + ROI graph-state KD | 0.31227 ± 0.04766 | 0.73894 ± 0.02805 | 0.53872 ± 0.03044 | 0.51942 ± 0.04210 | 0.55227 ± 0.03053 |
| S5 | + graph relation KD | 0.29117 ± 0.06482 | 0.72839 ± 0.04598 | 0.54821 ± 0.03775 | 0.54587 ± 0.04556 | 0.55976 ± 0.05848 |
| S6 | + prototype posterior KD | 0.29141 ± 0.04274 | 0.72518 ± 0.01846 | 0.53331 ± 0.02600 | 0.52528 ± 0.01171 | 0.55677 ± 0.01168 |

## Progressive component increments

| Increment | Fold-0 ΔMCC | Fold-1 ΔMCC | Fold-2 ΔMCC | Mean ΔMCC | Positive folds | Bootstrap 95% CI |
|---|---:|---:|---:|---:|---:|---:|
| S3_minus_S0 | +0.02762 | +0.00228 | +0.05410 | +0.02800 | 3/3 | [-0.00174, +0.09873] |
| S4_minus_S3 | +0.04900 | -0.01276 | -0.01457 | +0.00723 | 1/3 | [-0.02214, +0.03637] |
| S5_minus_S4 | -0.02680 | +0.02616 | -0.06265 | -0.02110 | 1/3 | [-0.06585, +0.02503] |
| S6_minus_S5 | -0.01864 | -0.07306 | +0.09242 | +0.00024 | 1/3 | [-0.05123, +0.03897] |

## Each KD variant versus S0

- S3: mean fold ΔMCC `+0.02800`; pooled MCC delta `+0.04778` (95% CI `[-0.00338, +0.09836]`); pooled AUC delta `+0.03460`; McNemar p `0.051481`.
- S4: mean fold ΔMCC `+0.03523`; pooled MCC delta `+0.05707` (95% CI `[+0.01133, +0.10529]`); pooled AUC delta `+0.04901`; McNemar p `0.005468`.
- S5: mean fold ΔMCC `+0.01413`; pooled MCC delta `+0.03648` (95% CI `[-0.00998, +0.08337]`); pooled AUC delta `+0.04335`; McNemar p `0.001170`.
- S6: mean fold ΔMCC `+0.01436`; pooled MCC delta `+0.03028` (95% CI `[-0.02377, +0.08252]`); pooled AUC delta `+0.02899`; McNemar p `0.022737`.

Component attribution is descriptive unless the paired uncertainty excludes zero. No held-out test result is used for model or component selection.

## Development decision

**Lock S4 for the next complete-CV stage.** It has the highest three-fold mean MCC. This selects a cumulative objective, not an independently proven causal contribution from every term.

- Keep: CE, logit KD, and ROI graph-state KD.
- Drop from the locked Student: graph-relation KD and prototype-posterior KD, because their progressive mean MCC increments are non-positive or negligible and are not stable across folds.
- Claim boundary: ROI graph-state KD has a small positive mean increment over S3 but only 1/3 folds improves and its paired bootstrap interval crosses zero.
- The held-out test remains unopened; this is a development-only selection.
- MRI-only audit: 3/3 selected checkpoints audited; all passed = `true`; all audit held-out flags false = `true`.
