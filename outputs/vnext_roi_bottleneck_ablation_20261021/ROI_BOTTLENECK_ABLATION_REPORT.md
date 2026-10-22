# CortexBridge ROI Bottleneck Ablation

Date: 2026-10-21

## Scope and integrity

This development-only study tests whether converting the deep MRI feature map into ROI tokens and then a graph weakens classification. No held-out test split was loaded or evaluated. The locked G3 Teacher and downstream task definition were unchanged.

Two complementary protocols are reported: D0–D4 switch branches inside already-trained checkpoints without retraining; B0–B4 retrain matched Students to isolate each component.

## D0–D4 checkpoint branch audit (no training)

| ID | Branch at inference | MCC | AUC | ACC | Macro-F1 | Recall/SEN | Specificity |
|---|---|---:|---:|---:|---:|---:|---:|
| D0 | Current S4 fused | 0.31227 ± 0.04766 | 0.73894 ± 0.02805 | 0.53872 ± 0.03044 | 0.51942 ± 0.04210 | 0.55227 ± 0.03053 | 0.75617 ± 0.01551 |
| D1 | S4 global only | 0.32201 ± 0.04332 | 0.73897 ± 0.02858 | 0.54014 ± 0.02714 | 0.52065 ± 0.03558 | 0.55980 ± 0.02670 | 0.75849 ± 0.01423 |
| D2 | S4 ROI only | 0.00000 ± 0.00000 | 0.53481 ± 0.02115 | 0.45654 ± 0.01800 | 0.20891 ± 0.00562 | 0.33333 ± 0.00000 | 0.66667 ± 0.00000 |
| D3 | S4 global + pre-graph ROI | 0.31152 ± 0.04821 | 0.73888 ± 0.02804 | 0.53872 ± 0.03044 | 0.52004 ± 0.04154 | 0.55213 ± 0.03072 | 0.75599 ± 0.01570 |
| D4 | S3 global only | 0.29766 ± 0.01430 | 0.72467 ± 0.02743 | 0.50737 ± 0.01867 | 0.47880 ± 0.02302 | 0.54853 ± 0.01161 | 0.74781 ± 0.00219 |

D0 reproduces the saved historical probabilities exactly; maximum absolute difference: `0`.

### Branch diagnostics

- Learned ROI-logit residual weight: `0.09943 ± 0.00028`.
- Learned graph residual gate: `0.09945 ± 0.00037`.
- Pre/post-graph off-diagonal ROI cosine: `0.92493 ± 0.03053` / `0.91967 ± 0.03679`.
- Pre/post-graph token variance: `0.00058 ± 0.00023` / `0.00070 ± 0.00037`.
- Graph delta relative norm: `0.05924 ± 0.01481`.
- Global/ROI logit norm: `2.07595 ± 0.18365` / `0.61552 ± 0.17108`.

## B0–B4 matched three-fold retraining

| ID | Training configuration | MCC | AUC | ACC | Macro-F1 | Recall/SEN | Specificity |
|---|---|---:|---:|---:|---:|---:|---:|
| B0 | global CE only | 0.24877 ± 0.03390 | 0.70297 ± 0.02411 | 0.49787 ± 0.01048 | 0.49108 ± 0.01481 | 0.53874 ± 0.04302 | 0.74310 ± 0.00954 |
| B1 | global CE + logit KD | 0.32870 ± 0.03922 | 0.74623 ± 0.02770 | 0.56490 ± 0.04083 | 0.55806 ± 0.04940 | 0.57938 ± 0.03251 | 0.76816 ± 0.01586 |
| B2 | B1 + raw ROI-token classifier | 0.25861 ± 0.04353 | 0.72076 ± 0.02814 | 0.52378 ± 0.05136 | 0.50347 ± 0.05271 | 0.52088 ± 0.02754 | 0.74515 ± 0.01641 |
| B3 | B2 + ROI graph | 0.25133 ± 0.03680 | 0.70350 ± 0.04079 | 0.49369 ± 0.05418 | 0.49384 ± 0.05769 | 0.54661 ± 0.03234 | 0.74326 ± 0.01262 |
| B4 | B3 + ROI-state KD | 0.31729 ± 0.03137 | 0.74272 ± 0.02342 | 0.54726 ± 0.02830 | 0.53928 ± 0.04806 | 0.57010 ± 0.03287 | 0.76087 ± 0.01593 |

All values are fold mean ± sample SD. Recall is macro recall (multiclass sensitivity/balanced accuracy); specificity is macro one-vs-rest specificity.

### Fold-level MCC

| ID | Fold 0 | Fold 1 | Fold 2 | Mean |
|---|---:|---:|---:|---:|
| B0 | 0.20986 | 0.27189 | 0.26457 | 0.24877 |
| B1 | 0.33398 | 0.36501 | 0.28710 | 0.32870 |
| B2 | 0.20868 | 0.28860 | 0.27855 | 0.25861 |
| B3 | 0.22417 | 0.29321 | 0.23660 | 0.25133 |
| B4 | 0.35025 | 0.28781 | 0.31381 | 0.31729 |

### Progressive paired comparisons

Pooled intervals use paired subject bootstrap over matched validation predictions. They complement, rather than replace, three-fold direction and sample SD.

| Component added | Contrast | Fold ΔMCC | Mean ΔMCC | Positive folds | Pooled ΔMCC (95% CI) | Pooled ΔAUC (95% CI) | McNemar p |
|---|---|---|---:|---:|---:|---:|---:|
| logit KD | B1 − B0 | +0.12412, +0.09312, +0.02253 | +0.07993 | 3/3 | +0.08060 [+0.03633, +0.12456] | +0.04397 [+0.02788, +0.05941] | 0.000005 |
| raw ROI classification residual | B2 − B1 | -0.12530, -0.07642, -0.00855 | -0.07009 | 0/3 | -0.06880 [-0.10959, -0.02818] | -0.02096 [-0.03474, -0.00691] | 0.001741 |
| ROI graph | B3 − B2 | +0.01549, +0.00461, -0.04195 | -0.00728 | 2/3 | -0.00668 [-0.04683, +0.03462] | -0.01326 [-0.02704, +0.00032] | 0.031494 |
| ROI-state KD | B4 − B3 | +0.12608, -0.00540, +0.07721 | +0.06596 | 2/3 | +0.06339 [+0.01712, +0.10807] | +0.02582 [+0.00864, +0.04282] | 0.000581 |
| complete ROI path and ROI-state KD | B4 − B1 | +0.01627, -0.07721, +0.02671 | -0.01141 | 2/3 | -0.01208 [-0.04713, +0.02192] | -0.00840 [-0.01922, +0.00223] | 0.137474 |

## Interpretation and mainline decision

**Select B1 as the revised Student classification mainline: global MRI classification + locked-Teacher logit KD, with the Student ROI classification residual and Student ROI graph disabled.**

The causal evidence is consistent across the two protocols. In the existing S4 checkpoints, D1 global-only improves mean MCC over D0 fused (`+0.00974`), whereas the ROI-only branch D2 is at chance and D3 is nearly identical to D0. In matched retraining, B2 loses `0.07009` mean MCC versus B1 in all three folds; its paired pooled 95% CI excludes zero. Because B2 already classifies raw, unnormalized ROI tokens, this loss cannot be explained only by the former unit-norm bottleneck.

The graph is not the primary cause. B3 changes mean MCC by only `-0.00728` versus B2, improves two of three folds, and its paired MCC interval crosses zero. The learned graph residual is small (about 5.9% of token norm), while bypassing the graph in D3 barely changes D0. The graph may be retained for representation analysis, but it is not justified in the classification path by these results.

ROI-state KD is useful as an auxiliary representation constraint but does not justify the full ROI classification path. B4 recovers `+0.06596` mean MCC over B3 and its pooled MCC interval is positive; however, B4 remains `-0.01141` below B1 on mean MCC, with heterogeneous fold effects and a CI spanning zero. Thus B4 is an informative auxiliary/anatomical ablation, not the selected classifier.

The checkpoint audit also indicates that ROI-state KD did not damage the global encoder: D1 (S4 global-only) exceeds D4 (S3 global-only) on mean MCC. The performance loss appears when weak ROI logits are fused back into the final classifier, not simply from learning an ROI-aligned representation.

The B experiments use raw projected ROI tokens for classification and normalized copies only for graph-state KD. This explicitly removes the unintended unit-norm classification bottleneck while preserving the teacher-alignment interface.

These are development-only selection results. The held-out test remains unopened and should be evaluated once, only after the revised B1 mainline and manuscript comparison protocol are frozen.

## Verification

- All 15 B0–B4 fold records set `held_out_test_evaluated: false`.
- D0 exactly reproduces historical S4 validation probabilities.
- Teacher checkpoints, folds, seed, optimizer schedule, early stopping and MCC checkpoint selection are matched.
- B0/B1 disable Student ROI classification and Student graph without changing the locked G3 Teacher.
- B2 adds raw ROI classification; B3 adds graph; B4 adds ROI-state KD.
