# CortexBridge P2/P3 Privileged-KD Protocol

## Objective

Test whether privileged clinical knowledge, alone or together with the existing cortical geometry branch, improves a locked MRI-only B1 Student. This stage does not test inference-time multimodal fusion and does not implement P4/P5.

## Locked development protocol

- Task: NC/MCI/DE three-class diagnosis.
- Splits: three subject-level development folds (`cross0`–`cross2`).
- Longitudinal leakage: train/validation patient overlap is zero in every fold.
- Held-out test: not loaded or evaluated.
- Model selection: validation MCC, at most 30 epochs, early-stop patience 4.
- Student: SFCN global feature and global classifier only.
- Student inference input: MRI only.
- Student ROI classification residual: disabled.
- Student graph classification branch: disabled.
- KD: temperature-4 logit KL, weight 0.5, plus ground-truth CE.

## Clean privileged clinical variables

The P2 and P3 Teachers use the same 16 variables:

`age`, `education`, `apoe`, `gds`, and the 12 `npiq_*` domain scores.

MMSE, CDR, ADAS-Cog, FAQ, RAVLT, logical-memory scores, and category-fluency scores are excluded from clean privileged input because of their diagnosis-proximal role. Their potential use as training-only auxiliary targets is a separate future experiment.

Clinical imputation and standardization statistics are fitted on the training fold only. Each feature is accompanied by a missingness indicator, giving a 32-dimensional clinical input.

## Variants

| ID | Privileged Teacher | MRI-only Student | Question |
|---|---|---|---|
| P0 | Existing G3 Teacher | Locked B1 | Existing baseline |
| P2 | MRI + clean clinical gated fusion | Locked B1 | Can clinical privilege transfer to MRI-only inference? |
| P3 | MRI + surface + ribbon hard correspondence + DKT graph + clean clinical gated fusion | Locked B1 | Does the complete geometry-privileged Teacher add value beyond P2? |

P3 uses the legacy volume classification path. Ribbon tokens remain supervision-only anchors and never replace the legacy MRI context or enter the Student classifier.

## Required comparisons

- P2 − P0: clinical privileged knowledge over the current B1 baseline.
- P3 − P0: joint privileged knowledge over the current B1 baseline.
- P3 − P2: increment from the complete geometry-privileged Teacher design.

P3 − P2 does **not** isolate the hard cross-geometry loss. It jointly changes the surface/ribbon/graph privileged branch. A later P4, if authorized, is required before attributing an effect specifically to cross-geometry alignment.

## Metrics and decision rule

Report fold values and mean ± sample SD for MCC, macro one-vs-rest AUC, accuracy, Macro-F1, macro recall, macro specificity, and MCI recall. For P2−P0, P3−P0, and P3−P2, also report pooled paired-subject bootstrap intervals and exact McNemar tests.

P3 is considered a candidate improvement only if:

1. mean MCC and AUC improve over P0;
2. the direction is reasonably consistent across folds;
3. Macro-F1 and MCI recall do not deteriorate in mean;
4. paired uncertainty supports a real gain rather than a single-fold peak.

Only after this decision should P4/P5 or cognitive auxiliary-target experiments be considered. No held-out test result is used for this decision.

## Reproducibility artifacts

- `configs/runtime_privileged_p2_clinical_teacher.yaml`
- `configs/runtime_privileged_p2_b1_student.yaml`
- `configs/runtime_privileged_p3_joint_teacher.yaml`
- `configs/runtime_privileged_p3_b1_student.yaml`
- `audit_privileged_kd_protocol.py`
- `summarize_privileged_kd.py`
- `tests/test_privileged_kd_protocol.py`
- `tests/test_privileged_kd_summary.py`
