# CortexBridge Unified MRI–Clinical Development Comparison

- Task: ADNI NC/MCI/DE three-class classification.
- Folds: subject-level development folds 0–2.
- Clinical set: 29 variables, MMSE excluded.
- Held-out test accessed: `false`.
- Student ROI/graph classification residual: disabled.

## Decision

- **Primary performance configuration: E3.** It has the highest neural-model development MCC (0.69535) and is not improved by the full E4 privileged Teacher (E4−E3 ΔMCC -0.00353, 95% CI [-0.01751, +0.01076]).
- **Mechanism-oriented CortexBridge configuration: E4.** E4 reaches MCC 0.69226 and AUC 0.93199, but its increment over direct MRI+clinical CE is small and uncertain (E4−E2 ΔMCC +0.00707, 95% CI [-0.01170, +0.02504]).
- **No classification-superiority claim for cross-geometry.** Isolating cross-geometry gives E4−E6 ΔMCC +0.00484, 95% CI [-0.01303, +0.02177]. Cross-geometry remains supported as correspondence/representation supervision, not as a proven classifier improvement.
- **Correct clinical pairing is essential.** E4−E5 ΔMCC is +0.40110, 95% CI [+0.34734, +0.45269], with McNemar p=3.51e-48.
- These are development-only conclusions. No held-out test should be opened until the method, clinical-variable policy, and missing-data handling are frozen.

## E0–E6 results

| Arm | Description | MCC | AUC | ACC | Macro-F1 | Recall | SPE | MCI recall |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| E0 | Clinical-only logistic regression | 0.67902 ± 0.01402 | 0.93996 ± 0.00476 | 0.79392 ± 0.01000 | 0.78749 ± 0.01284 | 0.80585 ± 0.02439 | 0.89328 ± 0.00533 | 0.72175 ± 0.00587 |
| E1 | MRI-only CE | 0.24334 ± 0.05451 | 0.69288 ± 0.02913 | 0.51568 ± 0.03319 | 0.46971 ± 0.08537 | 0.51059 ± 0.06985 | 0.74072 ± 0.01932 | 0.41556 ± 0.16259 |
| E2 | MRI+clinical CE | 0.68511 ± 0.02143 | 0.92948 ± 0.00347 | 0.79916 ± 0.01933 | 0.79153 ± 0.02320 | 0.80258 ± 0.01500 | 0.89460 ± 0.00568 | 0.73403 ± 0.06252 |
| E3 | MRI+clinical ordinary Teacher logit KD | 0.69535 ± 0.01743 | 0.93203 ± 0.00214 | 0.80708 ± 0.01159 | 0.79758 ± 0.01891 | 0.80618 ± 0.01988 | 0.89807 ± 0.00474 | 0.75383 ± 0.02793 |
| E4 | Full surface-privileged CortexBridge | 0.69226 ± 0.01562 | 0.93199 ± 0.00224 | 0.80555 ± 0.00844 | 0.79401 ± 0.01618 | 0.80015 ± 0.01994 | 0.89661 ± 0.00472 | 0.75912 ± 0.03373 |
| E5 | E4 Student clinical subject-shuffle | 0.29103 ± 0.02717 | 0.72911 ± 0.01484 | 0.54857 ± 0.01097 | 0.55437 ± 0.00492 | 0.57170 ± 0.02131 | 0.75665 ± 0.01079 | 0.52642 ± 0.01712 |
| E6 | E4 with Teacher cross-geometry disabled | 0.68721 ± 0.02206 | 0.93082 ± 0.00439 | 0.80056 ± 0.01500 | 0.79301 ± 0.01958 | 0.80493 ± 0.01259 | 0.89560 ± 0.00618 | 0.73974 ± 0.03929 |

## Predeclared paired contrasts

| Contrast | ΔMCC [95% CI] | ΔAUC [95% CI] | ΔACC [95% CI] | McNemar p |
|---|---:|---:|---:|---:|
| E4 − E2 | +0.00707 [-0.01170, +0.02504] | +0.00134 [-0.00245, +0.00517] | +0.00653 [-0.00581, +0.01814] | 0.349182 |
| E4 − E3 | -0.00353 [-0.01751, +0.01076] | -0.00202 [-0.00424, +0.00009] | -0.00145 [-0.01016, +0.00726] | 0.877614 |
| E4 − E6 | +0.00484 [-0.01303, +0.02177] | +0.00037 [-0.00349, +0.00418] | +0.00508 [-0.00653, +0.01597] | 0.463818 |
| E4 − E5 | +0.40110 [+0.34734, +0.45269] | +0.20480 [+0.18470, +0.22615] | +0.25689 [+0.22279, +0.28955] | 3.51021e-48 |
| E4 − E0 | +0.01285 [-0.01121, +0.03555] | -0.01089 [-0.01600, -0.00585] | +0.01161 [-0.00363, +0.02612] | 0.156066 |

Interpretation: E4−E2 is the total privileged-Teacher increment; E4−E3 is the increment over ordinary MRI–clinical KD; E4−E6 isolates hard cross-geometry training within the same surface/graph Teacher; E4−E5 tests correct Student clinical pairing; E4−E0 tests imaging plus privileged supervision beyond clinical scales alone.

## Validity and deployment constraints

- E2–E6 are MRI+clinical inference models, not MRI-only models. They must be compared against multimodal baselines and labeled accordingly.
- The 29-variable set excludes MMSE but includes memory and functional variables. Because some may overlap diagnostic criteria, the paper must disclose them explicitly and retain E0 as a shortcut/upper-bound audit.
- Missing values are represented with train-fold-fitted imputation plus missingness indicators. The separate E7/E8 audit shows substantial degradation under added missingness; see `missingness_audit/MISSINGNESS_AUDIT.md`.
- E4 does not justify claiming that surface/ribbon/graph privileged knowledge raises classification accuracy over ordinary clinical KD. Its defensible role is anatomical correspondence and mechanism-oriented regularization.
