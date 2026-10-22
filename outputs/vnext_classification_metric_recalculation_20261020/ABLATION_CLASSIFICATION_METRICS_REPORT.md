# CortexBridge 正式消融分类指标复算报告

日期：2026-10-20  
状态：**VERIFIED（由保存的 development validation 概率逐例复算）**  
边界：未重新训练；未读取、加载或评估 held-out test。

## 结论摘要

已对 Graph Teacher G0/G1/G2/G3/G3-R 与 MRI-only Student S0/S3/S4/S5/S6 的三折正式预测统一复算 MCC、ACC、SEN、SPE、AUC、macro-F1 和 macro Recall。S4 仍是 Student 消融中 mean MCC 最高的锁定方案。

本任务为三分类 NC/MCI/DE：`SEN` 定义为逐类 one-vs-rest sensitivity 的宏平均，因此 `SEN = macro Recall = balanced accuracy`。报告保留 SEN 与 Recall 两列以满足临床报告习惯，但它们不是独立指标。`SPE` 是逐类 one-vs-rest specificity 的宏平均。

## Graph Teacher

所有单元格均为三折 mean ± sample SD。

| ID | MCC | ACC | SEN | SPE | AUC | Macro-F1 | Macro Recall |
|---|---:|---:|---:|---:|---:|---:|---:|
| G0 | 0.27153 ± 0.04199 | 0.53228 ± 0.03521 | 0.54089 ± 0.02617 | 0.74809 ± 0.01279 | 0.71620 ± 0.03300 | 0.52452 ± 0.04474 | 0.54089 ± 0.02617 |
| G1 | 0.28308 ± 0.04855 | 0.54378 ± 0.02434 | 0.54839 ± 0.06680 | 0.75200 ± 0.01791 | 0.72546 ± 0.01855 | 0.52315 ± 0.06278 | 0.54839 ± 0.06680 |
| G2 | 0.27526 ± 0.04186 | 0.54389 ± 0.03617 | 0.52578 ± 0.04659 | 0.74804 ± 0.01280 | 0.72605 ± 0.03109 | 0.51794 ± 0.06952 | 0.52578 ± 0.04659 |
| G3 | 0.29236 ± 0.01713 | 0.54215 ± 0.00912 | 0.55718 ± 0.02183 | 0.75491 ± 0.00362 | 0.72346 ± 0.01842 | 0.53517 ± 0.02896 | 0.55718 ± 0.02183 |
| G3-R | 0.27845 ± 0.03031 | 0.52790 ± 0.03275 | 0.52268 ± 0.05641 | 0.74661 ± 0.01672 | 0.72837 ± 0.02981 | 0.49806 ± 0.07409 | 0.52268 ± 0.05641 |

## MRI-only Student

所有单元格均为三折 mean ± sample SD。

| ID | MCC | ACC | SEN | SPE | AUC | Macro-F1 | Macro Recall |
|---|---:|---:|---:|---:|---:|---:|---:|
| S0 | 0.27704 ± 0.03173 | 0.49560 ± 0.02572 | 0.53218 ± 0.05446 | 0.74132 ± 0.01294 | 0.71830 ± 0.01595 | 0.45080 ± 0.06271 | 0.53218 ± 0.05446 |
| S3 | 0.30505 ± 0.01183 | 0.52768 ± 0.01331 | 0.55300 ± 0.01117 | 0.75394 ± 0.00100 | 0.72422 ± 0.02664 | 0.50789 ± 0.01245 | 0.55300 ± 0.01117 |
| S4 | 0.31227 ± 0.04766 | 0.53872 ± 0.03044 | 0.55227 ± 0.03053 | 0.75617 ± 0.01551 | 0.73894 ± 0.02805 | 0.51942 ± 0.04210 | 0.55227 ± 0.03053 |
| S5 | 0.29117 ± 0.06482 | 0.54821 ± 0.03775 | 0.55976 ± 0.05848 | 0.75730 ± 0.01999 | 0.72839 ± 0.04598 | 0.54587 ± 0.04556 | 0.55976 ± 0.05848 |
| S6 | 0.29141 ± 0.04274 | 0.53331 ± 0.02600 | 0.55677 ± 0.01168 | 0.75325 ± 0.01191 | 0.72518 ± 0.01846 | 0.52528 ± 0.01171 | 0.55677 ± 0.01168 |

## 主模型 S4 的逐类表现

逐类指标采用 one-vs-rest 定义，均为三折 mean ± sample SD。Recall 与 Sensitivity 在逐类层面也是同一个量。

| Class | Support（三折总计） | SEN / Recall | SPE | AUC | F1 |
|---|---:|---:|---:|---:|---:|
| NC | 505 | 0.88377 ± 0.06206 | 0.53082 ± 0.06887 | 0.77256 ± 0.01078 | 0.65582 ± 0.00700 |
| MCI | 629 | 0.27406 ± 0.05755 | 0.81599 ± 0.00829 | 0.58725 ± 0.04267 | 0.36541 ± 0.06242 |
| DE | 244 | 0.49897 ± 0.10878 | 0.92170 ± 0.05981 | 0.85702 ± 0.03393 | 0.53702 ± 0.07047 |

## 模型选择解释

- Graph Teacher 中，G3 的 mean MCC 最高；G2 的 ACC 略高、G3-R 的 AUC 略高，因此不能说 G3 在所有指标上都占优。G3 是依据预设主指标 MCC 与结构解释性进入 Student。
- Student 中，S4 的 mean MCC（0.31227）与 AUC（0.73894）最高；S5 的 ACC、SEN、SPE 和 macro-F1 略高，但 MCC 更低且跨折波动更大。
- 实验在运行前以 MCC 为 selection metric；补报次要指标后不得事后改变选择准则。因此锁定 S4 的决定不变。完整五折时应同时报告全部指标、混淆矩阵和逐类结果，不应只报 MCC。

## 实验配置对应关系

| ID | 改动 | Runtime config | Graph/B/H/Shuffle | KD weights (logit/ROI/graph/proto) |
|---|---|---|---|---|
| G0 | hard CG, graph off | `configs/runtime_vnext_graph_g0_clinical_off.yaml` | False/False/False/False | 0.0/0.0/0.0/0.0 |
| G1 | graph, no topology bias | `configs/runtime_vnext_graph_g1_unbiased.yaml` | True/False/False/False | 0.0/0.0/0.0/0.0 |
| G2 | graph + true DKT boundary bias | `configs/runtime_vnext_graph_g2_boundary.yaml` | True/True/False/False | 0.0/0.0/0.0/0.0 |
| G3 | G2 + bilateral homology bias | `configs/runtime_vnext_graph_g3_boundary_homology.yaml` | True/True/True/False | 0.0/0.0/0.0/0.0 |
| G3-R | G3 with deterministically shuffled topology bias | `configs/runtime_vnext_graph_g3r_shuffled_boundary_homology.yaml` | True/True/True/True | 0.0/0.0/0.0/0.0 |
| S0 | CE only | `configs/runtime_vnext_graph_student_s0_scratch.yaml` | True/True/True/False | 0.0/0.0/0.0/0.0 |
| S3 | S0 + logit KD | `configs/runtime_vnext_graph_student_s3_logit_kd.yaml` | True/True/True/False | 0.5/0.0/0.0/0.0 |
| S4 | S3 + ROI graph-state KD (locked) | `configs/runtime_vnext_graph_student_s4_roi_kd.yaml` | True/True/True/False | 0.5/1.0/0.0/0.0 |
| S5 | S4 + graph-relation KD | `configs/runtime_vnext_graph_student_s5_graph_kd.yaml` | True/True/True/False | 0.5/1.0/0.2/0.0 |
| S6 | S5 + prototype-posterior KD | `configs/runtime_vnext_graph_student_s6_full_kd.yaml` | True/True/True/False | 0.5/1.0/0.2/0.2 |

共同设置：development folds 0/1/2；3 类 NC/MCI/DE；clinical off；legacy_mask volume pooling；hard Surface↔Ribbon cross-geometry alignment，`lambda_cg=0.01`；selection metric=MCC；seed=3407；`evaluate_test=false`。Graph Teacher 为 30 epochs、batch size 8、lr=1e-4。Student 为随机初始化 topology-seeded Graph Student，30 epochs、batch size 8、encoder/inherited lr=1e-4、warmup=0。完整扁平化配置栈见 `ablation_config_manifest.csv` 和各 run 的 `opt.txt`。

## 指标定义

- ACC：全部样本预测正确比例。
- SEN / Recall：先计算 NC、MCI、DE 各自 `TP/(TP+FN)`，再做不加权宏平均。
- SPE：先把每一类视为阳性、其余两类视为阴性，计算 `TN/(TN+FP)`，再做宏平均。
- AUC：三类 one-vs-rest ROC-AUC 的宏平均，使用保存的概率而非 hard label。
- F1：三类 F1 的宏平均；不是 micro-F1 或 weighted-F1。
- MCC：scikit-learn multiclass Matthews correlation coefficient。
- 跨折汇总：算术平均 ± sample SD（`ddof=1`），不是把三折样本合并后再报单一值。

## 逐类结果与可复现产物

逐 fold 指标、NC/MCI/DE 逐类指标及配置清单分别保存在：

- `classification_metrics_by_fold.csv`
- `classification_metrics_summary.csv`
- `classification_metrics_per_class.csv`
- `classification_metrics_per_class_summary.csv`
- `ablation_config_manifest.csv`
- `classification_metrics_recalculation.json`

## 完整性与一致性验证

- 10 个正式变体 × 3 folds = 30 份 validation prediction artifacts 均成功读取。
- 每折样本数为 454/457/467，三折合计 1,378；各变体 subject/session ID、标签及折划分一致。
- 所有概率有限、非负、逐行和为 1，保存的 prediction 与概率 argmax 一致。
- 所有 prediction/development metadata 的 held-out 标志均为 false。
- 与训练阶段保存的聚合指标逐项核对，最大绝对差为 `0.000e+00`。
- dry-run、早期无效 low-LR Student、缺逐例概率的历史 Phase 1/2.1 结果均未混入本复算表。

## Material Passport

- Input：30 份已保存的 development validation prediction artifacts；对应 development metrics 与 `opt.txt`。
- Transformation：固定公式逐例复算 → 折内指标 → 三折 mean ± sample SD；不拟合参数，不改变预测。
- Output：本报告、4 个指标 CSV、1 个配置 CSV、1 个机器可读 JSON。
- Exclusions：held-out test、dry-run、无效学习率协议、没有逐例概率的历史结果。
- Reproduction：`python recompute_ablation_classification_metrics.py`。

## Epistemic / fallacy scan（11/11）

1. 不把 development selection 写成 held-out 泛化。
2. 不把 mean 改善写成统计显著，除非另有配对区间支持。
3. 不把 SEN、Recall、balanced accuracy 重复计为三份证据。
4. 不把 macro 指标解释为任一具体疾病类别表现。
5. 不用 hard prediction 计算 AUC。
6. 不把 pooled 指标冒充 fold mean。
7. 不把 Graph Teacher 与 MRI-only Student 当成同一推理输入条件。
8. 不把 G3 优于 G3-R 写成已稳定证明的 anatomy-specific classification gain。
9. 不把 S4−S3 的小增量写成 ROI-state KD 已独立确证。
10. 不纳入 dry-run 或已判无效的 Student 协议。
11. 不从本复算推断外部验证、校准或亚组公平性。

## 历史 Phase 1/2.1 的边界

Phase 1/1.1/2.1 当时只落盘了聚合 development metrics 与 checkpoint，没有逐受试者 validation probabilities。因此本次无训练复算不能为这些历史变体新增可靠的 SPE/逐类 SEN；如必须补齐，需要单独执行 checkpoint validation inference（仍可保持 development-only），那将属于一次新的推理任务，而不是本报告的纯离线复算。

