# CortexBridge 顺序实验执行状态

更新日期：2026-10-20

## 当前锁定的方法主线

```text
MRI deep feature -> legacy MaskedVolumeTokenPooling -> V_context
SurfaceEncoder -> SurfaceROIPooling -> S_surface
MRI high-resolution feature -> RibbonSampler -> R_anchor

S_surface <-> R_anchor: hard same-ROI L_CG, lambda_cg=0.01
S_surface + V_context -> ROI-CLIP -> Z_teacher_base
Z_teacher_base + true DKT soft priors -> DiseaseAwareROIGraph -> Z_teacher_graph
Z_teacher_graph -> ROI classifier / prototype / relation
MRI global feature -> global classifier

MRI-only Student:
MRI -> SFCN encoder -> topology-seeded ROI queries -> Z_student_base
Z_student_base + same fixed DKT soft priors -> DiseaseAwareROIGraph -> Z_student_graph
training only: CE + 0.5 logit KD + 1.0 ROI graph-state KD
inference: image only; no surface/ribbon/clinical/PET/cognition
```

Ribbon 仍只是 cross-geometry anchor，不直接进入 classifier，也不替代 legacy volume pooling。

## 已完成：clinical label-proximity / leakage audit

审计只读取五折 `train.csv` 与 `valid.csv`，没有读取 test CSV、MRI、surface 或 checkpoint。

| Clinical-only arm | MCC | AUC | Balanced ACC |
|---|---:|---:|---:|
| Safe background（age/sex/education/APOE） | 0.17546 ± 0.04337 | 0.64304 ± 0.01929 | 0.47535 ± 0.03312 |
| High-risk cognition（memory/fluency/FAQ） | 0.67770 ± 0.01300 | 0.93526 ± 0.00363 | 0.80737 ± 0.01033 |
| Current clinical-29 | 0.69493 ± 0.02733 | 0.94046 ± 0.00523 | 0.81691 ± 0.02505 |
| Current clinical-29 with shuffled training labels | 0.02741 ± 0.07538 | 0.49938 ± 0.04358 | 0.34950 ± 0.05117 |

`current_clinical_29 − safe_background` 的 subject-bootstrap MCC 差为 `+0.52084`，95% CI `[0.48386, 0.55921]`。因此 FAQ、记忆和语义流畅性变量在来源诊断规则澄清前不能进入主方法比较。

决定：Graph Stage A 主分析统一使用 `clinical-off`。旧 Phase 2.1 P2-B full-clinical 结果保留为历史结果，但不能作为新的 Graph 主基线；必须重新运行 matched clinical-off G0。

审计报告：`outputs/vnext_clinical_leakage_audit_20261018/CLINICAL_LEAKAGE_AUDIT.md`。

## 已完成：DiseaseAwareROIGraph 工程实现

实现边界：

- 输入为 `Z_teacher_base`，不是 raw ribbon anchor；
- 一层、四头 full ROI self-attention；
- `A_boundary` 与 `A_hom` 仅作为 additive soft bias；
- remote ROI connection 不被 hard mask；
- residual gate 初始值为 `0.1`；
- classifier、prototype 与 teacher relation 使用 `Z_teacher_graph`；
- global MRI classifier 保持原路径；
- graph-off 不创建参数，并保持 same-seed 公共参数与 legacy forward 一致；
- G2-R 支持 deterministic topology permutation，但所有诊断仍相对真实 DKT topology 计算。

记录的 Graph diagnostics：

- anatomical attention mass；
- homology mass；
- identity mass；
- remote mass；
- normalized attention entropy；
- KL-to-uniform；
- attention standard deviation；
- true boundary-strength correlation；
- residual gate；
- graph output delta L2。

测试状态：Graph 实现时全量 `50 passed`；完成 Student 组件实验后的当前全量测试为 `61 passed`。

真实数据 CPU smoke（job `4218098`）已完成，exit code `0`：

- fold-0 G3，batch size 1，仅一个 train/validation batch；
- 日志确认 `test=NOT_LOADED (development-only)`；
- 完成 forward、backward、checkpoint reload 与 validation prediction 保存；
- 初始 graph remote mass `0.8451`、KL-to-uniform `0.0173`、boundary correlation `0.3179`、residual gate `0.1000`、output delta L2 `0.3319`；
- 该单样本分类指标不用于方法选择。

## 已完成：严格 development-only 协议修正

当 `runtime.evaluate_test=false` 时：

- `build_dataloaders` 不再实例化 test dataset；
- test CSV 不会被读取；
- 日志显式写入 `test=NOT_LOADED (development-only)`；
- 每个最佳 checkpoint 额外保存 validation-only paired prediction artifact；
- 汇总器拒绝缺 fold 或缺 prediction 的不完整实验。

## 已完成：Graph Stage A

| ID | 配置 | Clinical | 三折 MCC |
|---|---|---|---:|
| G0 | hard CG，无 graph | off | `0.27153 ± 0.04199` |
| G1 | graph，无 topology bias | off | `0.28308 ± 0.04855` |
| G2 | graph + boundary-strength bias | off | `0.27526 ± 0.04186` |
| G3 | G2 + bilateral homology bias | off | `0.29236 ± 0.01713` |

G3−G0 的 fold ΔMCC 为 `[-0.02057, +0.01440, +0.06865]`，mean `+0.02083`，2/3 folds 改善；paired subject bootstrap MCC difference 为 `+0.02021`，95% CI `[-0.02188, +0.06107]`。因此通过进入 Student 的 development gate，但不能声称 graph 显著提高分类。

报告：`outputs/vnext_graph_stageA_20261019/GRAPH_STAGE_A_REPORT.md`。

## 已完成：G3-R 随机拓扑控制

真实 G3 相对 G3-R 的 fold ΔMCC 为 `[-0.00825, -0.01003, +0.06002]`，mean `+0.01391`，仅 1/3 folds 改善；paired bootstrap MCC difference 为 `+0.01216`，95% CI `[-0.02561, +0.04900]`。

真实 DKT topology 的 boundary correlation、anatomical attention mass 和 homology mass 更符合解剖结构，但分类增益没有稳定优于随机 topology。决定：保留 G3 作为可解释结构先验并进入 Student；论文不得把 anatomy-specific classification gain 写成已证实结论。

报告：`outputs/vnext_graph_stageA_20261019/GRAPH_RANDOMIZATION_REPORT.md`。

## 已完成：纠正后的 S0/S6 MRI-only Graph Student

已经实现：

- fixed DKT Laplacian PE + trainable ROI residual；
- Student MRI ROI extraction 与 DiseaseAwareROIGraph；
- Teacher `z_teacher_graph` → Student `z_student_graph` ROI-state KD；
- logit、graph relation 与 prototype posterior KD；
- `student(image)` 严格 MRI-only forward；
- checkpoint 不包含 surface/ribbon/clinical/PET/cognition 参数；
- subject-specific privileged/cache 路径缺失时 logits 不变。

| ID | 模型 | 状态 |
|---|---|---|
| S0 | 相同 topology-seeded Graph Student，CE only | 三折完成；MCC `0.27704 ± 0.03173` |
| S6 | S0 + full privileged graph-state KD | 三折完成；MCC `0.29141 ± 0.04274` |

两者都从随机初始化开始，`warmup_epochs=0`，所有 Student 参数从 epoch 1 使用 `1e-4` 学习率。早期 `warmup=5 / lr=2e-6` 输出已标记无效，不用于结论。

S6−S0 fold ΔMCC 为 `[+0.03117, -0.05738, +0.06930]`，mean `+0.01436`，2/3 folds 改善；AUC/ACC/macro-F1/balanced accuracy mean delta 均为正。pooled MCC delta `+0.03028`，bootstrap 95% CI `[-0.02058, +0.08179]`；pooled AUC CI 为 `[+0.00559, +0.05246]`；McNemar exact `p=0.022737`。

预注册 development gate 已通过并授权组件消融，但 MCC CI 跨 0，因此不能写成稳定显著的最终收益。S6 三个正式 checkpoint 均通过 MRI-only audit：无 subject-specific privileged state，改变 clinical argument 或使 subject-specific cache 路径失效时最大 logit delta 都为 `0.0`，held-out test 未加载。

报告：`outputs/vnext_graph_student_20261019/GRAPH_STUDENT_REPORT.md`。

## 已完成：S3–S5 progressive component ablation

| ID | 累积 KD objective | MCC | 相对前一变体 mean ΔMCC | 正向 folds |
|---|---|---:|---:|---:|
| S0 | CE only | `0.27704 ± 0.03173` | — | — |
| S3 | + logit KD | `0.30505 ± 0.01183` | `+0.02800` | 3/3 |
| **S4** | **+ ROI graph-state KD** | **`0.31227 ± 0.04766`** | `+0.00723` | 1/3 |
| S5 | + graph relation KD | `0.29117 ± 0.06482` | `−0.02110` | 1/3 |
| S6 | + prototype posterior KD | `0.29141 ± 0.04274` | `+0.00024` | 1/3 |

S3–S5 与 S0/S6 使用相同 subjects、architecture、seed、初始化方式、optimizer、`warmup=0` 与 `1e-4` 学习率。三个 GPU dry-run 和九个正式 component folds 均完成，全部记录 `test=NOT_LOADED`。

按预先限定的 development-only 主指标，S4 的三折 mean MCC 最高，因此锁定 S4 进入完整 CV。S4 相对 S0 的 pooled ΔMCC 为 `+0.05707`，bootstrap 95% CI `[+0.01133,+0.10529]`，McNemar `p=0.005468`。但 S4−S3 的 pooled ΔMCC 仅 `+0.00928`，95% CI `[-0.02214,+0.03637]`，且只有 1/3 folds 改善；因此 ROI graph-state KD 的独立增量尚未稳定确证。

开发集决策：

- 保留 CE + logit KD + ROI graph-state KD（S4）；
- 删除 graph-relation KD 与 prototype-posterior KD；
- 不把 S4 的选择解释为每个蒸馏项均有独立因果收益；
- S4 三个 checkpoint 均通过 MRI-only audit，且 held-out test 始终未加载。

组件报告：`outputs/vnext_graph_student_20261019/GRAPH_STUDENT_COMPONENT_ABLATION_REPORT.md`。

锁定配置：`configs/runtime_vnext_graph_student_locked_s4.yaml`。自动测试确认其 KD 权重、MRI-only contract、legacy pooling、topology query、graph 配置、训练 schedule 与正式 S4 实验一致。

## 尚未执行

- 完整五折锁定评估；
- 外部验证、calibration、subgroup 与效率实验；
- topology-soft correspondence（仅 P1 可选）；
- cognition/PET privileged extension（当前不进入主线）。

以上未完成模块均不得写成已验证贡献。当前实验阶段已完成，下一步只能使用锁定的 S4 做完整 CV；在最终冻结前仍不得读取 held-out test。

## 已完成：正式消融分类指标统一复算

已从保存的 development validation 概率逐例复算 Graph Teacher G0/G1/G2/G3/G3-R 与 MRI-only Student S0/S3/S4/S5/S6 的 MCC、ACC、SEN、SPE、macro one-vs-rest AUC、macro-F1 和 macro Recall；未重新训练，held-out test 未读取。

锁定 S4 的三折结果为：MCC `0.31227 ± 0.04766`、ACC `0.53872 ± 0.03044`、SEN/Recall `0.55227 ± 0.03053`、SPE `0.75617 ± 0.01551`、AUC `0.73894 ± 0.02805`、macro-F1 `0.51942 ± 0.04210`。三分类下 SEN 定义为逐类 sensitivity 的宏平均，所以它与 macro Recall/balanced accuracy 数值相同，不作为重复独立证据。

完整结果、逐类 NC/MCI/DE 指标及实验配置清单见 `outputs/vnext_classification_metric_recalculation_20261020/ABLATION_CLASSIFICATION_METRICS_REPORT.md`。复算值与训练阶段保存值的最大绝对差为 `0.000e+00`。

## 已完成：S4 临床辅助扩展（不含 MMSE）

在不改变 S4 Teacher、MRI/graph 路径、KD 权重、优化器和 fold 的条件下，完成四组 gated clinical Student 三折 development 实验：

| ID | 推理输入 | MCC | AUC | 解释 |
|---|---|---:|---:|---|
| C0 | S4 MRI-only | `0.31227 ± 0.04766` | `0.73894 ± 0.02805` | MRI-only 核心 |
| C1 | MRI + background | `0.31072 ± 0.06202` | `0.73241 ± 0.03756` | 未稳定优于 C0 |
| **C2** | **MRI + background + GDS/NPI-Q** | **`0.44281 ± 0.02970`** | **`0.80494 ± 0.02215`** | **选定 S4-C** |
| C3 | MRI + cognition/FAQ | `0.66540 ± 0.01653` | `0.92079 ± 0.00632` | diagnosis-proximal upper bound |
| C4 | MRI + full29，no MMSE | `0.69241 ± 0.02588` | `0.92944 ± 0.00770` | diagnosis-proximal upper bound |

C2 相对 S4-MRI 的 pooled ΔMCC 为 `+0.13140`，subject bootstrap 95% CI `[+0.08592,+0.17725]`；相对 exact-ID safe-neuro clinical-only 的 pooled ΔMCC 为 `+0.10350`，95% CI `[+0.06346,+0.14313]`。因此 C2 同时优于 MRI-only 与 clinical-only，作为需要临床输入的扩展被锁定。

C3 相对 cognition/FAQ clinical-only 没有 MCC 优势且 AUC 更低；C4 相对 full29 clinical-only 的 MCC CI 跨 0 且 AUC 更低。因此 C3/C4 只作为临床标签邻近性上界，不进入主方法。C2 不含 MMSE、记忆量表、言语流畅性或 FAQ，但需要 demographics/APOE、GDS 和 NPI-Q 在推理时可用。

主方法现在必须分开命名：

- `CortexBridge S4-MRI`：严格 MRI-only 主线，配置 `configs/runtime_vnext_graph_student_locked_s4.yaml`；
- `CortexBridge S4-C`：临床辅助扩展，配置 `configs/runtime_vnext_graph_student_locked_s4c_safe_neuro.yaml`。

完整报告：`outputs/vnext_graph_student_clinical_20261020/S4_CLINICAL_ASSISTED_ABLATION_REPORT.md`。四组 12 个正式 folds 全部记录 `held_out_test_evaluated=false`；最终回归测试 `64 passed`。

## 已完成：Student-only MedicalNet/Med3D ResNet10 M0–M4 三折消融

在保持锁定 G3 Teacher、DKT topology graph/query、下游分类器、fold、seed 和 development-only 协议不变的条件下，实现 MedicalNet-compatible 3D ResNet10 Student adapter，并完成 M0–M4 三折实验。M1–M4 推理均严格 MRI-only，不使用 MMSE、clinical、surface、ribbon 或 PET。

| ID | Student 配置 | MCC | AUC | 结论 |
|---|---|---:|---:|---|
| **M0** | **锁定 SFCN S4 + ROI-state KD** | **`0.31227 ± 0.04766`** | **`0.73894 ± 0.02805`** | **继续作为主线** |
| M1 | ResNet10 scratch + CE | `0.26318 ± 0.04678` | `0.69495 ± 0.01693` | 最佳 Med3D-family 变体，但低于 M0 |
| M2 | Med3D pretrained + CE | `0.14511 ± 0.04697` | `0.62395 ± 0.01841` | 预训练在锁定 recipe 下明显有害 |
| M3 | M2 + logit KD | `0.22263 ± 0.06020` | `0.67722 ± 0.03190` | KD 部分恢复 M2 |
| M4 | M3 + ROI-state KD | `0.21078 ± 0.02745` | `0.66854 ± 0.02095` | 相对 M3 无稳定额外收益 |

关键配对结果：

- M1−M0 pooled ΔMCC `−0.04708`，95% CI `[−0.08691,−0.00931]`；
- M2−M1 pooled ΔMCC `−0.12174`，95% CI `[−0.16760,−0.07330]`；
- M3−M2 pooled ΔMCC `+0.07855`，95% CI `[+0.03222,+0.12591]`；
- M4−M3 pooled ΔMCC `−0.01824`，95% CI `[−0.05894,+0.02375]`；
- M2−M4 pooled ΔMCC `−0.06032`，95% CI `[−0.11219,−0.01078]`。

因此，预先约定的“若 M2 显著优于 M4，再考虑用 Med3D 重训 cross-geometry teacher”条件没有满足。当前决定是不重训 Med3D Teacher、不替换 SFCN S4 主线，并把该实验作为 backbone/pretraining 负结果与 KD 恢复性消融报告。最佳 Med3D 变体 M1 的三个 checkpoint 均通过 MRI-only audit，15 个正式 fold 均记录 `held_out_test_evaluated=false`。

实现与产物：

- adapter：`models/volume_encoder.py`、`models/student.py`、`train_student.py`；
- 配置：`configs/runtime_med3d_m1_resnet10_scratch_ce.yaml` 至 `configs/runtime_med3d_m4_resnet10_pretrained_roi_kd.yaml`；
- 汇总器：`summarize_med3d_student_ablation.py`；
- 完整报告：`outputs/vnext_med3d_resnet10_20261020/MED3D_RESNET10_STUDENT_ABLATION_REPORT.md`；
- 机器可读结果：`med3d_student_ablation_summary.json` 与 `med3d_student_ablation_metrics.csv`。

该阶段不改变已锁定的论文主方法：`CortexBridge S4-MRI` 仍为 MRI-only 核心，`CortexBridge S4-C` 仍为不含 MMSE 的临床辅助扩展。

## 已完成：ROI bottleneck D0–D4 / B0–B4 消融（修订 MRI-only 主线）

针对“deep feature 经 ROI token 再进 graph 会削弱分类”的假设，完成两组互补实验：D0–D4 在已有 S3/S4 checkpoint 内切换推理分支，不重新训练；B0–B4 使用相同 G3 Teacher、fold、seed、优化器、MCC 早停标准做三折配对重训。B2–B4 的分类器使用未归一化 raw ROI token，归一化副本仅供 ROI-state KD，从而排除单位范数分类瓶颈这一混杂因素。

| ID | 配置 | MCC | AUC | ACC | 结论 |
|---|---|---:|---:|---:|---|
| B0 | global CE | `0.24877 ± 0.03390` | `0.70297 ± 0.02411` | `0.49787 ± 0.01048` | 无蒸馏基线 |
| **B1** | **global CE + logit KD** | **`0.32870 ± 0.03922`** | **`0.74623 ± 0.02770`** | **`0.56490 ± 0.04083`** | **修订后的 MRI-only Student 主线** |
| B2 | B1 + raw ROI classifier | `0.25861 ± 0.04353` | `0.72076 ± 0.02814` | `0.52378 ± 0.05136` | ROI 分类残差稳定有害 |
| B3 | B2 + ROI graph | `0.25133 ± 0.03680` | `0.70350 ± 0.04079` | `0.49369 ± 0.05418` | graph 不是主要损失来源 |
| B4 | B3 + ROI-state KD | `0.31729 ± 0.03137` | `0.74272 ± 0.02342` | `0.54726 ± 0.02830` | 恢复 B3，但未超过 B1 |

关键因果对照：

- B1−B0：三折均提升，mean ΔMCC `+0.07993`，pooled 95% CI `[+0.03633,+0.12456]`；
- B2−B1：三折均下降，mean ΔMCC `−0.07009`，pooled 95% CI `[−0.10959,−0.02818]`；
- B3−B2：mean ΔMCC `−0.00728`，95% CI `[−0.04683,+0.03462]`，graph 增量不确定；
- B4−B3：mean ΔMCC `+0.06596`，95% CI `[+0.01712,+0.10807]`，ROI-state KD 能恢复 ROI 表征；
- B4−B1：mean ΔMCC `−0.01141`，95% CI `[−0.04713,+0.02192]`，完整 ROI 路径没有证据优于更简单的 B1。

D 审计提供一致证据：S4 global-only 的 mean MCC 为 `0.32201`，高于当前 fused S4 的 `0.31227`；ROI-only MCC 为 `0`；绕过 graph 几乎不改变 fused 结果。graph residual 仅约 token norm 的 `5.9%`，因此主要瓶颈是弱 ROI logits 被融合回分类器，而不是 graph 本身或 ROI-state 表征学习对全局 encoder 的损伤。

据此，MRI-only Student 的修订主线为：**global MRI classifier + locked G3 Teacher logit KD；关闭 Student ROI classification residual 与 Student ROI graph。** Cross-geometry/ROI graph 仍保留在 privileged Teacher、表征分析和消融叙事中，但不再作为 Student 最终分类路径。旧 S4-C 临床扩展建立在 fused S4 上；如将临床扩展纳入最终论文主表，需在冻结 test 前补做 `B1 + C2 safe-neuro` 的严格配对验证，不能直接把旧 C2 增益转移到 B1。

完整报告：`outputs/vnext_roi_bottleneck_ablation_20261021/ROI_BOTTLENECK_ABLATION_REPORT.md`。15 个 B0–B4 fold 均为 development-only，held-out test 始终未加载；相关回归测试 `16 passed`。
