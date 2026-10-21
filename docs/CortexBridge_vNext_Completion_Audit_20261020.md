# CortexBridge vNext 阶段完成性审计

日期：2026-10-20  
范围：clinical leakage audit → Graph Teacher → random-topology control → strict MRI-only Graph Student → component attribution  
数据边界：development-only；held-out test 未读取、未加载、未评估

## 结论

本阶段目标已完成。开发集最终锁定 **S4 Student**：

```text
Teacher:
legacy MRI ROI classification path
+ auxiliary hard Surface↔Ribbon alignment (lambda_cg=0.01)
+ DiseaseAwareROIGraph(true DKT boundary strength, bilateral homology)

Student:
MRI-only topology-seeded Graph Student
+ CE
+ logit KD (0.5)
+ Teacher graph-enhanced ROI-state KD (1.0)
```

graph-relation KD 与 prototype-posterior KD 已经完成消融，但不进入锁定配置。锁定入口为
`configs/runtime_vnext_graph_student_locked_s4.yaml`。

## 逐项验收

| 要求 | 权威证据 | 审计结果 |
|---|---|---|
| clinical label-proximity / leakage audit | `outputs/vnext_clinical_leakage_audit_20261018/CLINICAL_LEAKAGE_AUDIT.md`、`audit_metrics.json` | 完成；`held_out_test_accessed=false`；主分析关闭 cognition/clinical |
| DiseaseAwareROIGraph 工程实现 | `models/disease_aware_graph.py`、Teacher/Student 接线与单元测试 | 完成；soft boundary/homology bias、remote attention、residual gate 已实现 |
| G0–G3 三折 development 比较 | `outputs/vnext_graph_stageA_20261019/GRAPH_STAGE_A_REPORT.md`、`graph_stage_a_summary.json` | 完成；G3 mean MCC `0.29236 ± 0.01713`；Graph gate 通过 |
| 真实拓扑随机化控制 | `GRAPH_RANDOMIZATION_REPORT.md`、`graph_randomization_summary.json` | 完成；真实 topology 的结构诊断更合理，但分类未稳定优于随机 topology |
| Graph gate 后才进入 Student | Graph summary 中 `student_stage_authorized=true`；Student 输出时间与报告链 | 满足 |
| strict MRI-only Student | `models/student.py`、`audit_mri_only_graph_student.py` | 完成；forward 必需输入仅 `image` |
| matched S0/S6 gate | `outputs/vnext_graph_student_20261019/GRAPH_STUDENT_REPORT.md` | 完成；预注册 development gate 通过 |
| S3–S5 progressive component ablation | `GRAPH_STUDENT_COMPONENT_ABLATION_REPORT.md`、`graph_student_component_summary.json` | 完成；S4 mean MCC 最高并被选定 |
| selected Student MRI-only audit | `s4_mri_only_audit_fold0.json` 至 `fold2.json` | 3/3 通过；无 privileged state；路径/clinical 扰动 logit delta 均为 `0.0` |
| held-out test 隔离 | 所有 summary flag、15 个正式 Student fold 日志 | 全部 false；15/15 日志包含 `test=NOT_LOADED` |
| 回归测试 | `PYTHONPATH=. pytest -q` | `62 passed` |
| 完整方法与投稿路线图 | `CortexBridge_vNext_Journal_Method_Status_and_Experiment_Roadmap.md` | 已更新到 S4 锁定决策与证据边界 |

## Student 组件结果

| ID | 累积目标 | MCC | 相对前一变体 mean ΔMCC | 正向 folds |
|---|---|---:|---:|---:|
| S0 | CE only | `0.27704 ± 0.03173` | — | — |
| S3 | + logit KD | `0.30505 ± 0.01183` | `+0.02800` | 3/3 |
| **S4** | **+ ROI graph-state KD** | **`0.31227 ± 0.04766`** | `+0.00723` | 1/3 |
| S5 | + graph relation KD | `0.29117 ± 0.06482` | `−0.02110` | 1/3 |
| S6 | + prototype posterior KD | `0.29141 ± 0.04274` | `+0.00024` | 1/3 |

S4 相对 S0 的 pooled ΔMCC 为 `+0.05707`，paired bootstrap 95% CI
`[+0.01133,+0.10529]`，McNemar exact `p=0.005468`。S4−S3 的独立增量为
`+0.00928`，95% CI `[-0.02214,+0.03637]`，因此 S4 可以作为 cumulative development
candidate，但 ROI-state KD 的独立稳定作用尚未确证。

## 论文允许与不允许的结论

允许：

- hard cross-geometry alignment 学到显著 ROI identity correspondence；
- DiseaseAwareROIGraph 提供真实 DKT 结构先验，但其 anatomy-specific 分类收益仍不确定；
- matched 三折 development evidence 支持选择 S4 进入完整 CV；
- S4 inference 是严格 MRI-only。

不允许：

- ribbon pooling 已替代 legacy pooling并提高分类；
- correspondence 主要是 subject-specific；
- 真实 topology 已稳定优于随机 topology；
- ROI-state KD 已被单独稳定证明；
- 当前结果已经证明外部泛化或最终 held-out test 优势。

## 本阶段之外的下一步

以下属于投稿前后续实验，不是本阶段已经完成的证据：

1. 使用 locked S4 完成预先固定的完整 CV；
2. calibration、confusion matrix、subgroup 与效率分析；
3. OASIS3/Huashan external validation 与 domain-shift 分析；
4. 完成所有锁定后仅一次运行 held-out test；
5. topology-soft、cognition 与 PET 仍是可选扩展，不得先于上述主验证。

## 2026-10-20 追加完成：no-MMSE 临床辅助分支

已在完全相同的 development folds 0/1/2 上完成 C1–C4 gated clinical ablation。所有正式运行均保持 Teacher clinical-off、held-out test 未加载，并从保存的逐例概率统一重算 MCC/ACC/SEN/SPE/AUC/macro-F1/Recall。

审计结论：

- S4-MRI 仍是严格 MRI-only 核心，不被临床分支替代；
- 选择 C2（background + GDS/NPI-Q）作为 `CortexBridge S4-C`；其 mean MCC 为 `0.44281 ± 0.02970`；
- C2 相对 S4-MRI pooled ΔMCC `+0.13140`，95% CI `[+0.08592,+0.17725]`；
- C2 相对 exact-ID clinical-only pooled ΔMCC `+0.10350`，95% CI `[+0.06346,+0.14313]`；
- C3/C4 的高分主要由 diagnosis-proximal cognition/FAQ 变量解释，不进入提出方法；
- 所有 feature lists 均无 MMSE；锁定 C2 还排除 memory、fluency 与 FAQ；
- 12/12 正式 fold artifacts 完整，`held_out_test_evaluated=false`；测试 `64 passed`。

新增锁定配置：`configs/runtime_vnext_graph_student_locked_s4c_safe_neuro.yaml`。  
完整报告：`outputs/vnext_graph_student_clinical_20261020/S4_CLINICAL_ASSISTED_ABLATION_REPORT.md`。

这项追加实验不改变最终 test 规则：S4-MRI 与 S4-C 必须先各自完成冻结的完整 CV/校准/亚组评估，之后才允许一次性 held-out test；二者属于不同推理输入设置，不能混排为同输入公平比较。
