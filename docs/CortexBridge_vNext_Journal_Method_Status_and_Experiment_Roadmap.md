# CortexBridge vNext：已实现方法、待实现模块与期刊实验路线图

版本日期：2026-10-20  
适用代码库：`cortexweave_roiclip`  
目标期刊层级：TMI / Medical Image Analysis 方向  
文档性质：当前事实审计、最终方法收敛方案、投稿前实验清单

---

## 1. 执行摘要

当前工程已经完成可靠的 DKT ROI 数据链、legacy ROI-CLIP Teacher、真实 white–pial ribbon sampling、hard Surface↔Ribbon correspondence、DiseaseAwareROIGraph，以及严格 MRI-only Graph Student。Phase 2.1 的三折开发集结果证明 correspondence 明显高于随机且未造成稳定分类下降；Phase 2.1.5 证明该 correspondence 主要学习 atlas ROI identity，而不是强 subject-specific matching。clinical leakage audit 后，Graph 与 Student 主分析均关闭 cognition/clinical 输入。

Graph Stage A 的 G3 通过进入 Student 的 development gate，但 G3-R 控制未证明真实 topology 带来稳定的 anatomy-specific 分类收益。纠正后的 matched Student 三折中，full graph-state KD（S6）相对 scratch（S0）通过了预注册 Student gate，因此完成了 S3–S5 progressive component ablation。开发集最终选择 **S4（CE + logit KD + ROI graph-state KD）**：其 mean MCC 为 `0.31227 ± 0.04766`，在五个 matched 变体中最高；相对 S0 的 pooled ΔMCC 为 `+0.05707`，95% CI `[+0.01133,+0.10529]`。但 S4−S3 的独立增量仅 1/3 folds 为正且 CI 跨 0，因此只能说 S4 是 development-selected cumulative objective，不能说 ROI-state KD 已被独立稳定证实。S5 relation KD 与 S6 prototype KD 不进入锁定方案。

现阶段最稳妥的主线不是让 ribbon 替换 legacy volume pooling，也不是立即把 soft topology 注入 correspondence target，而是：

> **保留 legacy MRI ROI classification path；把 ribbon 固定为辅助 cortical anchor；在现有 fused ROI state 后加入真实 DKT topology 驱动的 disease-aware graph；只有 Teacher graph 经三折验证有效后，再把该 cortical graph state 蒸馏给严格 MRI-only Student。**

建议最终论文方法由三个核心模块组成：

1. **Auxiliary Cross-Geometry Cortical Anchoring**：Surface ROI 与 MRI ribbon anchor 的 hard same-ROI 对齐；
2. **Anatomy-Prior Disease-Aware Cortical Graph**：在 Teacher fused ROI tokens 上进行 subject-adaptive graph reasoning；
3. **MRI-only Privileged Cortical State Distillation**：Student 仅从 T1 MRI 学习 Teacher logits 与 graph-enhanced ROI state；relation 与 prototype target 已完成消融但不保留在锁定方案中。

PET 与 ROI-wise cognitive modulation 暂时不应作为主方法核心贡献。它们只能在 graph 与 MRI-only distillation 已经成立、并完成 cognitive-label leakage 审计后作为扩展模块加入。

---

## 2. 证据边界与状态标记

本文使用以下标记：

| 状态 | 含义 |
|---|---|
| **已实现、已验证** | 代码存在，至少完成单元测试、dry-run 或正式开发集实验 |
| **已实现、证据不足** | 代码存在，但尚不能支持稳定的方法学结论 |
| **仅完成离线资产/审计** | 数据或矩阵已生成，但尚未进入模型 forward/training |
| **尚未实现** | 当前生产代码没有该模块 |
| **历史结果** | 来自 vNext 前的结果，可提供背景，但不能替代新方法实验 |

数据使用边界：

- Phase 1 曾按预先设定 gate 查看过 fold-0 held-out test；该 test 结果已经消费，后续不得继续用其调参。
- Phase 1.1、Phase 2.1 和 Phase 2.1.5 均为 development-only，记录为 `held_out_test_evaluated/accessed: false`。
- 当前 vNext 只有三个 development folds 的主要证据，尚未完成最终五折、外部验证或一次性锁定测试评估。
- 所有后续 topology、graph、Student KD 超参数必须只用 training/validation folds 选择。

---

## 3. 当前已经实现的完整系统

### 3.1 数据、atlas 与空间对齐

状态：**已实现、已验证**。

当前默认使用 62 个 DKT cortical ROI，每侧 31 个。数据管线包含：

- ADNI-only split 与 subject-prefix 二次过滤；
- T1 MRI 中心裁剪/补零到 `(160, 192, 160)`；
- 根据裁剪 offset 更新 affine；
- 将 `aparc.DKTatlas+aseg.mgz`、`lh.ribbon.mgz`、`rh.ribbon.mgz` 以 nearest-neighbor 重采样到网络 MRI 网格；
- 将体积 ROI 以紧凑 label map 保存，并在低分辨率 feature map 上转换为 pooling mask；
- v3 cache 同时校验版本、shape、affine 和 alignment flag，避免静默加载旧 cache。

已经完成的关键证据：

- ADNI ribbon cache：`2343/2343` 成功，`0` 失败，约 `1.7 GiB`；
- 5 名受试者 white/mid/pial 坐标可视化 QC 通过；
- mean valid DKT vertex fraction：`0.921741`；
- white-to-pial thickness：约 `2.23–2.56` MRI voxels；
- affine 修正使 MRI–ribbon foreground overlap 从 `0.83585` 提高到 `0.88706`，20 个 QC subject 中 18 个改善。

相关代码：

- `data/dataset.py`
- `data/ribbon_geometry.py`
- `data/precompute_ribbon_bundle.py`
- `data/qc_ribbon_cache.py`
- `data/visualize_ribbon_bundle.py`

### 3.2 MRI encoder

状态：**已实现、已验证**。

Teacher 与 Student 使用 `SFCNVolumeEncoder3D`：

```text
T1 MRI
  → stem, stride 2, 32 channels
  → block1
  → block2 / deep ROI feature, 128 channels
  → block3–block5
  → global projection
```

输出包括：

- high-resolution feature：供 ribbon anchor 使用；
- deep ROI feature map：供 legacy mask pooling 使用；
- global MRI feature：供 global classifier 使用；
- multi-scale feature list：当前保留接口，但尚未实现正式 multi-scale ribbon fusion。

### 3.3 Surface encoder 与 Surface ROI pooling

状态：**已实现、已验证**。

```text
surface_data
  → SurfaceEncoder
  → optional SurfaceRefiner
  → SurfaceROIPooling
  → S_surface [B,62,C]
```

`SurfaceROIPooling` 使用 DKT ROI mask 对 vertex features 做归一化加权平均。当前 surface 是 Teacher 的 privileged input；严格 MRI-only Student 推理时不得读取 surface。

### 3.4 Legacy MRI ROI classification path

状态：**已实现、已验证，并应继续保留**。

```text
MRI deep feature
  → MaskedVolumeTokenPooling
  → local ribbon-mask token + full ROI-mask token
  → MLP fusion
  → V_context [B,62,C]
```

分类主路径随后执行：

```text
S_surface, V_context
  → ROIClipProjector
  → z_surface, z_volume
  → concat + teacher_fusion
  → z_teacher
  → ROI classifier
```

global MRI feature 同时经过 global classifier，最终：

\[
\mathrm{logits}=\mathrm{global\_logits}+\sigma(a)\,\mathrm{roi\_logits}.
\]

Phase 1/1.1 已证明 pure ribbon 没有稳定超过这一 legacy path，因此它是当前锁定的分类骨干。

### 3.5 ROI-CLIP alignment

状态：**已实现、已验证**。

`ROIClipProjector` 对 surface 与 volume ROI tokens 分别线性投影并 L2 normalize：

\[
z_s=\mathrm{norm}(W_sS),\qquad z_v=\mathrm{norm}(W_vV).
\]

现有 ROI-CLIP loss 将 subject 与 ROI 维展平。它属于 legacy Teacher 的 modality alignment，不应与 Phase 2.1 新增的 subject-local cross-geometry loss 混淆。

### 3.6 DiseasePrototypeBank

状态：**已实现、已验证基础功能；尚未迁移到 graph-enhanced state**。

prototype 参数形状：

```text
[num_classes, num_rois, prototype_dim]
```

当前分别对 `z_surface`、`z_volume`、`z_teacher` 计算 ROI-level class prototype logits。最终方法中应改为主要作用于尚未实现的 `z_teacher_graph`。

### 3.7 Relation consistency

状态：**legacy relation loss 与 DiseaseAwareROIGraph 均已实现；G0–G3 三折开发实验及随机拓扑控制已完成**。

当前 relation 定义为 token cosine matrix：

\[
R=ZZ^\top.
\]

Teacher 原有 surface/volume relation consistency 仍是普通 cosine relation。2026-10-19 新增的 `DiseaseAwareROIGraph` 对 `z_teacher` 执行 full ROI self-attention，并将真实 DKT boundary strength 与 bilateral homology 作为 soft additive bias；graph-enhanced classifier/prototype/relation 已接线。三折 clinical-off development 结果中，G3 相对 G0 的 mean MCC delta 为 `+0.02083`，但仅 2/3 folds 改善，paired subject bootstrap 95% CI 为 `[-0.02188, 0.06107]`。因此可以写“graph 未造成稳定下降并通过进入 Student 的开发 gate”，不能写“graph 显著改善分类”或“学到疾病传播”。

### 3.8 Clinical fusion

状态：**代码已实现；development-only label-proximity audit 已完成，不进入当前主分析**。

现有三种接口：

- `gated`：clinical feature 与 global MRI feature gated fusion；
- `cross`：imaging token 与 clinical tokens 的 cross transformer；
- `atlas_prompt`：clinical domain prompt 调制 global/ROI tokens。

当前 Phase 2.1 matched experiments 继承了 clinical-enabled baseline，因此它们证明的是“在相同 clinical 设置下加入 `L_CG` 的增量”，不是纯 MRI/Surface Teacher 的绝对表现。

五折 train/validation-only audit 显示：safe-background clinical-only MCC 为 `0.17546 ± 0.04337`，high-risk cognition-only MCC 为 `0.67770 ± 0.01300`，current clinical-29 MCC 为 `0.69493 ± 0.02733`。`current_clinical_29 − safe_background` 的 subject-bootstrap MCC 差为 `+0.52084`，95% CI `[0.48386, 0.55921]`。因此 Graph Stage A 主分析已锁定为 clinical-off；在来源诊断 adjudication protocol 被完整记录前，不把 FAQ/memory/fluency branch 作为主方法贡献。

### 3.9 White–pial RibbonSampler

状态：**已实现、几何已验证；作为分类替代路径证据不足**。

实现包括：

- FreeSurfer white/pial 坐标到 cropped MRI voxel/grid coordinate 的显式 affine 转换；
- `grid_sample(..., align_corners=False)`；
- grid 顺序严格为 `(x=W,y=H,z=D)`；
- 多深度采样、uniform mean 或 depth attention；
- vertex-to-ROI scatter pooling；
- vertex/ROI validity diagnostics；
- deep feature 与 high-resolution feature 两种 stage；
- `legacy_mask`、`ribbon`、`hybrid_ribbon` feature flags。

Phase 1.1 发现：

- deep-stage adjacent-depth cosine 为 `0.999319`，深度方向近似退化；
- high-resolution inner-depth cosine 为 `0.997458`，有所改善但仍非常相似；
- pure high-resolution ribbon 三折 mean MCC delta 为 `+0.003995`，paired SD 为 `0.008885`，不稳定；
- hybrid fold-0 MCC 为 `0.709173`，低于 legacy `0.710976`，且 gate 在 auxiliary loss 激活后从约 `0.08` 跳到 `0.59`。

结论：RibbonSampler 作为 geometry anchor 有价值，但不能替代 legacy volume context。

### 3.10 Phase 2.1 hard Surface↔Ribbon correspondence

状态：**已实现、三折验证通过**。

监督分支：

```text
SurfaceEncoder → SurfaceROIPooling → W_s → normalize → S

MRI encoder stem feature
  → uniform inner-5 RibbonSampler
  → W_r → normalize → R
```

每个 subject 独立形成 `62×62` cosine matrix：

\[
\mathrm{sim}_{b,i,j}=S_{b,i}^{\top}R_{b,j}/\tau.
\]

使用双向 hard identity target：

\[
L_{CG}=\tfrac12[L_{CE}(S\rightarrow R)+L_{CE}(R\rightarrow S)].
\]

锁定配置：

```yaml
depths: [0.2, 0.35, 0.5, 0.65, 0.8]
depth_attention: false
projection_dim: 128
temperature: 0.1
lambda_cg: 0.01
target: hard_same_roi
```

重要边界：

- `R_anchor` 不替换 `V_context`；
- `R_anchor` 不进入 classifier；
- 不存在 hybrid fusion gate；
- `L_CG` 更新 surface encoder/projector、ribbon projector 和 MRI stem；
- 因 anchor 取自 stem 输出，`L_CG` 在结构上不更新 block1–block5/global projection。

三折结果：

- Acc@1：`0.26160 ± 0.01662`，随机 `0.01613`；
- Acc@5：`0.63972 ± 0.01664`，随机 `0.08065`；
- MRR：`0.43137 ± 0.01656`，随机 `0.07601`；
- diagonal gap：`0.30506 ± 0.01566`；
- paired mean MCC delta：`+0.00170`。

这支持“学到稳定 ROI correspondence 且未造成稳定分类下降”，不支持“显著提高分类”。

### 3.11 Phase 2.1.5 correspondence audit

状态：**离线审计已完成**。

主要发现：

- same-subject/same-ROI cosine：`0.33219 ± 0.00747`；
- different-subject/same-ROI：`0.33012 ± 0.00753`；
- `G_subject=0.00206 ± 0.00043`，三折为正但幅度很小；
- correspondence excess 中 ROI identity share 为 `99.32%`；
- subject-specific share 为 `0.68%`；
- subject shuffle 使 Acc@1 从 `0.26160` 降到 `0.25773`，相对下降 `1.48%`。

因此允许的论文表述是：

> 模型获得了强 atlas ROI identity correspondence，并包含微弱但一致的 subject-specific increment。

不允许表述为：

> 模型主要学习了个体化 surface–MRI 精细对应。

### 3.12 真实 DKT topology 资产

状态：**矩阵、审计与模型接线均已完成**。

已经生成：

- `A_anat [62,62]`：fsaverage triangular mesh parcel adjacency；
- `B_boundary [62,62]`：共享 boundary mesh-edge count；
- `A_boundary [62,62]`：row-normalized boundary strength；
- `A_hom [62,62]`：严格一对一双侧同源矩阵。

统计：

- 160 条无向 anatomical edges；
- degree min/median/mean/max：`3/5/5.161290322580645/9`；
- 19,563 条共享 boundary edges；
- raw adjacency/boundary symmetric；
- zero diagonal；
- invalid cortical DKT labels：`0`；
- 每个 ROI 恰有一个 bilateral homologue。

由于计算节点 FreeSurfer 二进制库不兼容，DKTatlas40 annotation 由 FreeSurfer 自带 `bert` 经 registered sphere 最近邻转移到 fsaverage。该 provenance 必须保留在方法与局限性中。当前 `A_boundary` 与 `A_hom` 已作为 Graph Teacher/Student 的 additive attention bias；Laplacian positional encoding 作为 Student 的固定 template query buffer。

### 3.13 当前 MRI-only Student

状态：**Graph Student 已实现；纠正后的 S0/S3/S4/S5/S6 matched 三折组件归因完成，开发集锁定 S4**。

当前 Student：

```text
image
  → SFCN MRI encoder
  → fixed DKT normalized-Laplacian PE + trainable ROI residual
  → query-to-feature-map attention
  → z_student_base
  → DiseaseAwareROIGraph(A_boundary, A_hom)
  → z_student_graph
  → ROI/global classifier + prototype
```

vNext 已有蒸馏项：

- classification CE；
- logit KD；
- Teacher graph-state ROI KD：`z_teacher_graph → z_student_graph`；
- graph relation KD；
- prototype posterior KD。

完成 gate 验证的 S6 development loss 权重是：

```yaml
ce: 1.0
vnext_kd_logit: 0.5
vnext_kd_roi: 1.0
vnext_kd_graph: 0.2
vnext_kd_prototype: 0.2
```

S0 使用完全相同的 MRI-only architecture、subjects、seed、optimizer 和学习率，但所有 KD 权重为零。S0/S6 都从随机初始化开始，`warmup_epochs=0`，所有 Student 参数从 epoch 1 起用 `1e-4` 学习率训练。早期 `warmup=5 / lr=2e-6` 运行已判定为无效协议，不进入任何结论。

严格 MRI-only contract 已修正为默认 `student_use_clinical=false`，其真实 forward 可以只调用 `student(image)`。

但历史五折结果显示：

- MRI-only scratch：ACC `0.5444 ± 0.0412`；
- true MRI-only KD：ACC `0.4853 ± 0.0237`；
- paired ΔACC：`-0.0597524541186513`，95% CI `[-0.08706786171574904,-0.03243704652155356]`；
- exact McNemar `p=1.6785686427324062e-05`。

因此历史结果不能支持 privileged distillation。vNext 把 KD target 改为 privileged graph state，并用纠正后的 matched S0/S6 三折重新验证；S6 通过预先锁定的 development gate 后，已完成全部 progressive component ablation。当前仍需完整 CV 和外部验证后才能形成标题级结论。

纠正后的 vNext development 结果：

| Variant | MCC | AUC | Accuracy | Macro-F1 | Balanced accuracy |
|---|---:|---:|---:|---:|---:|
| S0 matched scratch | `0.27704 ± 0.03173` | `0.71830 ± 0.01595` | `0.49560 ± 0.02572` | `0.45080 ± 0.06271` | `0.53218 ± 0.05446` |
| S3 + logit KD | `0.30505 ± 0.01183` | `0.72422 ± 0.02664` | `0.52768 ± 0.01331` | `0.50789 ± 0.01245` | `0.55300 ± 0.01117` |
| **S4 + ROI graph-state KD** | **`0.31227 ± 0.04766`** | **`0.73894 ± 0.02805`** | `0.53872 ± 0.03044` | `0.51942 ± 0.04210` | `0.55227 ± 0.03053` |
| S5 + graph relation KD | `0.29117 ± 0.06482` | `0.72839 ± 0.04598` | **`0.54821 ± 0.03775`** | **`0.54587 ± 0.04556`** | **`0.55976 ± 0.05848`** |
| S6 full graph-state KD | `0.29141 ± 0.04274` | `0.72518 ± 0.01846` | `0.53331 ± 0.02600` | `0.52528 ± 0.01171` | `0.55677 ± 0.01168` |

progressive mean ΔMCC 依次为：S3−S0 `+0.02800`（3/3 folds 正向）、S4−S3 `+0.00723`（1/3）、S5−S4 `−0.02110`（1/3）、S6−S5 `+0.00024`（1/3）。S4 相对 S0 的 fold mean ΔMCC 为 `+0.03523`，pooled ΔMCC 为 `+0.05707`，paired bootstrap 95% CI `[+0.01133,+0.10529]`，McNemar exact `p=0.005468`。因此按“matched 三折 mean MCC 最高”锁定 S4，删除 relation/prototype KD；S4−S3 的 95% CI `[-0.02214,+0.03637]` 仍要求对 ROI-state 独立贡献保持谨慎。

---

## 4. 当前 Teacher 的真实 forward 与 loss

### 4.1 当前推荐的 P2-B Teacher

```text
MRI ───────────────────────────────────────────────────────────────┐
 │                                                                │
 ├─ SFCN stem → high-res feature → RibbonSampler → R_anchor ─┐     │
 │                                                           │     │
 ├─ SFCN deep feature → MaskedVolumeTokenPooling → V_context │     │
 │                                                           │     │
 └─ global feature → optional clinical fusion → global logits│     │
                                                             │     │
Surface → SurfaceEncoder → SurfaceROIPooling → S_surface ────┘     │
              │                         │                           │
              │                         └─ hard L_CG with R_anchor  │
              ▼                                                     │
S_surface + V_context → ROI-CLIP → teacher_fusion → ROI logits ────┘
                                      │
                                      └→ prototype/relation outputs
```

### 4.2 当前 loss

在 auxiliary warmup 之后，Teacher 大致使用：

\[
L_T=
L_{CE}
+0.01L_{ROIClip}
+0.01L_{prototype}
+0.005L_{relation}
+0.01L_{CG}.
\]

这不是未来 graph Teacher 的最终 loss，但应作为所有下一阶段实验的 matched baseline。

---

## 5. 已实现内容状态矩阵

| 模块 | 代码 | 单元/接口验证 | 正式实验 | 当前结论 |
|---|---:|---:|---:|---|
| DKT 62-ROI atlas/cache | 是 | 是 | 是 | 保留 |
| affine-aware MRI/atlas alignment | 是 | 是 | 是 | 强制保留 |
| SurfaceEncoder/ROIPooling | 是 | 是 | 是 | Teacher privileged path |
| SFCN MRI encoder | 是 | 是 | 是 | Teacher/Student backbone |
| legacy MaskedVolumeTokenPooling | 是 | 是 | 是 | 分类主路径 |
| ROI-CLIP | 是 | 是 | 是 | legacy alignment |
| DiseasePrototypeBank | 是 | 是 | 是 | 保留，未来迁移到 graph state |
| cosine relation consistency | 是 | 是 | 是 | 不是 graph reasoning |
| clinical gated/cross/atlas_prompt | 是 | 是 | 历史/基线 + audit | label-proximal；主分析关闭 |
| RibbonSampler | 是 | 是 | 是 | 作为 anchor，不替代 legacy |
| hybrid ribbon gate | 是 | 是 | fold-0 | 负结果，不进入主方法 |
| hard Surface↔Ribbon `L_CG` | 是 | 是 | 三折 | 保留，`lambda=0.01` |
| correspondence audit | 是 | 是 | 三折 validation | ROI identity 主导 |
| true DKT topology matrices | 是 | 是 | 离线审计 | 已接入 Graph soft bias |
| MRI-only Student contract | 是 | 是 | 历史五折 + S4/S6 三折自动审计 | 开发集选定的三个 S4 checkpoints 均通过；仅 `image` 为必需输入 |
| DiseaseAwareROIGraph | 是 | 是 | G0–G3 三折 + G3-R | gate 通过，但无稳定 anatomy-specific 分类收益 |
| graph-enhanced prototype/classifier | 是 | 是 | 三折 development | 输入统一为 graph state |
| topology-soft correspondence | 否 | 否 | 否 | 只做可选消融 |
| topology-seeded Student query | 是 | 是 | S0/S3/S4/S5/S6 纠正后三折完成 | 固定 template PE + trainable residual |
| graph/ROI-state KD | 是 | 是 | progressive 三折组件归因完成 | 锁定 S4；relation/prototype KD 删除 |
| ROI-wise cognitive modulation | 否 | 否 | 否 | label-proximity 风险高，暂缓 |
| PET privileged branch | 否 | 否 | 否 | 最后阶段可选 |

---

## 6. 已经完成的实验

### 6.1 Phase 1：legacy_mask 与 pure ribbon

fold-0 matched comparison：

| Metric | legacy_mask | ribbon | ribbon − legacy |
|---|---:|---:|---:|
| Best validation MCC | 0.710976 | 0.724350 | +0.013374 |
| Test ACC | 0.807440 | 0.789934 | -0.017505 |
| Test AUC | 0.930568 | 0.928009 | -0.002560 |
| Test MCC | 0.685984 | 0.661553 | -0.024431 |
| Test macro-F1 | 0.788576 | 0.774144 | -0.014432 |

结果：pure ribbon replacement 未通过严格 gate。该 test 已被查看，不能继续用于模型选择。

### 6.2 Phase 1.1：ribbon stabilization

三折 development-only：

| Fold | Legacy MCC | Inner-uniform ribbon MCC | Delta |
|---:|---:|---:|---:|
| 0 | 0.710976 | 0.725100 | +0.014124 |
| 1 | 0.704449 | 0.701963 | -0.002487 |
| 2 | 0.674381 | 0.674730 | +0.000349 |
| Mean | 0.696602 | 0.700597 | +0.003995 |

结果：ribbon 与 legacy 基本持平，但没有稳定优势；hybrid gate 也未通过 fold-0 gate。

### 6.3 Phase 2.1：hard correspondence

fold-0 lambda sweep：

| lambda_cg | MCC | Acc@1 | Acc@5 | MRR | Gap |
|---:|---:|---:|---:|---:|---:|
| 0.01 | 0.71169 | 0.27560 | 0.65456 | 0.44540 | 0.30577 |
| 0.05 | 0.70907 | 0.35242 | 0.73515 | 0.52092 | 0.41198 |
| 0.10 | 0.70673 | 0.34812 | 0.72923 | 0.51645 | 0.38026 |

三折 P2-A/P2-B：

| Fold | P2-A MCC | P2-B MCC | Delta MCC | Acc@1 | Gap |
|---:|---:|---:|---:|---:|---:|
| 0 | 0.71098 | 0.71169 | +0.00071 | 0.27560 | 0.30577 |
| 1 | 0.70445 | 0.71195 | +0.00750 | 0.26597 | 0.32034 |
| 2 | 0.67438 | 0.67127 | -0.00311 | 0.24323 | 0.28905 |
| Mean | 0.69660 | 0.69830 | +0.00170 | 0.26160 | 0.30506 |

结果：correspondence 成立，分类没有稳定下降，但分类增益很小。

### 6.4 Phase 2.1.5：correspondence audit

已经完成：

- subject-specific 2×2 similarity；
- 20 个 deterministic derangements 的 subject shuffle；
- fsaverage DKT mesh adjacency/boundary/homology；
- centroid spatial top-3 与 mesh neighbour 对照；
- top-1 mismatch decomposition；
- graph-distance TDE/ETD；
- 4 diagnostic batches/fold 的 gradient cosine。

关键 mismatch 汇总：

| Category | Count | Fraction |
|---|---:|---:|
| True mesh neighbour | 47,685 | 0.3778526148969889 |
| Bilateral homolog | 4,129 | 0.032717908082408875 |
| Spatial-only | 1,301 | 0.010309033280507132 |
| Remote ROI | 73,085 | 0.5791204437400951 |
| Total | 126,200 | 1.0 |

结果：真实 anatomical neighbourhood structure 存在，但还没有证据表明 topology-soft training 或 graph 一定提升分类。

### 6.5 软件验证

- Phase 1.1 时：`29 passed`；
- Phase 2.1 训练前：`34 passed`；
- Phase 2.1.5 最终：`40 passed`；
- P2-A/P2-B same-seed dry-run 初始 classification logits/CE 一致；
- P2-B peak allocated memory 约 `17.31 GiB`，硬件为 A100 80 GB。

---

## 7. 尚未实现的模块及其准确边界

### 7.1 Topology-soft correspondence

状态：**尚未实现**。

不应直接用 topology target 替换 hard identity。当前 evidence 显示 ROI identity 占 correspondence excess 的 `99.32%`。如果未来做 Phase 2.2，只建议：

\[
L_{CG}=(1-\alpha)L_{hard}+\alpha L_{topology},
\qquad \alpha\in\{0.05,0.10\}.
\]

topology target 应来自 `A_boundary` 与 `A_hom`，而不是 centroid top-3。该实验应作为消融，不应阻塞 graph 主线。

### 7.2 DiseaseAwareROIGraph

状态：**工程、回归测试、G0–G3 三折 development 实验和 G3-R 随机拓扑控制均已完成**。

推荐输入：当前 `z_teacher`，而不是 raw ribbon anchor。

第一版建议：

```yaml
roi_graph:
  enabled: true
  num_layers: 1
  num_heads: 4
  hidden_dim: 128
  dropout: 0.1
  residual_gate_init: 0.1
  use_anatomy: true
  use_boundary_strength: true
  use_homology: true
  use_centroid_spatial: false
```

graph attention：

\[
E_{ij}=Q_iK_j^\top/\sqrt d
+\beta_B\,\phi(A^{boundary}_{ij})
+\beta_H A^{hom}_{ij}.
\]

topology 只能作为 additive bias，不能 hard-mask remote connections。输出：

\[
Z_{teacher\_graph}=Z_{teacher}+g\odot W_o(\mathrm{softmax}(E)V).
\]

未加入 entropy loss 或强 KL prior，从而隔离 graph architecture 与 topology bias。本阶段结果为：

| Variant | MCC（mean ± SD） | 相对 G0 mean ΔMCC |
|---|---:|---:|
| G0 | 0.27153 ± 0.04199 | — |
| G1 | 0.28308 ± 0.04855 | +0.01155 |
| G2 | 0.27526 ± 0.04186 | +0.00373 |
| G3 | 0.29236 ± 0.01713 | +0.02083 |

G3 在 2/3 folds 超过 G0，但 paired bootstrap MCC CI 跨 0。G3-R 随机拓扑控制中，真实 G3 相对 G3-R 的 mean ΔMCC 为 `+0.01391`，仅 1/3 folds 改善；MCC bootstrap CI 为 `[-0.02561, 0.04900]`。真实 topology 明显提高 boundary correlation、anatomical/homology attention mass，但没有稳定证明 DKT prior 导致分类提升。

### 7.3 Graph-enhanced classifier/prototype

状态：**已实现并完成三折 development 验证**。

ROI classifier、relation 和 prototype 的主输入已统一为 `z_teacher_graph`。原 global MRI classifier 保留，避免 graph 失败时破坏强全局路径。

### 7.4 Topology-seeded MRI-only Student

状态：**已实现并完成纠正后的 S0/S3/S4/S5/S6 matched 三折 development 验证**。

Student query 已改为固定 DKT graph normalized-Laplacian positional encoding，经线性投影后加 trainable residual。该 topology 是 population/template prior，不是测试患者额外输入，因此符合 MRI-only inference。

### 7.5 Privileged graph-state distillation

状态：**progressive component ablation 已完成；开发集选择 S4（logit + ROI graph-state KD）**。

KD target 已从历史 `teacher z_volume` 改为：

```text
teacher z_teacher_graph → student z_student_graph
```

开发集锁定的 S4 loss：

```yaml
ce: 1.0
kd_logit: 0.5
kd_roi: 1.0
kd_graph: 0.0
kd_prototype: 0.0
kd_attention: 0.0
```

以上是进入完整 CV 的锁定配置，不是全局最优值。全部变体均随机初始化、从 epoch 1 全量训练，使用相同 subjects、architecture、seed、optimizer 与 `1e-4` 学习率；唯一差异是 progressive KD loss。旧的五轮冻结/`2e-6` preliminary runs 已标记无效，不用于结论。S4 以最高 mean MCC `0.31227` 被选中；其 pooled paired evidence 优于 S0，但 ROI-state 相对 S3 的独立增量 CI 跨 0，故证据强度仍是 development-level。

### 7.6 ROI-wise cognitive modulation

状态：**尚未实现**。

现有 clinical fusion 不是 ROI-wise cognitive modulation。真正模块需要：

- 将量表按 cognitive domain 结构化；
- observed value 与 missingness 分开编码；
- cognitive-domain token 对 ROI token 做 attention/modulation；
- 禁止 Student inference 读取 cognition；
- 在实现前完成诊断标签来源审计和 feature exclusion list。

### 7.7 PET privileged branch

状态：**尚未实现**。

需要真实 PET 图像、registration、ROI token 和 availability mask。不能使用 MRI tensor 模拟 PET。PET covariance 等 population statistics 必须在每个 training fold 内估计。

### 7.8 Availability-aware privileged fusion

状态：**尚未实现**。

当前没有统一融合 Surface/PET/Cognition 的 modality gate，也没有 privileged modality dropout。只有实际加入 cognitive/PET 后才需要该模块。

---

## 8. 建议的最终期刊方法

### 8.1 核心方法命名

建议技术描述：

> **CortexBridge: Anatomy-Prior Cross-Geometry Regularization and Cortical Graph Distillation for MRI-only Cognitive Diagnosis**

名称仍可调整，但应准确反映：cross-geometry 是 regularization/anchor，graph 与 distillation 才是最终 cortical state 建模核心。

### 8.2 最终 Teacher

```text
T1 MRI
  ├→ high-res stem → RibbonSampler → R_anchor ───────────────┐
  ├→ deep feature → legacy MaskedVolumeTokenPooling → V     │
  └→ global feature → global classifier                     │
                                                             │ L_CG
Surface → SurfaceEncoder → SurfaceROIPooling → S ────────────┘

S + V → ROI-CLIP → teacher_fusion → Z_base
Z_base + {A_anat,A_boundary,A_hom}
      → DiseaseAwareROIGraph
      → Z_teacher_graph
      ├→ ROI classifier
      ├→ DiseasePrototypeBank
      └→ graph/relation output

final logits = global logits + gated ROI logits
```

### 8.3 最终 Student

```text
T1 MRI only
  → MRI encoder
  → topology-seeded ROI queries
  → MRI-derived ROI tokens
  → same template DKT graph
  → Z_student_graph
  → ROI/global classifier
```

训练时蒸馏：

- prediction logits；
- graph-enhanced ROI state；
- row-wise graph distribution；
- disease prototype posterior。

测试时只能输入 `image`。

### 8.4 不进入核心方法的内容

- pure ribbon classification；
- hybrid ribbon fusion gate；
- centroid top-3 伪 anatomical graph；
- Sinkhorn OT；
- Grounding、LLM、text branch；
- 未完成 leakage audit 的 cognitive variables；
- 缺乏真实数据的 PET placeholder。

---

## 9. 期刊论文中如何组织模块

### 9.1 建议贡献点

最终论文最多保留三个贡献点：

1. **Geometry-aware auxiliary supervision**：利用真实 white–pial ribbon 建立 Surface↔MRI ROI correspondence，同时保留强 legacy MRI context path；
2. **Anatomy-prior disease-aware cortical graph**：将真实 mesh adjacency、boundary strength 和 bilateral homology 作为 soft prior，而由 subject tokens 学习 disease-sensitive graph；
3. **MRI-only cortical state distillation**：将 privileged Teacher 的 graph-enhanced ROI state 转移给不依赖 FreeSurfer/clinical/PET 的 Student。

不要把 cache、clinical MLP、prototype bank、每个 KD loss 都单独写成“主要创新”。它们属于支撑组件。

### 9.2 Methods 章节建议

```text
3.1 Problem formulation and privileged-training setting
3.2 DKT cortical representation and affine-aware preprocessing
3.3 Legacy MRI–surface ROI encoder backbone
3.4 Auxiliary surface–ribbon cross-geometry anchoring
3.5 Anatomy-prior disease-aware cortical graph
3.6 MRI-only topology-query Student
3.7 Cortical state, graph and prototype distillation
3.8 Training objectives and inference contract
```

### 9.3 Results 章节建议

```text
4.1 Dataset, splits, implementation and statistical protocol
4.2 Main internal cross-validation comparison
4.3 External/generalization evaluation
4.4 Cross-geometry ablation and correspondence audit
4.5 Topology/graph ablation
4.6 Distillation ablation and MRI-only deployment
4.7 Robustness, calibration and efficiency
4.8 Anatomical interpretation and failure analysis
```

### 9.4 可以立即写入论文的内容

- affine-aware DKT/ribbon preprocessing；
- ribbon geometry QC；
- pure ribbon replacement 负结果；
- hard correspondence 公式与三折 retrieval evidence；
- ROI identity/subject-specific audit；
- true DKT topology 构建方法；
- mismatch/TDE/ETD 审计；
- 现阶段 limitations。

### 9.5 现在不能写成完成贡献的内容

- topology-soft correspondence 提升性能；
- disease-aware graph improves classification；
- graph captures AD propagation；
- privileged Teacher improves MRI-only Student；
- cognitive modulation 提供 ROI-specific clinical knowledge；
- PET improves MRI-only inference；
- external generalization。

---

## 10. 投稿前还必须完成的实验

### 10.1 优先级 P0：Disease-aware graph 三折开发实验

状态：**已完成；选择 G3 进入 Student，但不支持 anatomy-specific 分类因果声明**。

| ID | 配置 | 目的 | 状态 |
|---|---|---|---|
| G0 | P2-B，无 graph | 锁定 baseline | 已完成三折 |
| G1 | graph，无 topology bias | graph architecture 对照 | 已完成三折 |
| G2 | graph + `A_boundary/A_anat` | anatomical prior 增量 | 已完成三折 |
| G3 | G2 + `A_hom` | bilateral homology 增量 | 已完成三折；选定 Student Teacher |
| G4 | G3 + centroid spatial | 检验 spatial prior 是否冗余 | 可选 |

至少报告：ACC、AUC、MCC、macro-F1、balanced accuracy、graph attention anatomical mass、homology mass、entropy、不同 seed/fold 稳定性。

进入 Student 阶段的 gate 已通过：

- graph 相对 G0 不出现稳定分类下降；
- 至少一个 topology configuration 在多数 folds 改善主选择指标；
- learned graph 不是 uniform、identity-only 或 topology copy；
- graph output 对 classifier 有非零贡献；
- legacy graph-off 模式保持可复现。

额外完成 G3-R deterministic topology permutation。真实 G3 的 attention diagnostics 更符合 DKT boundary/homology，但分类增益未稳定超过 G3-R。因此论文把真实 topology 表述为结构性归纳偏置和可解释 attention prior，不表述为已证明的性能来源。

### 10.2 优先级 P0：更新后的 MRI-only Student

历史 MRI-only KD 是负结果，因此按 matched architecture 重新建立对照。先完成 S0 对 S6 的预注册 gate；S6 通过后，已按 S3→S4→S5 顺序完成组件归因：

| ID | Student | KD target | 状态 |
|---|---|---|---|
| S0 | topology-seeded Graph Student scratch | 无 | 纠正后三折完成 |
| S1 | current learnable queries | old `z_volume` KD | 历史完成、负结果 |
| S2 | topology-seeded queries | CE only | 与 matched S0 合并，避免重复实验 |
| S3 | S0 + logit KD | Teacher logits | 三折完成；3/3 folds 高于 S0 |
| **S4** | **S3 + ROI state KD** | **`z_teacher_graph`** | **三折完成；mean MCC 最高，开发集锁定** |
| S5 | S4 + graph relation KD | Teacher graph relation | 三折完成；相对 S4 mean ΔMCC `−0.02110`，删除 |
| S6 | S5 + prototype KD | Teacher prototype posterior | 三折完成；相对 S5 mean ΔMCC `+0.00024`，删除 |

预先锁定的 S6 gate 为：mean ΔMCC `>0.01`、至少 `2/3` folds 的 MCC 改善，并且 AUC/ACC/macro-F1/balanced accuracy 的任一 mean delta 不低于 `-0.005`。该 gate 已通过并触发组件归因。按 matched 三折 mean MCC 的 development-only 选择规则，S4 以 `0.31227` 高于 S3 `0.30505`、S5 `0.29117` 和 S6 `0.29141`，因此锁定 S4 进入完整 CV；在完整 CV 和外部验证前仍不把 privileged distillation 写成已确证的标题级贡献。

### 10.3 优先级 P0：严格 MRI-only inference audit

状态：**已完成。开发集选定的 S4 三个正式 checkpoints 与 S6 三个 gate checkpoints 均通过自动 MRI-only audit。**

自动测试包括：

- inference dataloader 仅包含 `image + label + metadata`；
- 删除 surface/cache/clinical 路径后仍可运行；
- checkpoint 不要求 subject-specific atlas；
- topology PE 是固定 template buffer；
- batch 中添加/改变 clinical/surface 信息不会改变 Student logits；
- 推理所需位置参数仅为 `image`；额外传入任意 clinical tensor 不改变 logits。

### 10.4 优先级 P0：最终内部评估协议

当前 vNext 主要只有 3 个 development folds。投稿前需要：

- 锁定 architecture 和 hyperparameters；
- 完整 5-fold cross-validation 或预注册 train/validation/test protocol；
- subject-level out-of-fold prediction；
- paired bootstrap 95% CI；
- paired permutation/Wilcoxon 或适合 fold/subject 结构的检验；
- McNemar test 比较主要 classifier；
- 多重比较校正用于多项主要 ablation；
- 报告 mean、SD、CI，而不是只报最佳 fold。

### 10.5 优先级 P0：clinical leakage audit（实证部分已完成）

已完成五折 train/validation-only clinical-only 对照、subject-bootstrap CI、shuffled-label control 和 train/validation subject-overlap 检查；未读取 held-out test。仍必须从原始 cohort protocol 回答：

1. COG label 如何生成；
2. 每个 clinical variable 是否直接或间接参与诊断定义；
3. 哪些变量在真实 deployment 时可获得；
4. ~~clinical-only baseline 能达到什么水平~~（已完成）；
5. ~~移除高风险 cognition variables 后性能如何~~（clinical-only 分组已完成；imaging+safe/full 尚待最终消融）；
6. Teacher cognition 是否只作为 training privilege，Student inference 是否完全独立。

最少对照：

```text
MRI only
clinical only
MRI + safe demographics
MRI + full clinical
Surface + MRI without clinical
privileged Teacher → MRI-only Student
```

如果 clinical-only 已接近 full model，论文必须把任务定义为 multimodal diagnosis，而不能暗示主要性能来自 cortical representation。

### 10.6 优先级 P1：topology-soft correspondence 小型消融

只比较：

```text
hard identity
hard + 0.05 topology component
hard + 0.10 topology component
```

使用 `A_boundary + A_hom`。若没有分类或 graph-state收益，应删除 topology-soft training，不因已经实现而保留。

### 10.7 优先级 P1：外部验证

当前存在 OASIS3/Huashan cache，但 vNext graph/Student 尚无外部证据。需要预先明确两种不同问题：

- within-cohort cross-validation；
- ADNI-trained zero-shot external evaluation。

不得把两者混合平均。外部报告需要：

- scanner/site shift；
- demographics 与 label definition；
- missing modality rate；
- calibration shift；
- zero-shot 与允许 fine-tuning 的结果分开。

### 10.8 优先级 P1：标准基线

至少需要 matched preprocessing 下的：

- SFCN/3D CNN MRI-only；
- MRI + clinical baseline；
- surface-only baseline；
- direct surface-volume concatenation；
- current ROI-CLIP；
- P2-B cross-geometry regularized model；
- graph without anatomical prior；
- proposed graph Teacher；
- MRI-only scratch Student；
- proposed MRI-only KD Student。

如与公开方法比较，必须重新运行或严格说明引用值的数据集/划分不可直接比较。

### 10.9 优先级 P1：robustness 与 calibration

建议：

- seed sensitivity；
- atlas/ROI dropout；
- ribbon coordinate perturbation；
- missing clinical/PET；
- class imbalance sensitivity；
- ECE、Brier score、reliability diagram；
- per-class sensitivity/specificity；
- subgroup performance：sex、age、scanner/site；
- failure cases 与置信度分析。

### 10.10 优先级 P1：效率

报告：

- trainable parameters；
- FLOPs 或 MACs；
- peak GPU memory；
- training time/fold；
- inference latency/subject；
- Teacher preprocessing cost；
- Student 是否真正避免 FreeSurfer 的测试时成本。

### 10.11 优先级 P2：ROI-wise cognition 与 PET

只有 P0/P1 主链成立后再做。它们应作为扩展实验，而不是挽救主模型的模块。

### 10.12 投稿必需但当前材料不足的非模型信息

下列内容不能从当前代码和阶段报告完整恢复，投稿前必须补齐，不能猜测：

- cohort inclusion/exclusion flowchart；
- 各 fold、各类别的 subject 数与重复扫描处理规则；
- COG label 的临床定义与生成来源；
- age/sex/education/APOE 等 cohort characteristics 与组间检验；
- MRI scanner、field strength、sequence、site 分布；
- FreeSurfer/QC 失败样本及排除数量；
- longitudinal scan 是否按 subject 隔离，防止同一患者跨 split；
- ethics approval、consent/data-use statement；
- primary/secondary endpoint 的预先定义；
- external cohort label harmonization；
- code、checkpoint、derived topology 和 split 的 availability plan。

这些信息对 TMI/MedIA 的可信度与可复现性与模型消融同等重要。

---

## 11. 建议的完整消融矩阵

### 11.1 Teacher ablation

| Row | Legacy pooling | Hard CG | Soft topology CG | Graph | Anatomy | Homology | Clinical | 目的 |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| T0 | ✓ |  |  |  |  |  | matched | legacy baseline |
| T1 | ✓ | ✓ |  |  |  |  | matched | 已完成 P2-B |
| T2 | ✓ | ✓ |  | ✓ |  |  | matched | graph architecture |
| T3 | ✓ | ✓ |  | ✓ | ✓ |  | matched | anatomy prior |
| T4 | ✓ | ✓ |  | ✓ | ✓ | ✓ | matched | proposed Teacher |
| T5 | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | matched | optional topology-soft |
| T6 | ✓ | ✓ |  | ✓ | ✓ | ✓ | off | clinical dependence |

### 11.2 Student ablation

| Row | Query | Logit KD | ROI KD | Graph KD | Prototype KD | Inference input |
|---|---|---:|---:|---:|---:|---|
| S0 | learnable |  |  |  |  | MRI |
| S1 | topology-seeded |  |  |  |  | MRI |
| S2 | topology-seeded | ✓ |  |  |  | MRI |
| S3 | topology-seeded | ✓ | ✓ |  |  | MRI |
| S4 | topology-seeded | ✓ | ✓ | ✓ |  | MRI |
| S5 | topology-seeded | ✓ | ✓ | ✓ | ✓ | MRI |

### 11.3 Privileged extension ablation

| Row | Surface | Cognition | PET | Modality dropout | 说明 |
|---|---:|---:|---:|---:|---|
| P0 | ✓ |  |  |  | core Teacher |
| P1 | ✓ | ✓ |  | ✓ | leakage audit 通过后 |
| P2 | ✓ |  | ✓ | ✓ | 有真实 PET 时 |
| P3 | ✓ | ✓ | ✓ | ✓ | 完整 privileged extension |

---

## 12. 统计分析计划

### 12.1 主要终点

- primary：MCC；
- secondary：macro-F1、balanced accuracy、AUC、accuracy；
- Student deployment：MRI-only MCC 和 calibration；
- correspondence：Acc@1/5、MRR、diagonal gap；
- graph：TDE/ETD、anatomical attention mass、homology mass、entropy。

### 12.2 比较原则

- 所有 ablation 使用相同 split、seed、augmentation、early stopping 和 clinical setting；
- 主比较使用 paired out-of-fold predictions；
- fold 均值只作为摘要，统计检验尽量在 subject-level paired predictions 上进行；
- hyperparameter selection 不读取最终 test；
- 对多个模块/终点使用 Holm 或 FDR 校正；
- 同时报告 effect size 和 95% CI。

### 12.3 随机性

当前主要 seed 为 `3407`。期刊版本至少应对最终 baseline/proposed model 增加多 seed 稳定性，或解释计算约束并提供 fold-level paired CI。

---

## 13. 建议论文表格与图

### 主表

1. cohort 与 split characteristics；
2. internal main comparison；
3. external generalization；
4. Teacher module ablation；
5. Student KD ablation；
6. efficiency 与 MRI-only deployment cost。

### 关键图

1. 整体 Teacher–Student 架构图；
2. white–pial ribbon 与 MRI coordinate QC；
3. surface↔ribbon correspondence matrix；
4. normal versus subject-shuffled retrieval；
5. true DKT mesh graph 与 learned graph；
6. graph attention/mismatch anatomical decomposition；
7. Student calibration 与 confusion matrix；
8. external/site subgroup forest plot；
9. success/failure cortical maps。

不得把旧 spatial top-3 tolerance 称为 anatomical topology accuracy。统一写作 `spatially tolerant retrieval accuracy`。

---

## 14. 当前允许和不允许的论文论断

### 允许

- ribbon geometry/cache 经过 affine-aware QC；
- pure ribbon pooling 没有稳定优于 legacy pooling；
- hard Surface↔Ribbon loss 学到了显著高于随机的 ROI correspondence；
- 小权重 `L_CG` 未造成三折稳定分类下降；
- correspondence 主要由 ROI identity 驱动；
- subject-specific increment 为正但很弱；
- mismatch 对真实 mesh neighbours 显著富集；
- DiseaseAwareROIGraph 已实现，G3 通过进入 Student 的 development gate；
- G3 的结构诊断符合真实 DKT topology，但其分类优势尚未稳定超过随机 topology；
- 历史 MRI-only KD 没有成功；纠正后的 vNext graph-state KD 通过预注册三折 development gate；
- S6 inference 是严格 MRI-only，三个正式 checkpoint 均通过自动审计。

### 不允许

- ribbon pooling 已提高分类；
- correspondence 主要是个体特异；
- topology-soft target 已有效；
- disease-aware graph 已显著提高分类或真实 topology 已稳定优于随机 topology；
- graph 表示真实 AD disease propagation；
- privileged distillation 已获得稳定、显著且可外部泛化的最终收益；
- cognition/PET 已提供 ROI-level机制；
- vNext 已完成外部泛化。

---

## 15. 阶段路线与停止条件

### Stage A：Graph Teacher

状态：**已完成。** G0–G3 clinical-off 三折 development gate 选择 G3；G3-R 表明真实 topology 的结构诊断更合理，但尚未证明 anatomy-specific 分类增益。

### Stage B：Graph Student

状态：**已完成。** S0/S6 gate 通过，S3–S5 组件归因完成，开发集锁定 S4。S4 相对 S0 的 pooled MCC bootstrap CI 高于 0，但其 ROI-state 相对 S3 的独立增量不稳定；当前证据支持进入完整 CV，不支持把每个 KD target 都写成稳定贡献。

### Stage C：锁定与最终评估

冻结配置，补完整 CV、统计检验、外部验证和最终 test。测试集只运行一次。

### Stage D：可选 privilege

完成 cognition leakage audit 后决定 cognition；有真实 PET 数据、明确缺失机制和 registration 后才进入 PET。

### 每个后续阶段的工程验收

每个 phase 必须保存：

1. changed files 与 config diff；
2. forward tensor shapes；
3. backward/dry-run 结果；
4. GPU peak memory 与运行时间；
5. graph/ribbon/KD 专用 diagnostics；
6. feature flag 关闭后与 legacy 的等价性；
7. checkpoint backward-compatibility 说明；
8. `held_out_test_evaluated` 状态；
9. known limitations 与进入下一阶段的明确 gate。

---

## 16. 复现入口与证据位置

- Phase 1 report：`outputs/vnext_phase1_20260930/PHASE1_GATE_REPORT.md`
- Phase 1.1 report：`outputs/vnext_phase1_1_20261001/PHASE1_1_REPORT.md`
- Phase 2.1 report：`outputs/vnext_phase2_1_20261002/PHASE2_1_REPORT.md`
- Phase 2.1.5 audit：`outputs/vnext_phase2_1_5_20261003/PHASE2_1_5_CORRESPONDENCE_AUDIT.md`
- Phase 2.1.5 metrics：`outputs/vnext_phase2_1_5_20261003/audit_metrics.json`
- true topology：`outputs/vnext_phase2_1_5_20261003/true_dkt_topology.npz`
- clinical leakage audit：`outputs/vnext_clinical_leakage_audit_20261018/CLINICAL_LEAKAGE_AUDIT.md`
- Graph Stage A：`outputs/vnext_graph_stageA_20261019/GRAPH_STAGE_A_REPORT.md`
- Graph topology-randomization：`outputs/vnext_graph_stageA_20261019/GRAPH_RANDOMIZATION_REPORT.md`
- Graph Student gate：`outputs/vnext_graph_student_20261019/GRAPH_STUDENT_REPORT.md`
- Graph Student component ablation：`outputs/vnext_graph_student_20261019/GRAPH_STUDENT_COMPONENT_ABLATION_REPORT.md`
- Graph Student machine-readable summary：`outputs/vnext_graph_student_20261019/graph_student_component_summary.json`
- locked S4 config：`configs/runtime_vnext_graph_student_locked_s4.yaml`
- stage completion audit：`docs/CortexBridge_vNext_Completion_Audit_20261020.md`
- historical TMI evidence：`outputs/tmi_complete_20260924/RESULTS_REPORT.md`
- implementation guide：`../CortexBridge_vNext_Implementation_Guide.md`

配套的 machine-traversable research artifact 位于：

```text
docs/ara_cortexbridge_vnext/
```

---

## 17. 最终决策

当前已锁定：

```text
legacy classification backbone
+ hard Surface↔Ribbon auxiliary alignment (lambda_cg=0.01)
+ DiseaseAwareROIGraph with true DKT boundary/homology soft priors
```

当前获得 development-level 支持并已锁定的 Student 是：

```text
privileged Graph Teacher
  -> logit KD + graph-enhanced ROI-state KD
  -> strictly MRI-only S4 Student
```

S0/S6 gate 与 S3–S5 组件归因均已完成；relation KD 和 prototype-posterior KD 不进入锁定方案。下一优先级是用 S4 补完整 CV、外部验证、calibration、subgroup 与效率实验，而不是继续增加 cognition、PET 或 topology-soft correspondence 等主模型模块。

完整 CV 必须直接使用 `configs/runtime_vnext_graph_student_locked_s4.yaml`；除 fold 与输出目录外不得再根据 validation 或 held-out test 改动模型、loss、学习率或数据输入。

---

## 18. 2026-10-20 no-MMSE 临床辅助更新

已完成 S4 的 C0–C4 三折 gated clinical ablation，并加入 exact-ID clinical-only controls。开发集证据支持把整体方法拆成两个清晰的推理设置：

```text
CortexBridge S4-MRI:
MRI-only topology-seeded Graph Student
+ logit KD + Teacher graph-enhanced ROI-state KD

CortexBridge S4-C:
the same S4 Student
+ inference-time gated background/GDS/NPI-Q branch
```

S4-C 采用 C2：17 个输入变量为 age、gender、education、APOE、GDS 与 12 个 NPI-Q 条目；每个变量同时编码数值和缺失标志。它明确排除 MMSE、LM memory、animal fluency 与 FAQ。

三折 C2 的 MCC 为 `0.44281 ± 0.02970`，相比 S4-MRI pooled ΔMCC `+0.13140`（95% CI `[+0.08592,+0.17725]`），相比 matched clinical-only pooled ΔMCC `+0.10350`（95% CI `[+0.06346,+0.14313]`）。C3/C4 虽有更高绝对分数，但没有证明优于其 cognition/full29 clinical-only 对照，故只作为 diagnosis-proximal upper bounds。

论文组织应保持：S4-MRI 是主方法与机制证据；S4-C 是需要额外临床输入的扩展。两者必须分表或按输入设置分块报告。新增证据位置：

- clinical-assisted report：`outputs/vnext_graph_student_clinical_20261020/S4_CLINICAL_ASSISTED_ABLATION_REPORT.md`
- analysis tables：`outputs/vnext_graph_student_clinical_20261020/analysis/`
- matched clinical audit：`outputs/vnext_clinical_leakage_audit_s4c_20261020/`
- locked S4-C config：`configs/runtime_vnext_graph_student_locked_s4c_safe_neuro.yaml`

下一步不再调整 C2 的 feature set 或 gate。分别使用 locked S4-MRI 与 locked S4-C 补完整 CV、calibration、subgroup、missingness robustness 和 external validation；最后才允许一次性 held-out test。

---

## 19. 2026-10-21 MedicalNet/Med3D Student backbone 审计

已实现 MedicalNet-compatible 3D ResNet10 Student adapter，并在锁定的三折 development protocol 下完成 M0–M4：SFCN S4 主线、ResNet10 scratch、Med3D pretrained、pretrained + logit KD、pretrained + logit KD + ROI-state KD。

结果不支持用 Med3D ResNet10 替换当前 SFCN Student：M0 的 MCC 为 `0.31227 ± 0.04766`，最佳 MedicalNet-family 变体 M1 为 `0.26318 ± 0.04678`；M1−M0 paired pooled ΔMCC 为 `−0.04708`（95% CI `[−0.08691,−0.00931]`）。M2 相对 M1 明显下降；M3 的 logit KD 能部分恢复 M2；M4 的 ROI-state KD 相对 M3 没有稳定额外增益。

预先约定的 Med3D Teacher 触发条件未满足：M2−M4 pooled ΔMCC 为 `−0.06032`（95% CI `[−0.11219,−0.01078]`），方向与“M2 显著优于 M4”相反。因此：

- 不使用 Med3D ResNet10 替换锁定的 S4-MRI backbone；
- 不启动 Med3D cross-geometry Teacher 重训；
- 把该阶段作为预训练迁移负结果、logit KD 恢复效应和 ROI-state KD 适配边界的消融证据；
- 后续 journal 实验仍直接使用锁定的 S4-MRI 与 S4-C，不再基于该 development 结果搜索 backbone。

全部 12 个新 Med3D/MedicalNet folds 和复用的 3 个 M0 folds 均为 development-only；最佳 MedicalNet-family 变体 M1 的三个 checkpoint 均通过 MRI-only audit。证据位置：

- report：`outputs/vnext_med3d_resnet10_20261020/MED3D_RESNET10_STUDENT_ABLATION_REPORT.md`
- summary：`outputs/vnext_med3d_resnet10_20261020/med3d_student_ablation_summary.json`
- metrics：`outputs/vnext_med3d_resnet10_20261020/med3d_student_ablation_metrics.csv`
- reproducible summarizer：`summarize_med3d_student_ablation.py`
