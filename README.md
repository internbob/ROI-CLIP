# CortexWeave-ROIClip

这是按 `CortexWeave_ROIClip_method_architecture.md` 落地的一套独立工程，目录放在：

`/home_data/home/zhouwl2023/project/Cortexbridge/cortexweave_roiclip`

整体风格参考了 `/home_data/home/zhouwl2023/project/Cortexweave_DQ_MoE` 的 `dataset / model / train` 组织方式，但把方法主线改成了：

- teacher: `surface ROI tokens <-> volume ROI tokens` 的 ROI-CLIP 对齐
- student: MRI-only 的 learnable ROI queries
- distillation: `logit + token + attention + relation + prototype`

## 目录

```text
cortexweave_roiclip/
├── configs/
├── data/
├── losses/
├── models/
├── train_teacher.py
├── train_student.py
├── evaluate.py
├── visualize_roi_attention.py
└── README.md
```

## 数据约定

训练脚本默认沿用类似 `Cortexweave_DQ_MoE` 的 split 结构：

```text
split_root/
└── cross0/
    ├── train.csv
    ├── valid.csv
    └── test.csv
```

当前默认已经配置为：

- `surface_root = /public_bme/data/zhouwl2023/ADNI/BrainSurf_Results`
- `split_root = /home_data/home/zhouwl2023/project/Cortexbridge/cortexweave_roiclip/lookupcsv-full-huashan-adni-teacher-clean`
- `allowed_subject_prefixes = ["sub-ADNI"]`

CSV 至少需要：

- `filename`
- `path` 或者能在 `surface_root/<subject>/...` 下自动找到 MRI
- 任务标签列，例如 `AD`

说明：

- 这套 `lookupcsv-full-huashan-adni-teacher-clean` 是当前默认 split。
- 它基于 `New-AD/BHG-model/lookupcsv-full-huashan` 先过滤得到 ADNI-only 版本，再对 `cross0` 剔除了 15 个缺失 surface 的 ADNI 样本。
- 当前这份 CSV 的 `path` 已经是集群上的 Linux 路径，例如 `/public_bme/data/zhouwl2023/ADNI/ADNI_Linear/`。
- 代码层也会再次按 `subject_prefix` 过滤，只保留 `sub-ADNI...`，避免训练时混入 Huashan 样本。
- 代码仍然保留 MRI 本地回退逻辑；如果 CSV 路径不可用，仍会自动尝试 `BrainSurf_Results/<subject>/input/image.nii.gz`、`.../fake_bids/...T1w.nii.gz`、`.../workbench/...image.nii.gz` 或 `.../freesurfer/.../mri/T1.mgz`。

## ROI 掩膜输入

当前默认已经切到真实 atlas 模式：

- `data.atlas.enabled = true`
- `data.atlas.name = "DKTatlas"`
- `model.num_rois = 62`

也就是说，训练默认不再使用之前的 `16` 个空间分块 ROI，而是使用 FreeSurfer `DKTatlas` 的真实左右半球 cortical ROI，对应关系来自：

- surface: `lh/rh.aparc.DKTatlas.annot`
- volume: `aparc.DKTatlas+aseg.mgz`

这两侧 ROI 是一一对应的，volume 侧为了节省内存，代码内部改成了紧凑 `label-map` 表示，只在低分辨率特征图上再转 one-hot pooling。

### MRI-atlas 空间对齐与 v3 cache

体积侧采用以下固定流程，不能再对 MRI 与 FreeSurfer 标签分别做中心裁剪：

1. 从 split CSV 的 `path/subject/session/t1/filename.nii.gz` 读取 MRI，并中心裁剪/补零到 `(160, 192, 160)`。
2. 用裁剪起点更新 MRI affine，使裁剪后 voxel `(0, 0, 0)` 仍指向正确的世界坐标。
3. 根据该 target shape 和 adjusted affine，用最近邻插值分别重采样 `aparc.DKTatlas+aseg.mgz`、`lh.ribbon.mgz` 和 `rh.ribbon.mgz`。
4. 在已经对齐的 MRI 网格上构造 ROI/local-ribbon label map，再用最近邻规则降采样到 `(20, 24, 20)`。

当前 cache 版本为 `v3`。每个 `.npz` 都保存 `target_mri_shape`、`target_mri_affine` 和空间对齐标志；加载时会同时核对版本、shape 与 affine，旧 v1/v2 cache 不会被静默复用。默认 ADNI cache 为：

`/home_data/home/zhouwl2023/Data/CortexWeave_ROIClip_cache/adni_dktatlas_5124_roi20x24x20_affine_v3`

同级目录下还生成了 `oasis3_dktatlas_5124_roi20x24x20_affine_v3` 和 `huashan_dktatlas_5124_roi20x24x20_affine_v3`。每个目录的 `cache_manifest.json` 记录数据源、样本数、对齐方法和输出 shape。

推荐每个 subject 提前放一个 `roi_bundle.npz`，可放在下面任一位置：

- `<surface_root>/<subject>/roiclip/roi_bundle.npz`
- `<surface_root>/<subject>/roi_clip/roi_bundle.npz`
- `<surface_root>/<subject>/masks/roi_bundle.npz`
- `<surface_root>/<subject>/roi_bundle.npz`

`roi_bundle.npz` 推荐包含：

- `surface_roi_masks`: `[K, N]`
- `local_ribbon_masks`: `[K, H, W, D]`
- `roi_volume_masks`: `[K, H, W, D]`
- `global_brain_mask`: `[H, W, D]`，可选

如果关闭 atlas 模式，或者没有预计算 bundle，代码才会启用回退：

- `surface_roi_masks` 用球面坐标的角度分桶自动生成
- `local_ribbon_masks / roi_volume_masks` 用 MRI 脑区空间分块和脑壳层近似生成

这个回退主要用于先把流程跑通；真正复现实验时，建议替换成你方法文档里定义的 white-pial ribbon / ROI mask。

## vNext Phase 1：white–pial multi-depth ribbon

Phase 1 使用独立 ribbon cache，不修改也不废弃上述 v3 atlas cache。预计算真实
FreeSurfer `white/pial` 坐标：

```bash
python data/precompute_ribbon_bundle.py \
  --ribbon-cache-root /home_data/home/zhouwl2023/Data/CortexWeave_ROIClip_cache/adni_dktatlas_5124_ribbon_v1 \
  --fold 0 --num-workers 8
```

正式训练前，至少抽查 5 个受试者的坐标叠加图：

```bash
python data/visualize_ribbon_bundle.py \
  --mri /path/to/subject_T1.nii.gz \
  --bundle /path/to/ribbon_cache/subject.npz \
  --output /path/to/qc/subject.png
```

Phase 1 ribbon 实验：

```bash
python train_teacher.py --config \
  configs/data.yaml \
  configs/teacher.yaml \
  configs/runtime_vnext_phase1_ribbon_fold0.yaml
```

`model.volume_pooling.mode=legacy_mask` 保持历史模型行为；只有显式设置为
`ribbon` 时才要求 ribbon bundle。ribbon 模式不存在随机坐标或旧 mask 的静默回退。
空间增强中的 voxel shift 必须保持为 0，除非同时变换 `ribbon_grid`。

## 训练

Teacher:

```bash
cd /home_data/home/zhouwl2023/project/Cortexbridge/cortexweave_roiclip
python train_teacher.py
```

Student:

```bash
python train_student.py --config \
  configs/data.yaml \
  configs/student.yaml \
  configs/runtime_vnext_graph_student_locked_s4.yaml \
  --fold 0 --device cuda \
  --output_dir /path/to/locked_s4_output
```

`train_student.py` 会优先读取：

1. `training.teacher_checkpoint`
2. `--checkpoint`

如果给的是实验目录，它会自动找 `fold_x/best_teacher.pt`。

## 评估

```bash
python evaluate.py --checkpoint /path/to/fold_0/best_student.pt --fold 0
```

## 当前实现范围

已经实现：

- simple surface encoder
- 3D volume encoder
- surface ROI pooling
- masked volume ROI pooling
- real white–pial multi-depth ribbon sampling（可选）
- hard Surface↔Ribbon cross-geometry anchoring（ribbon 不进入 classifier）
- true DKT mesh boundary / bilateral homology topology assets
- DiseaseAwareROIGraph with soft anatomy priors
- ROI-CLIP alignment
- disease-aware prototype bank
- ROI relation consistency
- teacher training
- topology-seeded strict MRI-only Graph Student
- logit + graph-enhanced ROI-state distillation（development-locked S4）
- ROI attention 可视化脚本

开发集 progressive ablation、Med3D adapter 以及 ROI bottleneck 消融均已完成。最新的
B0–B4 配对重训表明，Student 的 ROI 分类残差会削弱最终分类；因此修订后的严格 MRI-only
主线是 **B1：global MRI classifier + locked G3 Teacher logit KD**，关闭 Student ROI
classification residual 和 Student ROI graph。Cross-geometry、DKT graph 与 ROI-state
supervision 仍保留在 privileged Teacher、表征分析和消融证据中，但不直接进入最终 Student
分类路径。所有选择均未读取 held-out test。详细证据见：

- `outputs/vnext_roi_bottleneck_ablation_20261021/ROI_BOTTLENECK_ABLATION_REPORT.md`
- `outputs/vnext_graph_student_20261019/GRAPH_STUDENT_COMPONENT_ABLATION_REPORT.md`
- `docs/CortexBridge_vNext_Journal_Method_Status_and_Experiment_Roadmap.md`

## 统一 MRI–clinical 主线（development-locked）

当部署端允许同时输入 MRI 与临床量表时，当前性能主线为 **E3**：Teacher 和 Student
均使用 MRI 与同一组 29 个临床变量（不含 MMSE），Student 保留 global classifier，关闭
ROI/graph classification residual，并接受 temperature-scaled logit KD。临床缺失值只使用
训练折统计量拟合的填补器处理，同时保留 missingness indicator；受试者划分、预处理统计量
和模型选择均严格限制在 development folds。

完整的 privileged **E4** Teacher 在训练期额外使用 surface、ribbon cross-geometry
correspondence 与 DKT graph。其 Logit + ROI-state 两层蒸馏版本 **D1** 已实现并完成三折实验，
但目前没有显著优于 E3，因此 D1 作为机制消融保留，不能宣称为性能最优配置。E3 的三折
development 结果为 MCC `0.69535`、AUC `0.93203`、ACC `0.80708`；D1 为 MCC
`0.69413`、AUC `0.93082`、ACC `0.80706`。held-out test 尚未读取。

相关可复现实验入口与小型聚合结果：

- `run_unified_clinical_students.sh`
- `run_clinical_two_level_kd.sh`
- `outputs/vnext_unified_clinical_20261023/UNIFIED_CLINICAL_COMPARISON_REPORT.md`
- `outputs/vnext_clinical_two_level_kd_20261027/CLINICAL_TWO_LEVEL_KD_REPORT.md`

仓库只版本化聚合报告、配置和分析表；checkpoint、逐受试者预测、训练日志及本地 cache
均不提交。

当前没有做成强依赖：

- 真正的 spherical CNN
- multi-scale ribbon sampling（Phase 1 当前只使用最后一级 ROI feature map）
- cortex-aware heterogeneous convolution
- topology-soft correspondence target
- PET branch 或 ROI-wise cognitive modulation

如果你后面已经有更精确的 ROI bundle 生成代码，可以直接接到 `data/dataset.py` 里的 `roi_bundle.npz` 读取接口上，不需要改 teacher / student 主干。
