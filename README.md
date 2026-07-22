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

## 训练

Teacher:

```bash
cd /home_data/home/zhouwl2023/project/Cortexbridge/cortexweave_roiclip
python train_teacher.py
```

Student:

```bash
python train_student.py --checkpoint /path/to/teacher_experiment_dir
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
- ROI-CLIP alignment
- disease-aware prototype bank
- ROI relation consistency
- teacher training
- MRI-only student distillation
- ROI attention 可视化脚本

当前没有做成强依赖：

- 真正的 spherical CNN
- white / pial surface 精确体素化
- cortex-aware heterogeneous convolution

如果你后面已经有更精确的 ROI bundle 生成代码，可以直接接到 `data/dataset.py` 里的 `roi_bundle.npz` 读取接口上，不需要改 teacher / student 主干。
