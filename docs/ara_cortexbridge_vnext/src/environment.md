# Environment

- **Python**: 3.11.5
- **Framework**: PyTorch 2.5.1+cu124
- **Hardware**: NVIDIA A100 80 GB PCIe for completed vNext training/audit; one GPU per run
- **Key dependencies**:
  - NumPy 1.26.2
  - SciPy 1.11.3
  - NiBabel 5.1.0
  - scikit-learn 1.3.2
  - PyYAML 6.0.1
  - FreeSurfer installation under `/home_data/home/zhouwl2023/softwares/freesurfer`
- **Random seeds**: 3407 for main training; deterministic audit permutations use fold-specific fixed seeds
- **Execution routing**: repository `run-on-gpu-node.sh`, with completed Phase 2.1.5 work on `bme_gpu12`
- **Observed peak memory**: approximately 17.31 GiB for P2-B Teacher
- **Known environment limitation**: compute-node FreeSurfer executables require unavailable system library versions; Python/NiBabel registered-sphere annotation transfer was used for the audited fsaverage topology.

