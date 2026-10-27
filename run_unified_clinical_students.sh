#!/usr/bin/env bash
set -euo pipefail

python_bin="${PYTHON_BIN:?PYTHON_BIN must be provided by run-on-gpu-node.sh}"
output_root="outputs/vnext_unified_clinical_20261023"
common=(
  configs/data.yaml
  configs/student.yaml
  configs/runtime_unified_clinical_common.yaml
)
train_args=(
  --num_folds 3
  --epochs 30
  --early_stop_patience 4
)

"${python_bin}" train_student.py \
  --config "${common[@]}" configs/runtime_unified_student_e1_mri_ce.yaml \
  "${train_args[@]}" --output_dir "${output_root}/e1_student"

"${python_bin}" train_student.py \
  --config "${common[@]}" configs/runtime_unified_student_e2_mri_clinical_ce.yaml \
  "${train_args[@]}" --output_dir "${output_root}/e2_student"

"${python_bin}" train_student.py \
  --config "${common[@]}" configs/runtime_unified_student_e3_mri_clinical_kd.yaml \
  "${train_args[@]}" --disable_roi_branch --output_dir "${output_root}/e3_student"

"${python_bin}" train_student.py \
  --config "${common[@]}" configs/runtime_unified_student_e6_cg_off_kd.yaml \
  "${train_args[@]}" --output_dir "${output_root}/e6_student_cg_off"

"${python_bin}" train_student.py \
  --config "${common[@]}" configs/runtime_unified_student_e4_full_kd.yaml \
  "${train_args[@]}" --output_dir "${output_root}/e4_student"

"${python_bin}" train_student.py \
  --config "${common[@]}" configs/runtime_unified_student_e4_full_kd.yaml \
  configs/runtime_unified_student_e5_clinical_shuffle.yaml \
  "${train_args[@]}" --output_dir "${output_root}/e5_student_shuffle"
