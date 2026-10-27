#!/usr/bin/env bash
set -euo pipefail

python_bin="${PYTHON_BIN:?PYTHON_BIN must be provided by run-on-gpu-node.sh}"
stage="${1:?usage: $0 d0_full|lambda_sweep|d1_remaining|d2_full|d3_fold <fold>}"
output_root="outputs/vnext_clinical_two_level_kd_20261027"
common=(
  configs/data.yaml
  configs/student.yaml
  configs/runtime_unified_clinical_common.yaml
  configs/runtime_unified_student_e4_full_kd.yaml
  configs/runtime_clinical_two_level_student_graph.yaml
)
train_args=(
  --num_folds 3
  --epochs 30
  --early_stop_patience 4
)

case "${stage}" in
  d0_full)
    "${python_bin}" train_student.py \
      --config "${common[@]}" configs/runtime_clinical_two_level_d0_logit_only.yaml \
      "${train_args[@]}" --output_dir "${output_root}/d0_logit_only"
    ;;
  lambda_sweep)
    for suffix in 010 050 100; do
      "${python_bin}" train_student.py \
        --config "${common[@]}" "configs/runtime_clinical_two_level_d1_roi${suffix}.yaml" \
        "${train_args[@]}" --fold 0 \
        --output_dir "${output_root}/lambda_sweep_roi${suffix}"
    done
    ;;
  d1_remaining)
    for fold in 1 2; do
      "${python_bin}" train_student.py \
        --config "${common[@]}" configs/runtime_clinical_two_level_d1_roi100.yaml \
        "${train_args[@]}" --fold "${fold}" \
        --output_dir "${output_root}/lambda_sweep_roi100"
    done
    ;;
  d2_full)
    d2_common=(
      configs/data.yaml
      configs/student.yaml
      configs/runtime_unified_clinical_common.yaml
      configs/runtime_unified_student_e6_cg_off_kd.yaml
      configs/runtime_clinical_two_level_student_graph.yaml
    )
    "${python_bin}" train_student.py \
      --config "${d2_common[@]}" configs/runtime_clinical_two_level_d2_cg_off.yaml \
      "${train_args[@]}" --output_dir "${output_root}/d2_cg_off"
    ;;
  d3_fold)
    fold="${2:?d3_fold requires a fold index}"
    "${python_bin}" train_student.py \
      --config "${common[@]}" configs/runtime_clinical_two_level_d3_roi_shuffle.yaml \
      "${train_args[@]}" --fold "${fold}" --control_shuffle_roi \
      --output_dir "${output_root}/d3_roi_shuffle"
    ;;
  *)
    echo "Unknown stage: ${stage}" >&2
    exit 2
    ;;
esac
