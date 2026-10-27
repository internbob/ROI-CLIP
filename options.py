import argparse
import os
from datetime import datetime

from config import Config, load_configs
from utils import mkdirs


def _set_nested(cfg, dotted_key, value):
    if value is None:
        return
    parts = dotted_key.split(".")
    current = cfg
    for part in parts[:-1]:
        if part not in current or not isinstance(current[part], dict):
            current[part] = Config()
        elif not isinstance(current[part], Config):
            current[part] = Config(current[part])
        current = current[part]
    current[parts[-1]] = value


def _flatten(value, prefix=""):
    items = {}
    if isinstance(value, dict):
        for key in sorted(value.keys()):
            child_prefix = f"{prefix}.{key}" if prefix else str(key)
            items.update(_flatten(value[key], child_prefix))
        return items
    items[prefix] = value
    return items


class Option:
    def __init__(self, stage, default_paths):
        self.stage = stage
        self.default_paths = list(default_paths)
        self.parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        self.opt = None

    def initialize(self, parser):
        parser.add_argument("--config", nargs="+", default=self.default_paths)
        parser.add_argument("--fold", type=int, default=None)
        parser.add_argument("--device", type=str, default="")
        parser.add_argument("--checkpoint", type=str, default="")
        parser.add_argument("--output_dir", type=str, default="")
        parser.add_argument("--dry_run", action="store_true")

        parser.add_argument("--checkpoints_dir", type=str, default="./checkpoints_surface")
        parser.add_argument("--experiment_name", type=str, default="")
        parser.add_argument("--model", type=str, default="CortexWeave-ROIClip")
        parser.add_argument("--method", type=str, default="cortexweave-roiclip")

        parser.add_argument("--task", type=str, default="COG")
        parser.add_argument("--mode", type=str, default="train")
        parser.add_argument("--batch_size", type=int, default=8)
        parser.add_argument("--num_workers", type=int, default=8)
        parser.add_argument("--lr", type=float, default=1e-4)
        parser.add_argument("--weight_decay", type=float, default=1e-4 if self.stage == "teacher" else 1e-5)
        parser.add_argument("--early_stop_patience", type=int, default=4)
        parser.add_argument("--selection_metric", type=str, default="mcc", choices=["mcc", "f1", "acc"])
        parser.add_argument(
            "--class_weight_mode",
            type=str,
            default="sqrt_balanced",
            choices=["none", "balanced", "sqrt_balanced"],
        )
        parser.add_argument("--seed", type=int, default=3407)
        parser.add_argument("--num_folds", type=int, default=5)

        parser.add_argument("--surface_root", type=str, default="/public_bme/data/zhouwl2023/ADNI/BrainSurf_Results")
        parser.add_argument("--split_root", type=str, default="./lookupcsv-full-huashan-adni-teacher-clean")
        parser.add_argument("--label_csv", type=str, default="")
        parser.add_argument("--allowed_subject_prefixes", nargs="+", default=["sub-ADNI"])
        parser.add_argument("--num_vertices", type=int, default=5124)
        parser.add_argument("--num_rois", type=int, default=62)
        parser.add_argument("--mri_depth", type=int, default=160)
        parser.add_argument("--mri_height", type=int, default=192)
        parser.add_argument("--mri_width", type=int, default=160)
        parser.add_argument("--roi_bundle_name", type=str, default="roi_bundle.npz")
        parser.add_argument("--preprocess_cache_root", type=str, default="")
        parser.add_argument("--roi_cache_depth", type=int, default=20)
        parser.add_argument("--roi_cache_height", type=int, default=24)
        parser.add_argument("--roi_cache_width", type=int, default=20)
        parser.add_argument("--atlas_name", type=str, default="DKTatlas")

        parser.add_argument("--feature_dim", type=int, default=128)
        parser.add_argument("--proj_dim", type=int, default=128)
        parser.add_argument("--base_channels", type=int, default=32)
        parser.add_argument("--surface_in_channels", type=int, default=3)
        parser.add_argument("--surface_hidden_dim", type=int, default=128)
        parser.add_argument("--surface_refiner", type=str, default="none")
        parser.add_argument("--surface_refiner_hidden_dim", type=int, default=0)
        parser.add_argument("--surface_refiner_dropout", type=float, default=0.0)
        parser.add_argument("--surface_refiner_residual_init", type=float, default=0.1)
        parser.add_argument("--dropout", type=float, default=0.2)
        parser.add_argument("--num_classes", type=int, default=3)
        parser.add_argument("--roi_residual_init", type=float, default=0.1)
        parser.add_argument("--volume_norm", type=str, default="batch", choices=["batch", "group"])
        parser.add_argument("--use_clinical", type=int, choices=[0, 1], default=None)
        parser.add_argument("--clinical_fusion_mode", type=str, default="gated", choices=["gated", "cross", "atlas_prompt"])
        parser.add_argument("--clinical_hidden_dim", type=int, default=256)
        parser.add_argument("--clinical_dropout", type=float, default=0.1)
        parser.add_argument("--clinical_num_heads", type=int, default=4)
        parser.add_argument("--clinical_num_layers", type=int, default=2)
        parser.add_argument("--disable_roi_branch", action="store_true")

        if self.stage == "teacher":
            parser.add_argument("--epochs", type=int, default=30)
            parser.add_argument("--teacher_ce_warmup_epochs", type=int, default=5)
            parser.add_argument("--lambda_ce", type=float, default=1.0)
            parser.add_argument("--lambda_roi_clip", type=float, default=0.01)
            parser.add_argument("--lambda_prototype", type=float, default=0.01)
            parser.add_argument("--lambda_relation", type=float, default=0.005)
            parser.add_argument("--clip_temperature", type=float, default=0.1)
            parser.add_argument("--prototype_temperature", type=float, default=0.1)
            parser.add_argument(
                "--roi_clip_alignment_mode",
                type=str,
                default="hard",
                choices=["hard", "topology_soft", "distance_weighted"],
            )
            parser.add_argument("--roi_clip_soft_epsilon", type=float, default=0.1)
            parser.add_argument("--roi_clip_distance_sigma", type=float, default=0.0)
            parser.add_argument("--roi_clip_homology_weight", type=float, default=1.0)
            parser.add_argument("--roi_clip_affinity_topk", type=int, default=0)
            parser.add_argument("--roi_clip_min_negative_weight", type=float, default=0.05)
        else:
            parser.add_argument("--epochs", type=int, default=30)
            parser.add_argument("--teacher_checkpoint", type=str, default="")
            parser.add_argument("--lambda_ce", type=float, default=1.0)
            parser.add_argument("--lambda_kd_logit", type=float, default=0.25)
            parser.add_argument("--lambda_kd_token", type=float, default=0.1)
            parser.add_argument("--lambda_kd_attention", type=float, default=0.0)
            parser.add_argument("--lambda_kd_relation", type=float, default=0.01)
            parser.add_argument("--lambda_kd_prototype", type=float, default=0.01)
            parser.add_argument("--lambda_roi_attention_mass", type=float, default=0.0)
            parser.add_argument("--lambda_roi_occupancy", type=float, default=0.0)
            parser.add_argument("--lambda_roi_center", type=float, default=0.0)
            parser.add_argument(
                "--relation_distill_mode",
                type=str,
                default="global",
                choices=["global", "local_remote"],
            )
            parser.add_argument("--relation_local_topk", type=int, default=6)
            parser.add_argument("--relation_distance_sigma", type=float, default=0.0)
            parser.add_argument("--relation_homology_weight", type=float, default=1.0)
            parser.add_argument("--relation_local_weight", type=float, default=0.7)
            parser.add_argument("--relation_remote_weight", type=float, default=0.3)
            parser.add_argument("--kd_temperature", type=float, default=4.0)
            parser.add_argument("--prototype_temperature", type=float, default=0.1)
            # None means "keep the YAML value". Non-None parser defaults used
            # to silently replace experiment-specific Student schedules even
            # when the corresponding CLI flag was absent.
            parser.add_argument("--warmup_epochs", type=int, default=None)
            parser.add_argument("--encoder_lr", type=float, default=None)
            parser.add_argument("--inherited_lr", type=float, default=None)
            # None means "keep the YAML value".  A False argparse default
            # silently disabled config-declared negative controls.
            parser.add_argument("--control_random_teacher", action="store_true", default=None)
            parser.add_argument("--control_shuffle_roi", action="store_true", default=None)
        return parser

    def _apply_overrides(self, cfg, args):
        if "runtime" not in cfg:
            cfg.runtime = Config()

        runtime_pairs = {
            "fold": "runtime.fold",
            "device": "runtime.device",
            "checkpoint": "runtime.checkpoint",
            "dry_run": "runtime.dry_run",
        }
        for arg_name, key in runtime_pairs.items():
            value = getattr(args, arg_name)
            if arg_name == "dry_run":
                _set_nested(cfg, key, bool(value))
            elif value not in [None, ""]:
                _set_nested(cfg, key, value)

        mapping = {
            "checkpoints_dir": "output.checkpoints_dir",
            "experiment_name": "experiment_name",
            "model": "model.name",
            "method": "model.method",
            "task": "data.task",
            "mode": "mode",
            "batch_size": "training.batch_size",
            "num_workers": "training.num_workers",
            "lr": "training.lr",
            "weight_decay": "training.weight_decay",
            "early_stop_patience": "training.early_stop_patience",
            "selection_metric": "training.selection_metric",
            "class_weight_mode": "training.class_weight_mode",
            "seed": "training.seed",
            "num_folds": "training.num_folds",
            "surface_root": "data.surface_root",
            "split_root": "data.split_root",
            "label_csv": "data.label_csv",
            "allowed_subject_prefixes": "data.allowed_subject_prefixes",
            "num_vertices": "data.num_vertices",
            "num_rois": "model.num_rois",
            "roi_bundle_name": "data.roi_bundle_name",
            "preprocess_cache_root": "data.preprocess_cache_root",
            "atlas_name": "data.atlas.name",
            "feature_dim": "model.feature_dim",
            "proj_dim": "model.proj_dim",
            "base_channels": "model.base_channels",
            "surface_in_channels": "model.surface_in_channels",
            "surface_hidden_dim": "model.surface_hidden_dim",
            "surface_refiner": "model.surface_refiner",
            "surface_refiner_hidden_dim": "model.surface_refiner_hidden_dim",
            "surface_refiner_dropout": "model.surface_refiner_dropout",
            "surface_refiner_residual_init": "model.surface_refiner_residual_init",
            "dropout": "model.dropout",
            "num_classes": "model.num_classes",
            "roi_residual_init": "model.roi_residual_init",
            "volume_norm": "model.volume_norm",
            "clinical_fusion_mode": "model.clinical_fusion_mode",
            "clinical_hidden_dim": "model.clinical_hidden_dim",
            "clinical_dropout": "model.clinical_dropout",
            "clinical_num_heads": "model.clinical_num_heads",
            "clinical_num_layers": "model.clinical_num_layers",
            "epochs": "training.epochs",
            "prototype_temperature": "temperature.prototype",
        }
        for arg_name, key in mapping.items():
            value = getattr(args, arg_name, None)
            if value is not None and value != "":
                _set_nested(cfg, key, value)

        _set_nested(cfg, "data.mri_shape", [args.mri_depth, args.mri_height, args.mri_width])
        _set_nested(cfg, "model.use_roi_branch", not bool(args.disable_roi_branch))
        if args.use_clinical is not None:
            _set_nested(cfg, "model.use_clinical", bool(args.use_clinical))
            _set_nested(cfg, "data.clinical.enabled", bool(args.use_clinical))
        _set_nested(
            cfg,
            "data.roi_cache_shape",
            [args.roi_cache_depth, args.roi_cache_height, args.roi_cache_width],
        )

        if self.stage == "teacher":
            _set_nested(cfg, "training.ce_warmup_epochs", args.teacher_ce_warmup_epochs)
            _set_nested(cfg, "loss_weights.ce", args.lambda_ce)
            _set_nested(cfg, "loss_weights.roi_clip", args.lambda_roi_clip)
            _set_nested(cfg, "loss_weights.prototype", args.lambda_prototype)
            _set_nested(cfg, "loss_weights.relation", args.lambda_relation)
            _set_nested(cfg, "temperature.clip", args.clip_temperature)
            _set_nested(cfg, "training.roi_clip_alignment_mode", args.roi_clip_alignment_mode)
            _set_nested(cfg, "training.roi_clip_soft_epsilon", args.roi_clip_soft_epsilon)
            _set_nested(cfg, "training.roi_clip_distance_sigma", args.roi_clip_distance_sigma)
            _set_nested(cfg, "training.roi_clip_homology_weight", args.roi_clip_homology_weight)
            _set_nested(cfg, "training.roi_clip_affinity_topk", args.roi_clip_affinity_topk)
            _set_nested(cfg, "training.roi_clip_min_negative_weight", args.roi_clip_min_negative_weight)
        else:
            _set_nested(cfg, "training.teacher_checkpoint", args.teacher_checkpoint)
            _set_nested(cfg, "loss_weights.ce", args.lambda_ce)
            _set_nested(cfg, "loss_weights.kd_logit", args.lambda_kd_logit)
            _set_nested(cfg, "loss_weights.kd_token", args.lambda_kd_token)
            _set_nested(cfg, "loss_weights.kd_attention", args.lambda_kd_attention)
            _set_nested(cfg, "loss_weights.kd_relation", args.lambda_kd_relation)
            _set_nested(cfg, "loss_weights.kd_prototype", args.lambda_kd_prototype)
            _set_nested(cfg, "loss_weights.roi_attention_mass", args.lambda_roi_attention_mass)
            _set_nested(cfg, "loss_weights.roi_occupancy", args.lambda_roi_occupancy)
            _set_nested(cfg, "loss_weights.roi_center", args.lambda_roi_center)
            _set_nested(cfg, "training.relation_distill_mode", args.relation_distill_mode)
            _set_nested(cfg, "training.relation_local_topk", args.relation_local_topk)
            _set_nested(cfg, "training.relation_distance_sigma", args.relation_distance_sigma)
            _set_nested(cfg, "training.relation_homology_weight", args.relation_homology_weight)
            _set_nested(cfg, "training.relation_local_weight", args.relation_local_weight)
            _set_nested(cfg, "training.relation_remote_weight", args.relation_remote_weight)
            _set_nested(cfg, "temperature.kd", args.kd_temperature)
            if args.warmup_epochs is not None:
                _set_nested(cfg, "training.warmup_epochs", args.warmup_epochs)
            if args.encoder_lr is not None:
                _set_nested(cfg, "training.encoder_lr", args.encoder_lr)
            if args.inherited_lr is not None:
                _set_nested(cfg, "training.inherited_lr", args.inherited_lr)
            if args.control_random_teacher is not None:
                _set_nested(cfg, "training.control_random_teacher", bool(args.control_random_teacher))
            if args.control_shuffle_roi is not None:
                _set_nested(cfg, "training.control_shuffle_roi", bool(args.control_shuffle_roi))

        return cfg

    def _prepare_expr_dir(self, cfg, args):
        output_dir = str(args.output_dir or "")
        if output_dir:
            mkdirs(output_dir)
            return output_dir

        current_time = datetime.now().strftime("%Y_%m_%d_%H%M%S")
        checkpoints_dir = str(cfg.output.checkpoints_dir)
        task = str(cfg.data.task)
        model_name = str(cfg.model.name)
        method = str(cfg.model.method)
        expr_dir = os.path.join(checkpoints_dir, f"{task}_{model_name}_{current_time}_{method}")
        mkdirs(expr_dir)
        return expr_dir

    def print_options(self, cfg, expr_dir):
        flat = _flatten(cfg)
        message = ""
        message += "----------------- Options ---------------\n"
        for key, value in sorted(flat.items()):
            message += "{:>32}: {}\n".format(str(key), value)
        print(message)

        file_name = os.path.join(expr_dir, "opt.txt")
        with open(file_name, "wt", encoding="utf-8") as opt_file:
            opt_file.write(message)
            opt_file.write("\n")
        print("Create opt file opt.txt")

    def parse(self):
        self.parser = self.initialize(self.parser)
        args = self.parser.parse_args()
        cfg = load_configs(args.config)
        cfg = self._apply_overrides(cfg, args)
        expr_dir = self._prepare_expr_dir(cfg, args)
        cfg.expr_dir = expr_dir
        cfg.runtime.output_dir = expr_dir
        cfg.runtime.config_paths = list(args.config)
        self.print_options(cfg, expr_dir)
        self.opt = cfg
        return cfg
