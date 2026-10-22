import argparse
import copy
import os
from datetime import datetime

import yaml


class Config(dict):
    def __getattr__(self, item):
        if item not in self:
            raise AttributeError(item)
        value = self[item]
        if isinstance(value, dict) and not isinstance(value, Config):
            value = Config(value)
            self[item] = value
        return value

    def __setattr__(self, key, value):
        self[key] = value


def _deep_update(base, update):
    for key, value in update.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _deep_update(base[key], value)
        else:
            base[key] = value
    return base


def _to_config(value):
    if isinstance(value, dict):
        return Config({k: _to_config(v) for k, v in value.items()})
    if isinstance(value, list):
        return [_to_config(v) for v in value]
    return value


def load_configs(config_paths):
    merged = {}
    for path in config_paths:
        with open(path, "r", encoding="utf-8") as f:
            content = yaml.safe_load(f) or {}
        _deep_update(merged, content)
    return _to_config(merged)


def build_parser(default_paths, description):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--config", nargs="+", default=default_paths)
    parser.add_argument("--fold", type=int, default=None)
    parser.add_argument("--device", type=str, default="")
    parser.add_argument("--checkpoint", type=str, default="")
    parser.add_argument("--output_dir", type=str, default="")
    parser.add_argument("--dry_run", action="store_true")
    return parser


def parse_config(default_paths, description):
    parser = build_parser(default_paths, description)
    args = parser.parse_args()
    cfg = load_configs(args.config)
    if "runtime" not in cfg:
        cfg.runtime = Config()
    if args.fold is not None:
        cfg.runtime.fold = int(args.fold)
    if args.device:
        cfg.runtime.device = args.device
    if args.checkpoint:
        cfg.runtime.checkpoint = args.checkpoint
    if args.output_dir:
        cfg.runtime.output_dir = args.output_dir
    cfg.runtime.dry_run = bool(args.dry_run)
    cfg.runtime.config_paths = list(args.config)
    return cfg


def clone_config(cfg):
    return _to_config(copy.deepcopy(dict(cfg)))


def prepare_experiment_dir(cfg, stage_name):
    output_dir = str(getattr(cfg.runtime, "output_dir", "") or "")
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        return output_dir

    current_time = datetime.now().strftime("%Y_%m_%d_%H%M%S")
    checkpoints_dir = str(cfg.output.checkpoints_dir)
    task = str(cfg.data.task)
    method = str(cfg.model.method)
    name = str(getattr(cfg, "experiment_name", stage_name))
    expr_dir = os.path.join(checkpoints_dir, f"{task}_{name}_{current_time}_{method}")
    os.makedirs(expr_dir, exist_ok=True)
    return expr_dir
