import json
import os
import random
import time

import numpy as np
import torch


def mkdir(path):
    os.makedirs(path, exist_ok=True)


def mkdirs(paths):
    if isinstance(paths, (list, tuple)):
        for path in paths:
            mkdir(path)
        return
    mkdir(paths)


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    os.environ["PYTHONHASHSEED"] = str(seed)


def count_parameters(module):
    return sum(param.numel() for param in module.parameters())


def now():
    return time.perf_counter()


def format_duration(seconds):
    seconds = float(seconds)
    if seconds < 60.0:
        return f"{seconds:.2f}s"
    minutes, rem = divmod(seconds, 60.0)
    if minutes < 60.0:
        return f"{int(minutes)}m{rem:05.2f}s"
    hours, minutes = divmod(int(minutes), 60)
    return f"{hours}h{minutes:02d}m{rem:05.2f}s"


def summarize_metric_results(results, metric_keys=None):
    metric_keys = list(metric_keys or ["acc", "auc", "mcc", "f1", "sen", "spe"])
    summary = {}
    for key in metric_keys:
        values = np.asarray([float(item.get(key, 0.0)) for item in results], dtype=np.float64)
        summary[key] = {
            "mean": float(values.mean()) if values.size else 0.0,
            "std": float(values.std()) if values.size else 0.0,
            "values": values.tolist(),
        }
    return summary


def format_metric_summary(summary, digits=4):
    parts = []
    for key in ["acc", "auc", "mcc", "f1", "sen", "spe"]:
        if key not in summary:
            continue
        item = summary[key]
        parts.append(f"{key}={float(item['mean']):.{digits}f}+/-{float(item['std']):.{digits}f}")
    return " ".join(parts)


class Logger:
    def __init__(self, log_dir, log_name="log.txt"):
        mkdir(log_dir)
        self.log_path = os.path.join(log_dir, log_name)
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(f"================ {self.log_path} ================\n")

    def print_message(self, msg):
        print(msg, flush=True)
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(f"{msg}\n")

    def save_json(self, payload, name):
        path = os.path.join(os.path.dirname(self.log_path), name)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
        return path


class CheckpointIO:
    def __init__(self, path):
        self.path = path

    def save(self, payload):
        torch.save(payload, self.path)

    def load(self, map_location="cpu"):
        if not os.path.exists(self.path):
            raise FileNotFoundError(self.path)
        return torch.load(self.path, map_location=map_location)


def move_to_device(batch, device):
    if torch.is_tensor(batch):
        return batch.to(device)
    if isinstance(batch, dict):
        return {k: move_to_device(v, device) for k, v in batch.items()}
    if isinstance(batch, list):
        return [move_to_device(v, device) for v in batch]
    if isinstance(batch, tuple):
        return tuple(move_to_device(v, device) for v in batch)
    return batch


def detach_to_cpu(value):
    if torch.is_tensor(value):
        return value.detach().cpu()
    if isinstance(value, dict):
        return {k: detach_to_cpu(v) for k, v in value.items()}
    if isinstance(value, list):
        return [detach_to_cpu(v) for v in value]
    if isinstance(value, tuple):
        return tuple(detach_to_cpu(v) for v in value)
    return value


def save_component_weights(model, save_dir):
    component_map = {
        "teacher_surface_encoder.pth": getattr(model, "surface_encoder", None),
        "teacher_volume_encoder.pth": getattr(model, "volume_encoder", None),
        "teacher_projection_heads.pth": getattr(model, "roi_clip", None),
        "teacher_classifier.pth": getattr(model, "classifier", None),
        "teacher_global_head.pth": getattr(model, "global_head", None),
        "teacher_clinical_fusion.pth": getattr(model, "clinical_fusion", None),
        "teacher_global_to_cross.pth": getattr(model, "global_to_cross", None),
        "teacher_cross_global_head.pth": getattr(model, "cross_global_head", None),
        "teacher_clinical_cross_fusion.pth": getattr(model, "clinical_cross_fusion", None),
        "teacher_roi_graph.pth": getattr(model, "roi_graph", None),
        "prototypes.pth": getattr(model, "prototype_bank", None),
    }
    saved = {}
    for filename, module in component_map.items():
        if module is None:
            continue
        path = os.path.join(save_dir, filename)
        torch.save(module.state_dict(), path)
        saved[filename] = path
    return saved
