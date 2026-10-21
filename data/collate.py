import torch


def roiclip_collate_fn(batch):
    tensor_keys = [
        "image",
        "label",
        "clinical_data",
        "surface_data",
        "surface_roi_masks",
        "local_ribbon_masks",
        "roi_volume_masks",
        "global_brain_mask",
        "affine",
        "has_teacher_inputs",
        "ribbon_grid",
        "ribbon_valid_mask",
        "vertex_valid_mask",
        "vertex_roi_index",
    ]
    out = {}
    for key in tensor_keys:
        out[key] = torch.stack([item[key] for item in batch], dim=0)

    for key in ["subject_id", "filename", "mri_path"]:
        out[key] = [item[key] for item in batch]
    return out
