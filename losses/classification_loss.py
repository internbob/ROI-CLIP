import torch
import torch.nn.functional as F


def classification_loss(logits, labels, class_weights=None):
    weight = None
    if class_weights is not None:
        weight = torch.as_tensor(class_weights, dtype=torch.float32, device=logits.device)
    return F.cross_entropy(logits, labels, weight=weight)

