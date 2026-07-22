import torch.nn.functional as F


def prototype_loss(prototype_logits, labels):
    bsz, num_rois, num_classes = prototype_logits.shape
    targets = labels.unsqueeze(1).expand(-1, num_rois).reshape(-1)
    logits = prototype_logits.reshape(bsz * num_rois, num_classes)
    return F.cross_entropy(logits, targets)

