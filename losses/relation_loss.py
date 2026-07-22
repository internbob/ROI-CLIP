import torch.nn.functional as F


def relation_consistency_loss(surface_relation, volume_relation):
    return F.mse_loss(surface_relation, volume_relation)

