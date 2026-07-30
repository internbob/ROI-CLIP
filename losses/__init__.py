from .classification_loss import classification_loss
from .clip_loss import anatomical_transport_roi_clip_loss, distance_weighted_roi_clip_loss, roi_clip_loss
from .distill_loss import (
    attention_distill_loss,
    kd_logit_loss,
    multiscale_relation_distill_loss,
    prototype_distill_loss,
    relation_distill_loss,
    token_distill_loss,
)
from .prototype_loss import prototype_loss
from .relation_loss import relation_consistency_loss

__all__ = [
    "classification_loss",
    "roi_clip_loss",
    "anatomical_transport_roi_clip_loss",
    "distance_weighted_roi_clip_loss",
    "prototype_loss",
    "relation_consistency_loss",
    "kd_logit_loss",
    "token_distill_loss",
    "attention_distill_loss",
    "relation_distill_loss",
    "multiscale_relation_distill_loss",
    "prototype_distill_loss",
]
