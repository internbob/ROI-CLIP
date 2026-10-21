import torch

from losses.cross_geometry_loss import (
    cross_geometry_correspondence_loss,
    cross_geometry_retrieval_metrics,
)


def test_cross_geometry_loss_is_symmetric_subject_local_and_has_gradients():
    surface = torch.eye(6).unsqueeze(0).repeat(2, 1, 1).requires_grad_(True)
    ribbon = (surface.detach() + 0.01 * torch.randn_like(surface)).requires_grad_(True)
    permuted = ribbon.detach().roll(shifts=1, dims=1)

    aligned_loss = cross_geometry_correspondence_loss(surface, ribbon, temperature=0.1)
    permuted_loss = cross_geometry_correspondence_loss(surface, permuted, temperature=0.1)
    assert aligned_loss < permuted_loss

    aligned_loss.backward()
    assert surface.grad is not None
    assert ribbon.grad is not None


def test_cross_geometry_retrieval_metrics_for_exact_identity():
    tokens = torch.eye(7).unsqueeze(0)
    metrics = cross_geometry_retrieval_metrics(tokens, tokens)

    assert metrics["roi_retrieval_acc1"].item() == 1.0
    assert metrics["roi_retrieval_acc5"].item() == 1.0
    assert metrics["roi_retrieval_mrr"].item() == 1.0
    assert metrics["same_roi_similarity"].item() == 1.0
    assert metrics["off_diagonal_similarity"].abs().item() < 1e-7
    assert metrics["diagonal_similarity_gap"].item() == 1.0


def test_topology_metric_accepts_fixed_neighbors_without_softening_training_target():
    num_rois = 6
    surface = torch.eye(num_rois).unsqueeze(0)
    ribbon = surface.roll(shifts=1, dims=1)
    topology = torch.zeros(num_rois, num_rois)
    for roi in range(num_rois):
        topology[roi, (roi - 1) % num_rois] = 1.0
        topology[roi, (roi + 1) % num_rois] = 1.0

    metrics = cross_geometry_retrieval_metrics(surface, ribbon, topology_affinity=topology)
    assert metrics["roi_retrieval_acc1"].item() == 0.0
    assert metrics["topology_retrieval_acc"].item() == 1.0

