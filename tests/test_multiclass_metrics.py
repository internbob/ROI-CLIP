import numpy as np

from metrics import compute_classification_metrics


def test_perfect_three_class_metrics():
    labels = [0, 0, 1, 1, 2, 2]
    probs = np.eye(3, dtype=np.float32)[labels]
    metrics = compute_classification_metrics(labels, probs)

    for key in ["acc", "auc", "mcc", "f1", "sen", "spe"]:
        assert abs(metrics[key] - 1.0) < 1e-6
    assert metrics["confusion_matrix"] == [[2, 0, 0], [0, 2, 0], [0, 0, 2]]

