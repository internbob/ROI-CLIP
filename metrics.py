import numpy as np

try:
    from sklearn.metrics import (
        accuracy_score,
        balanced_accuracy_score,
        confusion_matrix,
        f1_score,
        matthews_corrcoef,
        precision_score,
        recall_score,
        roc_auc_score,
    )
except ImportError:  # pragma: no cover
    accuracy_score = None
    balanced_accuracy_score = None
    confusion_matrix = None
    f1_score = None
    matthews_corrcoef = None
    precision_score = None
    recall_score = None
    roc_auc_score = None


class AverageMeter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.value = 0.0
        self.sum = 0.0
        self.count = 0

    def update(self, value, n=1):
        self.value = float(value)
        self.sum += float(value) * int(n)
        self.count += int(n)

    @property
    def avg(self):
        if self.count <= 0:
            return 0.0
        return self.sum / self.count


def _numpy_confusion_matrix(labels, preds, num_classes):
    matrix = np.zeros((num_classes, num_classes), dtype=np.int64)
    for true_label, pred_label in zip(labels, preds):
        if 0 <= true_label < num_classes and 0 <= pred_label < num_classes:
            matrix[true_label, pred_label] += 1
    return matrix


def _mcc_from_confusion_matrix(matrix):
    matrix = np.asarray(matrix, dtype=np.float64)
    total = float(matrix.sum())
    correct = float(np.trace(matrix))
    true_totals = matrix.sum(axis=1)
    pred_totals = matrix.sum(axis=0)
    numerator = correct * total - float(np.dot(true_totals, pred_totals))
    denominator = np.sqrt(
        max(0.0, total * total - float(np.dot(pred_totals, pred_totals)))
        * max(0.0, total * total - float(np.dot(true_totals, true_totals)))
    )
    return 0.0 if denominator <= 0.0 else numerator / denominator


def compute_classification_metrics(labels, probs):
    labels = np.asarray(labels, dtype=np.int64)
    probs = np.asarray(probs, dtype=np.float32)
    preds = probs.argmax(axis=1)
    num_classes = int(probs.shape[1])

    if accuracy_score is None:
        acc = float((preds == labels).mean()) if labels.size else 0.0
        bacc = acc
        f1_macro = acc
        precision_macro = acc
        recall_macro = acc
        auc = 0.0
    else:
        acc = float(accuracy_score(labels, preds))
        bacc = float(balanced_accuracy_score(labels, preds))
        f1_macro = float(f1_score(labels, preds, average="macro", zero_division=0))
        precision_macro = float(precision_score(labels, preds, average="macro", zero_division=0))
        recall_macro = float(recall_score(labels, preds, average="macro", zero_division=0))
        auc = 0.0
        try:
            if num_classes == 2:
                auc = float(roc_auc_score(labels, probs[:, 1]))
            else:
                auc = float(roc_auc_score(labels, probs, multi_class="ovr", average="macro"))
        except Exception:
            auc = 0.0

    metrics = {
        "acc": acc,
        "balanced_acc": bacc,
        "f1_macro": f1_macro,
        "precision_macro": precision_macro,
        "recall_macro": recall_macro,
        "auc": auc,
    }

    if num_classes == 2:
        tp = int(np.sum((labels == 1) & (preds == 1)))
        tn = int(np.sum((labels == 0) & (preds == 0)))
        fp = int(np.sum((labels == 0) & (preds == 1)))
        fn = int(np.sum((labels == 1) & (preds == 0)))
        sen = tp / max(1, tp + fn)
        spe = tn / max(1, tn + fp)
        precision_pos = tp / max(1, tp + fp)
        f1_pos = 0.0 if precision_pos + sen <= 0.0 else 2.0 * precision_pos * sen / (precision_pos + sen)
        denom = max(1e-8, float((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)) ** 0.5)
        mcc = ((tp * tn) - (fp * fn)) / denom
        metrics["f1"] = float(f1_pos)
        metrics["sen"] = float(sen)
        metrics["spe"] = float(spe)
        metrics["mcc"] = float(mcc)
        metrics["tp"] = tp
        metrics["tn"] = tn
        metrics["fp"] = fp
        metrics["fn"] = fn
    else:
        metrics["f1"] = float(f1_macro)
        metrics["sen"] = float(recall_macro)
        if confusion_matrix is None:
            matrix = _numpy_confusion_matrix(labels, preds, num_classes)
        else:
            matrix = confusion_matrix(labels, preds, labels=list(range(num_classes)))
        specificities = []
        total = int(matrix.sum())
        for class_idx in range(num_classes):
            fp = int(matrix[:, class_idx].sum() - matrix[class_idx, class_idx])
            fn = int(matrix[class_idx, :].sum() - matrix[class_idx, class_idx])
            tp = int(matrix[class_idx, class_idx])
            tn = total - tp - fp - fn
            specificities.append(tn / max(1, tn + fp))
        metrics["spe"] = float(np.mean(specificities))
        metrics["mcc"] = (
            float(matthews_corrcoef(labels, preds))
            if matthews_corrcoef is not None
            else float(_mcc_from_confusion_matrix(matrix))
        )
        metrics["confusion_matrix"] = matrix.tolist()

    return metrics


def format_classification_metrics(metrics, prefix=""):
    keys = ("acc", "auc", "mcc", "f1", "sen", "spe")
    return " ".join(f"{prefix}{key}={float(metrics.get(key, 0.0)):.4f}" for key in keys)
