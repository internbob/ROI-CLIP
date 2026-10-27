import pandas as pd
import pytest

from summarize_privileged_kd import metric_from_frame, paired_bootstrap, paired_frames


def _frame(predictions):
    labels = [0, 1, 2, 0, 1, 2]
    rows = []
    for index, (label, prediction) in enumerate(zip(labels, predictions)):
        scores = [0.05, 0.05, 0.05]
        scores[prediction] = 0.90
        rows.append(
            {
                "filename": f"sub-{index}",
                "label": label,
                "pred": prediction,
                "score_0": scores[0],
                "score_1": scores[1],
                "score_2": scores[2],
                "fold": 0,
                "key": f"0:sub-{index}",
            }
        )
    return pd.DataFrame(rows)


def test_metrics_include_mci_recall_and_macro_specificity():
    frame = _frame([0, 1, 2, 0, 0, 2])
    assert metric_from_frame(frame, "acc") == pytest.approx(5 / 6)
    assert metric_from_frame(frame, "mci_recall") == pytest.approx(0.5)
    assert 0.0 <= metric_from_frame(frame, "specificity_macro") <= 1.0
    assert 0.0 <= metric_from_frame(frame, "auc") <= 1.0


def test_pairing_requires_identical_subject_keys_and_labels():
    left = _frame([0, 1, 2, 0, 0, 2])
    right = _frame([0, 1, 2, 0, 1, 2])
    assert len(paired_frames(left, right)) == len(left)

    missing = right.iloc[:-1].copy()
    with pytest.raises(RuntimeError, match="identical"):
        paired_frames(left, missing)

    wrong_label = right.copy()
    wrong_label.loc[0, "label"] = 2
    with pytest.raises(RuntimeError, match="Labels differ"):
        paired_frames(left, wrong_label)


def test_paired_bootstrap_resamples_within_fold_and_averages_fold_deltas():
    left_fold0 = _frame([0, 0, 2, 0, 0, 2])
    right_fold0 = _frame([0, 1, 2, 0, 1, 2])
    left_fold1 = _frame([0, 1, 2, 0, 1, 2])
    right_fold1 = _frame([0, 1, 2, 0, 0, 2])
    for frame in (left_fold1, right_fold1):
        frame["fold"] = 1
        frame["filename"] = "fold1-" + frame["filename"]
        frame["key"] = "1:" + frame["filename"]

    result = paired_bootstrap(
        pd.concat([left_fold0, left_fold1], ignore_index=True),
        pd.concat([right_fold0, right_fold1], ignore_index=True),
        metric="acc",
        replicates=100,
        seed=7,
    )

    assert result["observed_fold_deltas"] == pytest.approx([2 / 6, -1 / 6])
    assert result["observed_delta_right_minus_left"] == pytest.approx(1 / 12)
    assert "within each fold" in result["resampling_unit"]
