import copy

import pytest

from summarize_graph_stage_a import align_predictions, assess_gate


def _metric(values):
    return {
        "mean": sum(values) / len(values),
        "sample_sd": 0.0,
        "values": list(values),
    }


def _variant(mcc_values, *, learned=True):
    return {
        "mcc": _metric(mcc_values),
        "graph_attention_uniform_kl": _metric([0.02 if learned else 0.0] * 3),
        "graph_identity_mass": _metric([0.04] * 3),
        "graph_boundary_correlation": _metric([0.3] * 3),
        "graph_remote_mass": _metric([0.5] * 3),
        "graph_output_delta_l2": _metric([0.2] * 3),
    }


def test_gate_selects_best_topology_variant_only_after_majority_improvement():
    summaries = {
        "g0": _variant([0.50, 0.52, 0.48], learned=False),
        "g1": _variant([0.51, 0.51, 0.48]),
        "g2": _variant([0.53, 0.54, 0.47]),
        "g3": _variant([0.52, 0.55, 0.49]),
    }
    gate = assess_gate(summaries)
    assert gate["graph_teacher_gate_passed"] is True
    assert gate["selected_topology_variant"] == "g3"
    assert gate["student_stage_authorized"] is True


def test_gate_rejects_uniform_graph_even_when_classification_improves():
    summaries = {
        "g0": _variant([0.50, 0.52, 0.48], learned=False),
        "g1": _variant([0.51, 0.53, 0.49], learned=False),
        "g2": _variant([0.53, 0.54, 0.50], learned=False),
        "g3": _variant([0.54, 0.55, 0.51], learned=False),
    }
    gate = assess_gate(summaries)
    assert gate["graph_teacher_gate_passed"] is False
    assert gate["selected_topology_variant"] is None
    assert gate["student_stage_authorized"] is False


def test_align_predictions_requires_identical_subject_keys_and_labels():
    record = {
        "filename": "sub-ADNI001_ses-M00",
        "subject": "sub-ADNI001",
        "label": 1,
        "prediction": 1,
        "probabilities": [0.1, 0.8, 0.1],
    }
    predictions = {variant: [copy.deepcopy(record)] for variant in ("g0", "g1", "g2", "g3")}
    aligned = align_predictions(predictions)
    assert set(aligned["g3"]) == {record["filename"]}

    predictions["g2"][0]["label"] = 2
    with pytest.raises(RuntimeError, match="label mismatch"):
        align_predictions(predictions)
