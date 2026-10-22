import json

import pytest

from run_graph_randomization import selected_variant


def _summary(tmp_path, gate):
    path = tmp_path / "summary.json"
    path.write_text(json.dumps({"gate": gate}), encoding="utf-8")
    return path


def test_randomization_gate_accepts_only_selected_topology_variant(tmp_path):
    path = _summary(
        tmp_path,
        {"graph_teacher_gate_passed": True, "selected_topology_variant": "g3"},
    )
    assert selected_variant(path) == "g3"


def test_randomization_gate_rejects_failed_stage_a(tmp_path):
    path = _summary(
        tmp_path,
        {"graph_teacher_gate_passed": False, "selected_topology_variant": None},
    )
    with pytest.raises(RuntimeError, match="did not pass"):
        selected_variant(path)


def test_randomization_gate_rejects_unrecognized_variant(tmp_path):
    path = _summary(
        tmp_path,
        {"graph_teacher_gate_passed": True, "selected_topology_variant": "g4"},
    )
    with pytest.raises(RuntimeError, match="Unsupported"):
        selected_variant(path)
