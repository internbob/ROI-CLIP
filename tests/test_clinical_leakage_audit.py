import csv
from pathlib import Path

import numpy as np

from audit_clinical_leakage import (
    Row,
    feature_matrix,
    read_rows,
    subject_from_filename,
)


def test_subject_from_filename_removes_session():
    assert subject_from_filename("sub-ADNI001S0001_ses-M00") == "sub-ADNI001S0001"


def test_feature_matrix_encodes_gender_and_missing_values():
    rows = [
        Row(0, "train", "a", "a", 0, {"age": "70", "gender": "female"}),
        Row(0, "train", "b", "b", 1, {"age": "", "gender": "male"}),
    ]
    matrix = feature_matrix(rows, ("age", "gender"))
    assert matrix.shape == (2, 2)
    assert matrix[0, 1] == 0.0
    assert matrix[1, 1] == 1.0
    assert np.isnan(matrix[1, 0])


def test_reader_rejects_test_split(tmp_path: Path):
    path = tmp_path / "test.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["filename", "COG"])
        writer.writeheader()
        writer.writerow({"filename": "sub-ADNI001S0001_ses-M00", "COG": "0"})
    try:
        read_rows(path, fold=0, split="test")
    except RuntimeError as exc:
        assert "never read a test split" in str(exc)
    else:
        raise AssertionError("test split must be rejected")
