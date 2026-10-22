from types import SimpleNamespace

import data.dataset as dataset_module


class _FakeDataset:
    paths = []

    def __init__(self, split_csv, **kwargs):
        del kwargs
        self.paths.append(split_csv)
        self.clinical_stats = {}
        self.pseudo_atlas_definition = None
        self.num_rois = 4
        self.clinical_input_dim = 0

    def __len__(self):
        return 8

    def __getitem__(self, index):
        return {"index": index}


def _cfg(evaluate_test):
    return SimpleNamespace(
        runtime=SimpleNamespace(evaluate_test=evaluate_test),
        data=SimpleNamespace(
            split_root="/tmp/strict-development-splits",
            clinical=SimpleNamespace(enabled=False),
            surface_root="/tmp/surface",
            task="COG",
            label_csv="",
            num_vertices=84,
            mri_shape=[32, 32, 32],
            normalize_surface=True,
            allowed_subject_prefixes=["sub-ADNI"],
            roi_bundle_name="roi_bundle.npz",
            preprocess_cache_root="",
            roi_cache_shape=[4, 4, 4],
            allow_unaligned_control=False,
            atlas=SimpleNamespace(enabled=True, name="DKTatlas"),
            mri_aug=SimpleNamespace(
                prob=0.0,
                max_voxel_shift=0,
                intensity_scale=0.0,
                intensity_shift=0.0,
                gamma=0.0,
                noise_std=0.0,
            ),
            ribbon_cache_root="",
        ),
        model=SimpleNamespace(
            num_rois=4,
            volume_pooling=SimpleNamespace(mode="legacy_mask"),
            cross_geometry=SimpleNamespace(enabled=False),
            ribbon=SimpleNamespace(num_depths=5, depths=[0.2, 0.35, 0.5, 0.65, 0.8]),
        ),
        training=SimpleNamespace(seed=3407, num_workers=0, batch_size=2),
    )


def test_development_mode_does_not_construct_test_dataset(monkeypatch):
    _FakeDataset.paths = []
    monkeypatch.setattr(dataset_module, "CortexWeaveROIClipDataset", _FakeDataset)
    _, _, test_loader = dataset_module.build_dataloaders(
        _cfg(evaluate_test=False), fold=0, require_teacher_inputs=True
    )
    assert test_loader is None
    assert all(not path.endswith("test.csv") for path in _FakeDataset.paths)
    assert [path.rsplit("/", 1)[-1] for path in _FakeDataset.paths] == ["train.csv", "valid.csv"]


def test_test_dataset_is_only_constructed_when_explicitly_enabled(monkeypatch):
    _FakeDataset.paths = []
    monkeypatch.setattr(dataset_module, "CortexWeaveROIClipDataset", _FakeDataset)
    _, _, test_loader = dataset_module.build_dataloaders(
        _cfg(evaluate_test=True), fold=0, require_teacher_inputs=True
    )
    assert test_loader is not None
    assert [path.rsplit("/", 1)[-1] for path in _FakeDataset.paths] == [
        "train.csv",
        "valid.csv",
        "test.csv",
    ]
