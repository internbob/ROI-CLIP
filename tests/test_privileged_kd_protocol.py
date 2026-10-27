from config import load_configs


BASE_TEACHER = ["configs/data.yaml", "configs/teacher.yaml"]
BASE_STUDENT = [
    "configs/data.yaml",
    "configs/student.yaml",
    "configs/runtime_vnext_graph_student_locked_s4.yaml",
    "configs/runtime_roi_bottleneck_b1_global_logit_kd.yaml",
]
SAFE_FEATURES = (
    "age",
    "education",
    "apoe",
    "gds",
    "npiq_DEL",
    "npiq_HALL",
    "npiq_AGIT",
    "npiq_DEPD",
    "npiq_ANX",
    "npiq_ELAT",
    "npiq_APA",
    "npiq_DISN",
    "npiq_IRR",
    "npiq_MOT",
    "npiq_NITE",
    "npiq_APP",
)


def _features(cfg):
    return tuple(str(value) for value in cfg.data.clinical.features)


def test_p2_and_p3_use_identical_clean_clinical_features():
    p2 = load_configs(BASE_TEACHER + ["configs/runtime_privileged_p2_clinical_teacher.yaml"])
    p3 = load_configs(BASE_TEACHER + ["configs/runtime_privileged_p3_joint_teacher.yaml"])
    assert _features(p2) == _features(p3) == SAFE_FEATURES
    forbidden = ("mmse", "cdr", "adas", "faq", "ravlt", "lm_", "animal")
    assert not any(token in feature.lower() for feature in SAFE_FEATURES for token in forbidden)
    assert p2.data.clinical.enabled is p3.data.clinical.enabled is True
    assert p2.runtime.evaluate_test is p3.runtime.evaluate_test is False


def test_p2_is_clinical_privileged_without_geometry():
    cfg = load_configs(BASE_TEACHER + ["configs/runtime_privileged_p2_clinical_teacher.yaml"])
    assert cfg.model.use_clinical is True
    assert cfg.model.use_roi_branch is False
    assert cfg.model.cross_geometry.enabled is False
    assert cfg.model.roi_graph.enabled is False


def test_p3_is_joint_clinical_and_geometry_privileged():
    cfg = load_configs(
        BASE_TEACHER
        + [
            "configs/runtime_vnext_graph_g3_boundary_homology.yaml",
            "configs/runtime_privileged_p3_joint_teacher.yaml",
        ]
    )
    assert cfg.model.use_clinical is True
    assert cfg.model.use_roi_branch is True
    assert cfg.model.cross_geometry.enabled is True
    assert cfg.model.cross_geometry.lambda_cg == 0.01
    assert cfg.model.roi_graph.enabled is True
    assert cfg.model.roi_graph.use_boundary_strength is True
    assert cfg.model.roi_graph.use_homology is True


def test_p2_and_p3_students_keep_the_b1_mri_only_contract():
    for name in ("p2", "p3"):
        cfg = load_configs(BASE_STUDENT + [f"configs/runtime_privileged_{name}_b1_student.yaml"])
        assert cfg.model.use_clinical is True
        assert cfg.model.student_use_clinical is False
        assert cfg.model.student_roi_classifier.enabled is False
        assert cfg.model.student_roi_graph.enabled is False
        assert cfg.loss_weights.vnext_kd_logit == 0.5
        assert cfg.loss_weights.vnext_kd_roi == 0.0
        assert cfg.loss_weights.vnext_kd_graph == 0.0
        assert cfg.loss_weights.vnext_kd_prototype == 0.0
        assert cfg.training.initialize_from_teacher is False
        assert cfg.training.warmup_epochs == 0
        assert cfg.training.encoder_lr == 1e-4
        assert cfg.training.inherited_lr == 1e-4
        assert cfg.runtime.evaluate_test is False
        assert _features(cfg) == SAFE_FEATURES
