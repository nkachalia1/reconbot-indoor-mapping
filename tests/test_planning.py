from __future__ import annotations

import pytest

from reconmap.planning import MappingProfile, build_mapping_plan, evaluate_sparse_run


def test_hallway_room_plan_scales_keyframes_and_dense_segments():
    plan = build_mapping_plan("hallway_001", 180.0)

    assert plan.target_keyframes == 450
    assert plan.dense_segments == 2
    assert "place_recognition_loop_closure" in plan.stages
    assert plan.stages.index("trajectory_quality_gate") < plan.stages.index(
        "segmented_dense_reconstruction"
    )


def test_plan_respects_keyframe_limits():
    profile = MappingProfile(min_keyframes=100, max_keyframes=300)
    assert build_mapping_plan("short", 10, profile).target_keyframes == 100
    assert build_mapping_plan("long", 1000, profile).target_keyframes == 300


def test_sparse_quality_gate_accepts_good_closed_loop():
    accepted, failures = evaluate_sparse_run(
        selected_keyframes=450,
        registered_images=430,
        mean_reprojection_error_px=0.9,
        loop_closures=2,
        loop_drift_ratio=0.012,
    )

    assert accepted is True
    assert failures == ()


def test_sparse_quality_gate_reports_each_failure():
    accepted, failures = evaluate_sparse_run(
        selected_keyframes=100,
        registered_images=70,
        mean_reprojection_error_px=2.1,
        loop_closures=0,
        loop_drift_ratio=0.08,
    )

    assert accepted is False
    assert len(failures) == 4
    assert any("registration" in failure for failure in failures)


def test_invalid_profile_is_rejected():
    with pytest.raises(ValueError):
        MappingProfile(min_registration_ratio=0)
