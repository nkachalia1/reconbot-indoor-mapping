"""Environment-scale capture profiles, run plans, and sparse quality gates."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math


@dataclass(frozen=True)
class MappingProfile:
    name: str = "hallway_room"
    target_keyframe_hz: float = 2.5
    min_keyframes: int = 90
    max_keyframes: int = 600
    sequential_overlap: int = 25
    loop_anchor_window: int = 30
    dense_segment_duration_s: float = 90.0
    min_registration_ratio: float = 0.90
    max_reprojection_error_px: float = 1.50
    min_loop_closures: int = 1
    max_loop_drift_ratio: float = 0.02

    def __post_init__(self) -> None:
        if self.target_keyframe_hz <= 0:
            raise ValueError("target_keyframe_hz must be positive")
        if not 0 < self.min_registration_ratio <= 1:
            raise ValueError("min_registration_ratio must be in (0, 1]")
        if self.min_keyframes <= 0 or self.max_keyframes < self.min_keyframes:
            raise ValueError("keyframe limits are invalid")


@dataclass(frozen=True)
class MappingPlan:
    session: str
    profile: MappingProfile
    video_duration_s: float
    target_keyframes: int
    dense_segments: int
    stages: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "session": self.session,
            "profile": asdict(self.profile),
            "video_duration_s": self.video_duration_s,
            "target_keyframes": self.target_keyframes,
            "dense_segments": self.dense_segments,
            "stages": list(self.stages),
        }


def build_mapping_plan(
    session: str,
    video_duration_s: float,
    profile: MappingProfile | None = None,
) -> MappingPlan:
    if video_duration_s <= 0:
        raise ValueError("video_duration_s must be positive")
    selected = profile or MappingProfile()
    target = round(video_duration_s * selected.target_keyframe_hz)
    target = min(selected.max_keyframes, max(selected.min_keyframes, target))
    segments = max(1, math.ceil(video_duration_s / selected.dense_segment_duration_s))
    return MappingPlan(
        session=session.strip() or "mapping_session",
        profile=selected,
        video_duration_s=float(video_duration_s),
        target_keyframes=target,
        dense_segments=segments,
        stages=(
            "video_quality_gate",
            "adaptive_keyframes",
            "sequential_sfm",
            "place_recognition_loop_closure",
            "global_bundle_adjustment",
            "trajectory_quality_gate",
            "metric_scale_anchor",
            "segmented_dense_reconstruction",
            "map_fusion",
            "floor_and_occupancy_projection",
            "publication_and_evaluation",
        ),
    )


def evaluate_sparse_run(
    *,
    selected_keyframes: int,
    registered_images: int,
    mean_reprojection_error_px: float,
    loop_closures: int,
    loop_drift_ratio: float,
    profile: MappingProfile | None = None,
) -> tuple[bool, tuple[str, ...]]:
    selected = profile or MappingProfile()
    if selected_keyframes <= 0:
        raise ValueError("selected_keyframes must be positive")
    registration_ratio = registered_images / selected_keyframes
    failures: list[str] = []
    if registration_ratio < selected.min_registration_ratio:
        failures.append(
            f"registration ratio {registration_ratio:.3f} is below "
            f"{selected.min_registration_ratio:.3f}"
        )
    if mean_reprojection_error_px > selected.max_reprojection_error_px:
        failures.append(
            f"reprojection error {mean_reprojection_error_px:.3f}px exceeds "
            f"{selected.max_reprojection_error_px:.3f}px"
        )
    if loop_closures < selected.min_loop_closures:
        failures.append("no validated loop closure")
    if loop_drift_ratio > selected.max_loop_drift_ratio:
        failures.append(
            f"loop drift {loop_drift_ratio:.3%} exceeds "
            f"{selected.max_loop_drift_ratio:.3%}"
        )
    return not failures, tuple(failures)
