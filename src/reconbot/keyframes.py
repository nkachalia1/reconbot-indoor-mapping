"""Adaptive keyframe selection from a deterministic candidate frame set."""

from __future__ import annotations

import csv
import json
import math
import re
import shutil
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np

_FRAME_NAME = re.compile(
    r"^frame_(?P<sequence>\d+)_t(?P<timestamp>\d+(?:\.\d+)?)\.(?:jpg|jpeg|png)$",
    re.I,
)


@dataclass(frozen=True)
class AdaptiveKeyframeConfig:
    """Selection thresholds expressed in time and image-normalized units."""

    analysis_width: int = 480
    min_interval_s: float = 0.4
    max_interval_s: float = 1.2
    target_motion_ratio: float = 0.012
    min_track_ratio: float = 0.35
    min_features: int = 80
    blur_percentile: float = 20.0
    max_corners: int = 600
    coverage_bin_s: float = 5.0

    def __post_init__(self) -> None:
        if self.analysis_width < 64:
            raise ValueError("analysis_width must be at least 64")
        if self.min_interval_s <= 0:
            raise ValueError("min_interval_s must be positive")
        if self.max_interval_s < self.min_interval_s:
            raise ValueError("max_interval_s must be at least min_interval_s")
        if self.target_motion_ratio <= 0:
            raise ValueError("target_motion_ratio must be positive")
        if not 0 <= self.min_track_ratio <= 1:
            raise ValueError("min_track_ratio must be in [0, 1]")
        if self.min_features < 1:
            raise ValueError("min_features must be positive")
        if not 0 <= self.blur_percentile <= 100:
            raise ValueError("blur_percentile must be in [0, 100]")
        if self.max_corners < self.min_features:
            raise ValueError("max_corners must be at least min_features")
        if self.coverage_bin_s <= 0:
            raise ValueError("coverage_bin_s must be positive")


@dataclass(frozen=True)
class CandidateFrame:
    source_sequence: int
    timestamp_s: float
    path: Path
    sharpness: float
    brightness_mean: float
    feature_count: int


def _analysis_gray(path: Path, width: int) -> np.ndarray:
    frame = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if frame is None:
        raise ValueError(f"OpenCV could not read candidate frame: {path}")
    if frame.shape[1] <= width:
        return frame
    scale = width / frame.shape[1]
    return cv2.resize(
        frame,
        (width, max(1, round(frame.shape[0] * scale))),
        interpolation=cv2.INTER_AREA,
    )


def _corners(gray: np.ndarray, max_corners: int) -> np.ndarray | None:
    return cv2.goodFeaturesToTrack(
        gray,
        maxCorners=max_corners,
        qualityLevel=0.01,
        minDistance=7,
        blockSize=7,
    )


def _frame_metrics(path: Path, config: AdaptiveKeyframeConfig) -> tuple[float, float, int]:
    gray = _analysis_gray(path, config.analysis_width)
    sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    points = _corners(gray, config.max_corners)
    return sharpness, float(gray.mean()), 0 if points is None else len(points)


def _pair_metrics(
    previous: np.ndarray,
    current: np.ndarray,
    max_corners: int,
) -> tuple[int, int, float, float]:
    points = _corners(previous, max_corners)
    if points is None or len(points) == 0:
        return 0, 0, 0.0, 0.0
    tracked, status, _ = cv2.calcOpticalFlowPyrLK(
        previous,
        current,
        points,
        None,
        winSize=(21, 21),
        maxLevel=3,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
    )
    if tracked is None or status is None:
        return len(points), 0, 0.0, 0.0
    valid = status.reshape(-1).astype(bool)
    tracked_count = int(valid.sum())
    if tracked_count == 0:
        return len(points), 0, 0.0, 0.0
    displacement = np.linalg.norm(
        tracked.reshape(-1, 2)[valid] - points.reshape(-1, 2)[valid],
        axis=1,
    )
    diagonal = math.hypot(*previous.shape)
    return (
        len(points),
        tracked_count,
        tracked_count / len(points),
        float(np.median(displacement) / diagonal),
    )


def _distribution(values: list[float]) -> dict[str, float]:
    if not values:
        return {}
    return {
        "min": float(np.min(values)),
        "p10": float(np.percentile(values, 10)),
        "median": float(np.median(values)),
        "p90": float(np.percentile(values, 90)),
        "max": float(np.max(values)),
    }


def _discover_candidates(frames_dir: Path, config: AdaptiveKeyframeConfig) -> list[CandidateFrame]:
    candidates: list[CandidateFrame] = []
    for path in sorted(Path(frames_dir).iterdir()):
        match = _FRAME_NAME.match(path.name)
        if not path.is_file() or match is None:
            continue
        sharpness, brightness, feature_count = _frame_metrics(path, config)
        candidates.append(
            CandidateFrame(
                source_sequence=int(match.group("sequence")),
                timestamp_s=float(match.group("timestamp")),
                path=path,
                sharpness=sharpness,
                brightness_mean=brightness,
                feature_count=feature_count,
            )
        )
    if not candidates:
        raise ValueError(f"no candidate frames matched the expected naming scheme in {frames_dir}")
    return candidates


def select_adaptive_keyframes(
    frames_dir: Path,
    output_dir: Path,
    config: AdaptiveKeyframeConfig | None = None,
) -> dict[str, object]:
    """Select sharp, trackable keyframes while bounding temporal coverage gaps."""

    selected_config = config or AdaptiveKeyframeConfig()
    source = Path(frames_dir)
    destination = Path(output_dir)
    if not source.is_dir():
        raise FileNotFoundError(f"candidate frame directory not found: {source}")
    if destination.exists() and any(destination.iterdir()):
        raise FileExistsError(f"output directory is not empty: {destination}")
    output_frames = destination / "frames"
    output_frames.mkdir(parents=True, exist_ok=True)

    candidates = _discover_candidates(source, selected_config)
    blur_threshold = float(
        np.percentile([item.sharpness for item in candidates], selected_config.blur_percentile)
    )

    rows: list[dict[str, object]] = []
    selected_indices: list[int] = []
    last_selected_gray: np.ndarray | None = None
    last_selected_time: float | None = None

    for index, candidate in enumerate(candidates):
        current_gray = _analysis_gray(candidate.path, selected_config.analysis_width)
        initial_features = 0
        tracked_features = 0
        track_ratio = 0.0
        motion_ratio = 0.0
        elapsed_s = (
            0.0
            if last_selected_time is None
            else candidate.timestamp_s - last_selected_time
        )
        if last_selected_gray is not None:
            initial_features, tracked_features, track_ratio, motion_ratio = _pair_metrics(
                last_selected_gray,
                current_gray,
                selected_config.max_corners,
            )

        is_first = index == 0
        is_last = index == len(candidates) - 1
        sharp_enough = candidate.sharpness >= blur_threshold
        features_enough = candidate.feature_count >= selected_config.min_features
        overlap_enough = track_ratio >= selected_config.min_track_ratio
        motion_enough = motion_ratio >= selected_config.target_motion_ratio

        selected = False
        reason = "reject_low_motion"
        if is_first:
            selected, reason = True, "first_frame"
        elif is_last:
            selected, reason = True, "last_frame"
        elif elapsed_s < selected_config.min_interval_s:
            reason = "reject_min_interval"
        elif elapsed_s >= selected_config.max_interval_s:
            selected, reason = True, "selected_max_gap_fallback"
        elif not sharp_enough:
            reason = "reject_blur"
        elif not features_enough:
            reason = "reject_low_features"
        elif not overlap_enough:
            reason = "reject_low_overlap"
        elif motion_enough:
            selected, reason = True, "selected_motion"

        if selected:
            selected_indices.append(index)
            last_selected_gray = current_gray
            last_selected_time = candidate.timestamp_s

        rows.append(
            {
                "source_sequence": candidate.source_sequence,
                "source_filename": candidate.path.name,
                "timestamp_s": candidate.timestamp_s,
                "sharpness": candidate.sharpness,
                "brightness_mean": candidate.brightness_mean,
                "feature_count": candidate.feature_count,
                "elapsed_from_keyframe_s": elapsed_s,
                "initial_tracked_features": initial_features,
                "tracked_features": tracked_features,
                "track_ratio": track_ratio,
                "motion_ratio": motion_ratio,
                "selected": selected,
                "reason": reason,
                "output_filename": "",
            }
        )

    for selected_sequence, candidate_index in enumerate(selected_indices):
        candidate = candidates[candidate_index]
        filename = (
            f"frame_{selected_sequence:05d}_source_{candidate.source_sequence:05d}"
            f"_t{candidate.timestamp_s:09.3f}{candidate.path.suffix.lower()}"
        )
        shutil.copy2(candidate.path, output_frames / filename)
        rows[candidate_index]["output_filename"] = f"frames/{filename}"

    csv_path = destination / "keyframes.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    image_list_path = destination / "colmap_image_list.txt"
    image_list_path.write_text(
        "\n".join(str(row["source_filename"]) for row in rows if row["selected"]) + "\n",
        encoding="utf-8",
    )

    selected_rows = [row for row in rows if row["selected"]]
    selected_times = [float(row["timestamp_s"]) for row in selected_rows]
    gaps = [
        later - earlier
        for earlier, later in zip(selected_times, selected_times[1:], strict=False)
    ]
    start_time = candidates[0].timestamp_s
    duration_s = candidates[-1].timestamp_s - start_time
    bin_count = max(1, math.ceil(max(duration_s, 1e-9) / selected_config.coverage_bin_s))
    coverage_bins: list[dict[str, object]] = []
    for bin_index in range(bin_count):
        bin_start = start_time + bin_index * selected_config.coverage_bin_s
        bin_end = min(candidates[-1].timestamp_s, bin_start + selected_config.coverage_bin_s)
        candidate_count = sum(bin_start <= item.timestamp_s <= bin_end for item in candidates)
        selected_count = sum(bin_start <= timestamp <= bin_end for timestamp in selected_times)
        coverage_bins.append(
            {
                "start_s": bin_start,
                "end_s": bin_end,
                "candidate_count": candidate_count,
                "selected_count": selected_count,
                "covered": selected_count > 0,
            }
        )

    selected_pair_rows = selected_rows[1:]
    max_selected_gap_s = max(gaps, default=0.0)
    covered_bins = sum(bool(item["covered"]) for item in coverage_bins)
    report: dict[str, object] = {
        "config": asdict(selected_config),
        "candidate_frames": len(candidates),
        "selected_keyframes": len(selected_rows),
        "selection_rate": len(selected_rows) / len(candidates),
        "duration_s": duration_s,
        "effective_keyframe_fps": (len(selected_rows) - 1) / duration_s if duration_s > 0 else 0.0,
        "blur_threshold": blur_threshold,
        "rejections": dict(Counter(str(row["reason"]) for row in rows if not row["selected"])),
        "quality": {
            "candidate_sharpness": _distribution([item.sharpness for item in candidates]),
            "selected_sharpness": _distribution(
                [float(row["sharpness"]) for row in selected_rows]
            ),
            "candidate_features": _distribution(
                [float(item.feature_count) for item in candidates]
            ),
            "selected_features": _distribution(
                [float(row["feature_count"]) for row in selected_rows]
            ),
        },
        "motion": {
            "selected_motion_ratio": _distribution(
                [float(row["motion_ratio"]) for row in selected_pair_rows]
            ),
            "selected_track_ratio": _distribution(
                [float(row["track_ratio"]) for row in selected_pair_rows]
            ),
        },
        "coverage": {
            "bin_size_s": selected_config.coverage_bin_s,
            "covered_bins": covered_bins,
            "total_bins": len(coverage_bins),
            "coverage_fraction": covered_bins / len(coverage_bins),
            "max_selected_gap_s": max_selected_gap_s,
            "bins": coverage_bins,
        },
        "gates": {
            "all_time_bins_covered": covered_bins == len(coverage_bins),
            "max_gap_at_most_1_5x_target": max_selected_gap_s
            <= 1.5 * selected_config.max_interval_s,
            "median_selected_track_ratio_at_least_minimum": (
                float(np.median([float(row["track_ratio"]) for row in selected_pair_rows]))
                >= selected_config.min_track_ratio
                if selected_pair_rows
                else False
            ),
        },
        "frames_csv": csv_path.name,
        "colmap_image_list": image_list_path.name,
        "frames": rows,
    }
    (destination / "keyframe_report.json").write_text(
        json.dumps(report, indent=2) + "\n",
        encoding="utf-8",
    )
    return report
