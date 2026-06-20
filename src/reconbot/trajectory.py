"""Camera trajectory extraction and closed-loop diagnostics for COLMAP models."""

from __future__ import annotations

import csv
import json
import math
import re
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np


@dataclass(frozen=True)
class CameraPose:
    image_id: int
    image_name: str
    quaternion_wxyz: tuple[float, float, float, float]
    translation_xyz: tuple[float, float, float]
    center_xyz: tuple[float, float, float]


@dataclass(frozen=True)
class TrajectoryReport:
    registered_images: int
    expected_images: int | None
    registration_rate: float | None
    path_length_units: float
    start_end_distance_units: float
    position_closure_error_percent: float
    orientation_closure_error_deg: float
    median_step_units: float
    p95_step_units: float
    max_step_units: float
    max_step_to_median_ratio: float | None
    largest_frame_index_gap: int | None
    gates: dict[str, bool | None]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _camera_center(
    quaternion_wxyz: tuple[float, float, float, float],
    translation_xyz: tuple[float, float, float],
) -> tuple[float, float, float]:
    quaternion = np.asarray(quaternion_wxyz, dtype=np.float64)
    norm = float(np.linalg.norm(quaternion))
    if norm == 0:
        raise ValueError("COLMAP quaternion must be non-zero")
    w, x, y, z = quaternion / norm
    rotation = np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )
    center = -(rotation.T @ np.asarray(translation_xyz, dtype=np.float64))
    return tuple(float(value) for value in center)


def parse_colmap_images_text(path: Path) -> tuple[CameraPose, ...]:
    """Parse poses from COLMAP's two-line-per-image ``images.txt`` format."""
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    poses: list[CameraPose] = []
    index = 0
    while index < len(lines):
        line = lines[index].strip()
        if not line or line.startswith("#"):
            index += 1
            continue

        fields = line.split(maxsplit=9)
        if len(fields) != 10:
            raise ValueError(f"invalid COLMAP image record at line {index + 1}: {line}")
        try:
            image_id = int(fields[0])
            quaternion = tuple(float(value) for value in fields[1:5])
            translation = tuple(float(value) for value in fields[5:8])
            int(fields[8])
        except ValueError as error:
            raise ValueError(
                f"invalid COLMAP image record at line {index + 1}: {line}"
            ) from error

        pose = CameraPose(
            image_id=image_id,
            image_name=fields[9],
            quaternion_wxyz=quaternion,  # type: ignore[arg-type]
            translation_xyz=translation,  # type: ignore[arg-type]
            center_xyz=_camera_center(quaternion, translation),  # type: ignore[arg-type]
        )
        poses.append(pose)
        index += 2  # The following line contains POINTS2D, and may be empty.

    if not poses:
        raise ValueError(f"no registered image poses found in {path}")
    return tuple(sorted(poses, key=lambda item: item.image_name))


def _orientation_error_deg(first: CameraPose, last: CameraPose) -> float:
    first_q = np.asarray(first.quaternion_wxyz, dtype=np.float64)
    last_q = np.asarray(last.quaternion_wxyz, dtype=np.float64)
    first_q /= np.linalg.norm(first_q)
    last_q /= np.linalg.norm(last_q)
    cosine = float(np.clip(abs(np.dot(first_q, last_q)), 0.0, 1.0))
    return math.degrees(2.0 * math.acos(cosine))


def _frame_index(name: str) -> int | None:
    frame_match = re.search(r"(?:^|[/\\])frame_(\d+)(?:_|\.)", name)
    if frame_match:
        return int(frame_match.group(1))
    match = re.search(r"(\d+)(?=\.[^.]+$)", name)
    return int(match.group(1)) if match else None


def analyze_trajectory(
    poses: tuple[CameraPose, ...],
    *,
    expected_images: int | None = None,
) -> TrajectoryReport:
    if len(poses) < 2:
        raise ValueError("at least two registered poses are required")
    if expected_images is not None and expected_images < len(poses):
        raise ValueError("expected_images cannot be lower than registered poses")

    centers = np.asarray([pose.center_xyz for pose in poses], dtype=np.float64)
    steps = np.linalg.norm(np.diff(centers, axis=0), axis=1)
    path_length = float(np.sum(steps))
    if path_length == 0:
        raise ValueError("trajectory has zero path length")
    closure_distance = float(np.linalg.norm(centers[-1] - centers[0]))
    closure_percent = closure_distance / path_length * 100
    median_step = float(np.median(steps))
    max_step = float(np.max(steps))
    step_ratio = max_step / median_step if median_step > 0 else None
    frame_indices = [_frame_index(pose.image_name) for pose in poses]
    valid_indices = [value for value in frame_indices if value is not None]
    largest_gap = (
        max(
            second - first
            for first, second in zip(valid_indices, valid_indices[1:], strict=False)
        )
        if len(valid_indices) > 1
        else None
    )
    registration_rate = (
        len(poses) / expected_images
        if expected_images is not None and expected_images > 0
        else None
    )
    orientation_error = _orientation_error_deg(poses[0], poses[-1])

    gates: dict[str, bool | None] = {
        "registration_rate_at_least_80_percent": (
            registration_rate >= 0.80 if registration_rate is not None else None
        ),
        "position_closure_error_at_most_5_percent": closure_percent <= 5.0,
        "orientation_closure_error_at_most_15_deg": orientation_error <= 15.0,
        "max_step_at_most_10x_median": step_ratio <= 10.0 if step_ratio is not None else None,
    }
    return TrajectoryReport(
        registered_images=len(poses),
        expected_images=expected_images,
        registration_rate=registration_rate,
        path_length_units=path_length,
        start_end_distance_units=closure_distance,
        position_closure_error_percent=closure_percent,
        orientation_closure_error_deg=orientation_error,
        median_step_units=median_step,
        p95_step_units=float(np.percentile(steps, 95)),
        max_step_units=max_step,
        max_step_to_median_ratio=step_ratio,
        largest_frame_index_gap=largest_gap,
        gates=gates,
    )


def write_trajectory_artifacts(
    images_text: Path,
    output_dir: Path,
    *,
    expected_images: int | None = None,
) -> TrajectoryReport:
    poses = parse_colmap_images_text(images_text)
    report = analyze_trajectory(poses, expected_images=expected_images)
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)

    with (destination / "trajectory.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["sequence", "image_id", "image_name", "x", "y", "z"])
        for sequence, pose in enumerate(poses):
            writer.writerow([sequence, pose.image_id, pose.image_name, *pose.center_xyz])

    with (destination / "camera_trajectory.ply").open("w", encoding="ascii") as handle:
        handle.write("ply\nformat ascii 1.0\n")
        handle.write(f"element vertex {len(poses)}\n")
        handle.write("property float x\nproperty float y\nproperty float z\n")
        handle.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        handle.write("end_header\n")
        for index, pose in enumerate(poses):
            if index == 0:
                color = (0, 255, 0)
            elif index == len(poses) - 1:
                color = (255, 0, 0)
            else:
                color = (40, 140, 255)
            x, y, z = pose.center_xyz
            red, green, blue = color
            handle.write(f"{x} {y} {z} {red} {green} {blue}\n")

    _write_trajectory_plot(poses, report, destination / "trajectory_plot.png")

    (destination / "trajectory_report.json").write_text(
        json.dumps(report.to_dict(), indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return report


def _write_trajectory_plot(
    poses: tuple[CameraPose, ...],
    report: TrajectoryReport,
    destination: Path,
) -> None:
    """Render a dependency-light PCA projection of the reconstructed camera path."""
    centers = np.asarray([pose.center_xyz for pose in poses], dtype=np.float64)
    centered = centers - np.mean(centers, axis=0)
    _, _, axes = np.linalg.svd(centered, full_matrices=False)
    projected = centered @ axes[:2].T

    width, height, margin = 1200, 800, 80
    lower = np.min(projected, axis=0)
    upper = np.max(projected, axis=0)
    span = np.maximum(upper - lower, 1e-12)
    scale = min((width - 2 * margin) / span[0], (height - 2 * margin) / span[1])
    pixels = np.empty_like(projected)
    pixels[:, 0] = margin + (projected[:, 0] - lower[0]) * scale
    pixels[:, 1] = height - margin - (projected[:, 1] - lower[1]) * scale
    points = np.rint(pixels).astype(np.int32)

    canvas = np.full((height, width, 3), 248, dtype=np.uint8)
    cv2.polylines(canvas, [points.reshape(-1, 1, 2)], False, (220, 120, 30), 3)
    cv2.circle(canvas, tuple(points[0]), 10, (30, 170, 30), -1)
    cv2.circle(canvas, tuple(points[-1]), 10, (30, 30, 220), -1)
    title = (
        f"COLMAP camera trajectory (PCA projection) | "
        f"closure {report.position_closure_error_percent:.2f}%"
    )
    cv2.putText(
        canvas,
        title,
        (margin, 42),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (25, 25, 25),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        canvas,
        "green=start  red=end",
        (margin, height - 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (60, 60, 60),
        1,
        cv2.LINE_AA,
    )
    if not cv2.imwrite(str(destination), canvas):
        raise RuntimeError(f"failed to write trajectory plot: {destination}")
