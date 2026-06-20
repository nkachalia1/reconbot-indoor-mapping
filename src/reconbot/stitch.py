"""Metric calibration and constrained stitching of disconnected COLMAP components."""

from __future__ import annotations

import csv
import json
import math
import re
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np

from .trajectory import CameraPose, parse_colmap_images_text

_FRAME_INDEX = re.compile(r"(?:^|[/\\])frame_(\d+)(?:_|\.)")


@dataclass(frozen=True)
class TapeCalibration:
    frame_index: int
    image_name: str
    endpoints_px: tuple[tuple[float, float], tuple[float, float]]
    reconstructed_distance_units: float
    known_distance_m: float
    meters_per_unit: float
    floor_candidate_points: int
    floor_inliers: int
    floor_inlier_fraction: float
    floor_rmse_units: float

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class ConstrainedStitchReport:
    main_calibration: TapeCalibration
    secondary_calibration: TapeCalibration
    secondary_to_main_scale: float
    seam_main_frame: int
    seam_secondary_frame: int
    seam_position_gap_m: float
    seam_orientation_gap_deg: float
    seam_geometry_median_nearest_m: float
    seam_geometry_within_0_25m_fraction: float
    seam_geometry_gate_passed: bool
    incoming_step_m: float
    outgoing_step_m: float
    merged_points: int
    merged_cameras: int
    missing_frame_ranges: tuple[tuple[int, int], ...]
    method: str
    limitations: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["main_calibration"] = self.main_calibration.to_dict()
        payload["secondary_calibration"] = self.secondary_calibration.to_dict()
        return payload


def _frame_index(name: str) -> int:
    match = _FRAME_INDEX.search(name)
    if match is None:
        raise ValueError(f"image name has no frame index: {name}")
    return int(match.group(1))


def _quaternion_rotation(quaternion_wxyz: tuple[float, float, float, float]) -> np.ndarray:
    quaternion = np.asarray(quaternion_wxyz, dtype=np.float64)
    quaternion /= np.linalg.norm(quaternion)
    w, x, y, z = quaternion
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def _load_points(model_dir: Path) -> dict[int, tuple[np.ndarray, tuple[int, int, int]]]:
    points: dict[int, tuple[np.ndarray, tuple[int, int, int]]] = {}
    for line in (Path(model_dir) / "points3D.txt").read_text(encoding="utf-8").splitlines():
        if not line or line.startswith("#"):
            continue
        fields = line.split()
        points[int(fields[0])] = (
            np.asarray(fields[1:4], dtype=np.float64),
            tuple(int(value) for value in fields[4:7]),  # type: ignore[arg-type]
        )
    if not points:
        raise ValueError(f"no points found in {model_dir}")
    return points


def _load_frame_observations(
    model_dir: Path,
    frame_index: int,
    points: dict[int, tuple[np.ndarray, tuple[int, int, int]]],
) -> tuple[str, list[tuple[float, float, np.ndarray]]]:
    lines = (Path(model_dir) / "images.txt").read_text(encoding="utf-8").splitlines()
    index = 0
    while index < len(lines):
        line = lines[index].strip()
        if not line or line.startswith("#"):
            index += 1
            continue
        fields = line.split(maxsplit=9)
        image_name = fields[9]
        point_line = lines[index + 1] if index + 1 < len(lines) else ""
        if _frame_index(image_name) == frame_index:
            observations: list[tuple[float, float, np.ndarray]] = []
            values = point_line.split()
            for offset in range(0, len(values), 3):
                point_id = int(values[offset + 2])
                if point_id != -1 and point_id in points:
                    observations.append(
                        (
                            float(values[offset]),
                            float(values[offset + 1]),
                            points[point_id][0],
                        )
                    )
            return image_name, observations
        index += 2
    raise ValueError(f"frame {frame_index} is not registered in {model_dir}")


def _local_points_for_frame_range(
    model_dir: Path,
    points: dict[int, tuple[np.ndarray, tuple[int, int, int]]],
    start_frame: int,
    end_frame: int,
) -> np.ndarray:
    point_ids: set[int] = set()
    lines = (Path(model_dir) / "images.txt").read_text(encoding="utf-8").splitlines()
    index = 0
    while index < len(lines):
        line = lines[index].strip()
        if not line or line.startswith("#"):
            index += 1
            continue
        fields = line.split(maxsplit=9)
        frame = _frame_index(fields[9])
        if start_frame <= frame <= end_frame:
            values = lines[index + 1].split() if index + 1 < len(lines) else []
            point_ids.update(
                point_id
                for point_id in (int(values[offset + 2]) for offset in range(0, len(values), 3))
                if point_id != -1 and point_id in points
            )
        index += 2
    return np.asarray([points[point_id][0] for point_id in sorted(point_ids)])


def _load_camera_params(model_dir: Path) -> tuple[float, float, float, float]:
    records = [
        line.split()
        for line in (Path(model_dir) / "cameras.txt").read_text(encoding="utf-8").splitlines()
        if line and not line.startswith("#")
    ]
    if len(records) != 1 or records[0][1] != "SIMPLE_RADIAL":
        raise ValueError("metric stitch currently requires one SIMPLE_RADIAL camera")
    return tuple(float(value) for value in records[0][4:8])  # type: ignore[return-value]


def detect_yellow_tape_endpoints(image_path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Detect endpoints of the largest elongated yellow object in the lower image."""
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"OpenCV could not read image: {image_path}")
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array([18, 70, 70]), np.array([42, 255, 255]))
    mask[: int(0.45 * image.shape[0])] = 0
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    candidates: list[tuple[float, int]] = []
    for label in range(1, count):
        _, _, width, height, area = stats[label]
        elongation = max(width, height) / max(1, min(width, height))
        if area >= 100:
            candidates.append((float(area) * elongation, label))
    if not candidates:
        raise ValueError(f"no elongated yellow tape detected in {image_path}")
    label = max(candidates)[1]
    pixels = np.column_stack(np.where(labels == label))[:, ::-1].astype(np.float64)
    center = np.mean(pixels, axis=0)
    _, _, axes = np.linalg.svd(pixels - center, full_matrices=False)
    axis = axes[0]
    projected = (pixels - center) @ axis
    return center + axis * np.min(projected), center + axis * np.max(projected)


def _fit_floor_plane(points: np.ndarray) -> tuple[np.ndarray, float, np.ndarray, float]:
    if len(points) < 30:
        raise ValueError("too few candidate floor points")
    center = np.median(points, axis=0)
    scene_scale = float(np.median(np.linalg.norm(points - center, axis=1)))
    threshold = max(scene_scale * 0.012, 1e-4)
    generator = np.random.default_rng(9)
    best_inliers: np.ndarray | None = None
    for _ in range(5000):
        first, second, third = points[generator.choice(len(points), 3, replace=False)]
        normal = np.cross(second - first, third - first)
        norm = float(np.linalg.norm(normal))
        if norm < 1e-9:
            continue
        normal /= norm
        offset = -float(normal @ first)
        inliers = np.abs(points @ normal + offset) < threshold
        if best_inliers is None or int(inliers.sum()) > int(best_inliers.sum()):
            best_inliers = inliers
    if best_inliers is None or int(best_inliers.sum()) < 20:
        raise ValueError("floor plane RANSAC did not converge")
    inlier_points = points[best_inliers]
    centroid = np.mean(inlier_points, axis=0)
    _, _, axes = np.linalg.svd(inlier_points - centroid, full_matrices=False)
    normal = axes[-1]
    offset = -float(normal @ centroid)
    residuals = inlier_points @ normal + offset
    return normal, offset, best_inliers, float(np.sqrt(np.mean(np.square(residuals))))


def _ray_plane_intersection(
    pose: CameraPose,
    camera_params: tuple[float, float, float, float],
    pixel_xy: np.ndarray,
    plane_normal: np.ndarray,
    plane_offset: float,
) -> np.ndarray:
    focal, principal_x, principal_y, radial = camera_params
    matrix = np.array(
        [[focal, 0.0, principal_x], [0.0, focal, principal_y], [0.0, 0.0, 1.0]]
    )
    normalized = cv2.undistortPoints(
        np.asarray(pixel_xy, dtype=np.float64).reshape(1, 1, 2),
        matrix,
        np.array([radial, 0.0, 0.0, 0.0], dtype=np.float64),
    ).reshape(2)
    camera_ray = np.array([normalized[0], normalized[1], 1.0], dtype=np.float64)
    rotation = _quaternion_rotation(pose.quaternion_wxyz)
    center = np.asarray(pose.center_xyz, dtype=np.float64)
    world_ray = rotation.T @ camera_ray
    denominator = float(plane_normal @ world_ray)
    if abs(denominator) < 1e-9:
        raise ValueError("tape endpoint ray is parallel to the floor plane")
    distance = -float(plane_normal @ center + plane_offset) / denominator
    if distance <= 0:
        raise ValueError("tape endpoint floor intersection is behind the camera")
    return center + distance * world_ray


def calibrate_tape_frame(
    model_dir: Path,
    images_dir: Path,
    *,
    frame_index: int,
    known_distance_m: float = 1.2192,
    floor_image_y_min: float = 850.0,
) -> TapeCalibration:
    model = Path(model_dir)
    points = _load_points(model)
    image_name, observations = _load_frame_observations(model, frame_index, points)
    candidates = np.asarray(
        [point for _, image_y, point in observations if image_y >= floor_image_y_min],
        dtype=np.float64,
    )
    normal, offset, inliers, rmse = _fit_floor_plane(candidates)
    first_pixel, second_pixel = detect_yellow_tape_endpoints(Path(images_dir) / image_name)
    poses = {
        _frame_index(pose.image_name): pose
        for pose in parse_colmap_images_text(model / "images.txt")
    }
    camera_params = _load_camera_params(model)
    first_point = _ray_plane_intersection(
        poses[frame_index], camera_params, first_pixel, normal, offset
    )
    second_point = _ray_plane_intersection(
        poses[frame_index], camera_params, second_pixel, normal, offset
    )
    reconstructed_distance = float(np.linalg.norm(first_point - second_point))
    return TapeCalibration(
        frame_index=frame_index,
        image_name=image_name,
        endpoints_px=(
            tuple(float(value) for value in first_pixel),
            tuple(float(value) for value in second_pixel),
        ),
        reconstructed_distance_units=reconstructed_distance,
        known_distance_m=known_distance_m,
        meters_per_unit=known_distance_m / reconstructed_distance,
        floor_candidate_points=len(candidates),
        floor_inliers=int(inliers.sum()),
        floor_inlier_fraction=float(inliers.mean()),
        floor_rmse_units=rmse,
    )


def _write_ply(path: Path, points: np.ndarray, colors: np.ndarray) -> None:
    with path.open("w", encoding="ascii") as handle:
        handle.write("ply\nformat ascii 1.0\n")
        handle.write(f"element vertex {len(points)}\n")
        handle.write("property float x\nproperty float y\nproperty float z\n")
        handle.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        handle.write("end_header\n")
        for point, color in zip(points, colors, strict=True):
            handle.write(
                f"{point[0]} {point[1]} {point[2]} "
                f"{int(color[0])} {int(color[1])} {int(color[2])}\n"
            )


def _orientation_error_deg(first: np.ndarray, second: np.ndarray) -> float:
    relative = first @ second.T
    cosine = float(np.clip((np.trace(relative) - 1.0) / 2.0, -1.0, 1.0))
    return math.degrees(math.acos(cosine))


def _missing_ranges(frame_indices: list[int]) -> tuple[tuple[int, int], ...]:
    ranges = []
    for first, second in zip(frame_indices, frame_indices[1:], strict=False):
        if second - first > 1:
            ranges.append((first + 1, second - 1))
    return tuple(ranges)


def stitch_metric_components(
    main_model_dir: Path,
    secondary_model_dir: Path,
    images_dir: Path,
    output_dir: Path,
    *,
    main_tape_frame: int = 750,
    secondary_tape_frame: int = 660,
    main_seam_frame: int = 706,
    secondary_seam_frame: int = 705,
    known_distance_m: float = 1.2192,
) -> ConstrainedStitchReport:
    """Scale both models and align their consecutive boundary camera poses."""
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    main_calibration = calibrate_tape_frame(
        main_model_dir,
        images_dir,
        frame_index=main_tape_frame,
        known_distance_m=known_distance_m,
    )
    secondary_calibration = calibrate_tape_frame(
        secondary_model_dir,
        images_dir,
        frame_index=secondary_tape_frame,
        known_distance_m=known_distance_m,
    )
    relative_scale = secondary_calibration.meters_per_unit / main_calibration.meters_per_unit

    main_poses = {
        _frame_index(pose.image_name): pose
        for pose in parse_colmap_images_text(Path(main_model_dir) / "images.txt")
    }
    secondary_poses = {
        _frame_index(pose.image_name): pose
        for pose in parse_colmap_images_text(Path(secondary_model_dir) / "images.txt")
    }
    main_seam = main_poses[main_seam_frame]
    secondary_seam = secondary_poses[secondary_seam_frame]
    main_rotation = _quaternion_rotation(main_seam.quaternion_wxyz)
    secondary_rotation = _quaternion_rotation(secondary_seam.quaternion_wxyz)
    world_rotation = main_rotation.T @ secondary_rotation
    main_center = np.asarray(main_seam.center_xyz)
    secondary_center = np.asarray(secondary_seam.center_xyz)
    translation = main_center - relative_scale * (world_rotation @ secondary_center)

    main_points_map = _load_points(main_model_dir)
    secondary_points_map = _load_points(secondary_model_dir)
    main_points = np.asarray([value[0] for value in main_points_map.values()])
    main_colors = np.asarray([value[1] for value in main_points_map.values()], dtype=np.uint8)
    secondary_points = np.asarray([value[0] for value in secondary_points_map.values()])
    secondary_colors = np.asarray(
        [value[1] for value in secondary_points_map.values()], dtype=np.uint8
    )
    transformed_secondary = (
        relative_scale * (world_rotation @ secondary_points.T)
    ).T + translation
    merged_main_units = np.vstack((main_points, transformed_secondary))
    merged_colors = np.vstack((main_colors, secondary_colors))
    merged_metric = merged_main_units * main_calibration.meters_per_unit
    _write_ply(destination / "continuous_sparse_map_metric.ply", merged_metric, merged_colors)

    camera_rows: list[tuple[int, str, np.ndarray, str]] = []
    for frame, pose in main_poses.items():
        camera_rows.append(
            (
                frame,
                pose.image_name,
                np.asarray(pose.center_xyz) * main_calibration.meters_per_unit,
                "main",
            )
        )
    for frame, pose in secondary_poses.items():
        center_main = (
            relative_scale * (world_rotation @ np.asarray(pose.center_xyz)) + translation
        )
        camera_rows.append(
            (
                frame,
                pose.image_name,
                center_main * main_calibration.meters_per_unit,
                "secondary",
            )
        )
    camera_rows.sort(key=lambda item: item[0])
    with (destination / "continuous_trajectory_metric.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.writer(handle)
        writer.writerow(["frame_index", "image_name", "x_m", "y_m", "z_m", "component"])
        for frame, name, center, component in camera_rows:
            writer.writerow([frame, name, *center, component])
    trajectory_points = np.asarray([row[2] for row in camera_rows])
    trajectory_colors = np.asarray(
        [(40, 140, 255) if row[3] == "main" else (255, 130, 40) for row in camera_rows],
        dtype=np.uint8,
    )
    _write_ply(
        destination / "continuous_camera_trajectory_metric.ply",
        trajectory_points,
        trajectory_colors,
    )

    transformed_seam_center = (
        relative_scale * (world_rotation @ secondary_center) + translation
    )
    transformed_seam_rotation = secondary_rotation @ world_rotation.T
    secondary_seam_points = _local_points_for_frame_range(
        secondary_model_dir,
        secondary_points_map,
        max(0, secondary_seam_frame - 10),
        secondary_seam_frame,
    )
    main_seam_points = _local_points_for_frame_range(
        main_model_dir,
        main_points_map,
        main_seam_frame,
        main_seam_frame + 10,
    )
    transformed_secondary_seam_points = (
        relative_scale * (world_rotation @ secondary_seam_points.T)
    ).T + translation
    matcher = cv2.BFMatcher(cv2.NORM_L2)
    seam_matches = matcher.match(
        (transformed_secondary_seam_points * main_calibration.meters_per_unit).astype(
            np.float32
        ),
        (main_seam_points * main_calibration.meters_per_unit).astype(np.float32),
    )
    seam_distances_m = np.asarray([match.distance for match in seam_matches])
    seam_geometry_median = float(np.median(seam_distances_m))
    seam_geometry_fraction = float(np.mean(seam_distances_m <= 0.25))
    incoming_step = float(
        np.linalg.norm(
            relative_scale
            * world_rotation
            @ (
                np.asarray(secondary_poses[secondary_seam_frame].center_xyz)
                - np.asarray(secondary_poses[secondary_seam_frame - 1].center_xyz)
            )
        )
        * main_calibration.meters_per_unit
    )
    outgoing_step = float(
        np.linalg.norm(
            np.asarray(main_poses[main_seam_frame + 1].center_xyz)
            - np.asarray(main_poses[main_seam_frame].center_xyz)
        )
        * main_calibration.meters_per_unit
    )
    frame_indices = sorted({row[0] for row in camera_rows})
    report = ConstrainedStitchReport(
        main_calibration=main_calibration,
        secondary_calibration=secondary_calibration,
        secondary_to_main_scale=relative_scale,
        seam_main_frame=main_seam_frame,
        seam_secondary_frame=secondary_seam_frame,
        seam_position_gap_m=float(
            np.linalg.norm(transformed_seam_center - main_center)
            * main_calibration.meters_per_unit
        ),
        seam_orientation_gap_deg=_orientation_error_deg(
            main_rotation, transformed_seam_rotation
        ),
        seam_geometry_median_nearest_m=seam_geometry_median,
        seam_geometry_within_0_25m_fraction=seam_geometry_fraction,
        seam_geometry_gate_passed=(
            seam_geometry_median <= 0.25 and seam_geometry_fraction >= 0.50
        ),
        incoming_step_m=incoming_step,
        outgoing_step_m=outgoing_step,
        merged_points=len(merged_metric),
        merged_cameras=len(camera_rows),
        missing_frame_ranges=_missing_ranges(frame_indices),
        method=(
            "Independent tape scaling followed by a six-degree-of-freedom boundary-pose "
            "constraint between consecutive frames 705 and 706."
        ),
        limitations=(
            "The local sparse geometry fails the seam-overlap quality gate.",
            "The stitch is constrained, not jointly bundle-adjusted.",
            "No verified image correspondences cross the 705/706 seam.",
            "Frames without a registered component remain unobserved and are not synthesized.",
            "Metric accuracy depends on automatic tape endpoint and floor-plane estimates.",
        ),
    )
    (destination / "continuous_map_report.json").write_text(
        json.dumps(report.to_dict(), indent=2) + "\n",
        encoding="utf-8",
    )
    return report
