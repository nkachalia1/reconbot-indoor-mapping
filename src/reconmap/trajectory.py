"""Quantitative metrics for reconstructed camera trajectories."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np


@dataclass(frozen=True)
class TrajectoryMetrics:
    poses: int
    path_length: float
    start_end_drift: float
    drift_ratio: float
    median_step: float
    max_step: float

    def to_dict(self) -> dict[str, float | int]:
        return asdict(self)


def trajectory_metrics(positions: np.ndarray) -> TrajectoryMetrics:
    points = np.asarray(positions, dtype=float)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("positions must have shape (N, 3)")
    if len(points) < 2:
        raise ValueError("at least two camera positions are required")
    if not np.isfinite(points).all():
        raise ValueError("positions contain non-finite values")

    steps = np.linalg.norm(np.diff(points, axis=0), axis=1)
    path_length = float(steps.sum())
    drift = float(np.linalg.norm(points[-1] - points[0]))
    return TrajectoryMetrics(
        poses=len(points),
        path_length=path_length,
        start_end_drift=drift,
        drift_ratio=drift / path_length if path_length else 0.0,
        median_step=float(np.median(steps)),
        max_step=float(np.max(steps)),
    )


def metric_scale(estimated_distance: float, measured_distance_m: float) -> float:
    if estimated_distance <= 0 or measured_distance_m <= 0:
        raise ValueError("distances must be positive")
    return measured_distance_m / estimated_distance


def apply_metric_scale(positions: np.ndarray, scale_m_per_unit: float) -> np.ndarray:
    if scale_m_per_unit <= 0:
        raise ValueError("scale_m_per_unit must be positive")
    return np.asarray(positions, dtype=float) * scale_m_per_unit
