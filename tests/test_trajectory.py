from __future__ import annotations

import numpy as np
import pytest

from reconmap.trajectory import apply_metric_scale, metric_scale, trajectory_metrics


def test_closed_square_trajectory_has_zero_loop_drift():
    positions = np.array(
        [[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0], [0, 0, 0]],
        dtype=float,
    )

    metrics = trajectory_metrics(positions)

    assert metrics.path_length == pytest.approx(4.0)
    assert metrics.start_end_drift == 0
    assert metrics.drift_ratio == 0
    assert metrics.median_step == pytest.approx(1.0)


def test_open_trajectory_reports_normalized_drift():
    positions = np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0]], dtype=float)
    metrics = trajectory_metrics(positions)

    assert metrics.path_length == pytest.approx(2.0)
    assert metrics.start_end_drift == pytest.approx(2.0)
    assert metrics.drift_ratio == pytest.approx(1.0)


def test_metric_scale_converts_reconstruction_units_to_meters():
    scale = metric_scale(5.0, 2.0)
    scaled = apply_metric_scale(np.array([[0, 0, 0], [5, 0, 0]]), scale)

    assert scale == pytest.approx(0.4)
    assert scaled[-1, 0] == pytest.approx(2.0)


def test_trajectory_rejects_invalid_shape():
    with pytest.raises(ValueError):
        trajectory_metrics(np.zeros((4, 2)))
