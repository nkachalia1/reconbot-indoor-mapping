from __future__ import annotations

import pytest

from reconbot.scale import DimensionCheck, build_metric_scale_report


def test_metric_scale_and_dimension_error():
    report = build_metric_scale_report(
        anchor_name="four_foot_tape",
        anchor_distance_m=1.2192,
        anchor_reconstructed_units=0.5,
        checks=(
            DimensionCheck(
                name="door_width",
                actual_m=0.9144,
                reconstructed_units=0.36,
            ),
        ),
    )

    assert report.meters_per_reconstruction_unit == pytest.approx(2.4384)
    assert report.checks[0].estimated_m == pytest.approx(0.877824)
    assert report.checks[0].percent_error == pytest.approx(4.0)
    assert report.mean_percent_error == pytest.approx(4.0)


def test_metric_scale_rejects_non_positive_anchor():
    with pytest.raises(ValueError):
        build_metric_scale_report(
            anchor_name="bad",
            anchor_distance_m=1.0,
            anchor_reconstructed_units=0.0,
        )
