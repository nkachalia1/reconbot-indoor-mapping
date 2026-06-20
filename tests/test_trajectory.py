from __future__ import annotations

import json
from pathlib import Path

import pytest

from reconbot.trajectory import (
    analyze_trajectory,
    parse_colmap_images_text,
    write_trajectory_artifacts,
)


def _write_model(path: Path, records: list[str]) -> None:
    path.write_text(
        "# Image list with two lines of data per image:\n"
        "# IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n"
        + "\n".join(f"{record}\n" for record in records),
        encoding="utf-8",
    )


def test_closed_loop_trajectory_metrics(tmp_path):
    images = tmp_path / "images.txt"
    _write_model(
        images,
        [
            "1 1 0 0 0 0 0 0 1 frame_000000.jpg",
            "2 1 0 0 0 -1 0 0 1 frame_000001.jpg",
            "3 1 0 0 0 0 0 0 1 frame_000002.jpg",
        ],
    )

    poses = parse_colmap_images_text(images)
    report = analyze_trajectory(poses, expected_images=3)

    assert poses[1].center_xyz == pytest.approx((1.0, 0.0, 0.0))
    assert report.path_length_units == pytest.approx(2.0)
    assert report.start_end_distance_units == pytest.approx(0.0)
    assert report.position_closure_error_percent == pytest.approx(0.0)
    assert report.orientation_closure_error_deg == pytest.approx(0.0)
    assert report.registration_rate == pytest.approx(1.0)
    assert all(value is True for value in report.gates.values())


def test_trajectory_artifacts_and_failed_orientation_gate(tmp_path):
    images = tmp_path / "images.txt"
    _write_model(
        images,
        [
            "1 1 0 0 0 0 0 0 1 000000.jpg",
            "2 0 0 0 1 -1 0 0 1 000003.jpg",
        ],
    )
    output = tmp_path / "trajectory"

    report = write_trajectory_artifacts(images, output, expected_images=3)

    assert report.orientation_closure_error_deg == pytest.approx(180.0)
    assert report.largest_frame_index_gap == 3
    assert report.gates["orientation_closure_error_at_most_15_deg"] is False
    assert (output / "trajectory.csv").is_file()
    assert (output / "camera_trajectory.ply").is_file()
    assert (output / "trajectory_plot.png").is_file()
    payload = json.loads((output / "trajectory_report.json").read_text(encoding="utf-8"))
    assert payload["registered_images"] == 2


def test_frame_sequence_takes_priority_over_timestamp_suffix(tmp_path):
    images = tmp_path / "images.txt"
    _write_model(
        images,
        [
            "1 1 0 0 0 0 0 0 1 frame_00005_t00001.000.jpg",
            "2 1 0 0 0 -1 0 0 1 frame_00008_t00001.600.jpg",
        ],
    )

    report = analyze_trajectory(parse_colmap_images_text(images))

    assert report.largest_frame_index_gap == 3


def test_trajectory_requires_two_poses(tmp_path):
    images = tmp_path / "images.txt"
    _write_model(images, ["1 1 0 0 0 0 0 0 1 000000.jpg"])

    with pytest.raises(ValueError, match="at least two"):
        analyze_trajectory(parse_colmap_images_text(images))
