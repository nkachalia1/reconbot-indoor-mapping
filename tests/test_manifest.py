from __future__ import annotations

import pytest

from reconbot.manifest import DatasetManifest, MetricReference, load_manifest, save_manifest


def test_manifest_round_trip(tmp_path):
    manifest = DatasetManifest(
        dataset_id="hallway_room_001",
        video_path="data/raw/hallway_room_001.mp4",
        capture_device="iPhone 14",
        environment="hallway and room",
        route_description="closed loop",
        metric_references=(
            MetricReference(
                name="tape",
                distance_m=1.2192,
                description="0 to 4 ft",
            ),
        ),
    )
    path = tmp_path / "manifest.json"

    save_manifest(path, manifest)

    assert load_manifest(path) == manifest


def test_manifest_requires_metric_reference():
    with pytest.raises(ValueError):
        DatasetManifest(
            dataset_id="bad",
            video_path="video.mp4",
            capture_device="camera",
            environment="room",
            route_description="loop",
            metric_references=(),
        )
