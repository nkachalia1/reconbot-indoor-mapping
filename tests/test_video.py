from __future__ import annotations

import json

import cv2
import numpy as np
import pytest

from reconbot.keyframes import AdaptiveKeyframeConfig, select_adaptive_keyframes
from reconbot.stitch import detect_yellow_tape_endpoints
from reconbot.video import FrameExtractionConfig, extract_frames, inspect_video


def _write_synthetic_video(path, *, fps: float = 10.0, frames: int = 30) -> None:
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"MJPG"),
        fps,
        (160, 120),
    )
    if not writer.isOpened():
        pytest.skip("OpenCV build has no MJPG video writer")
    rng = np.random.default_rng(4)
    texture = rng.integers(0, 256, size=(120, 160, 3), dtype=np.uint8)
    for index in range(frames):
        matrix = np.float32([[1, 0, index % 8], [0, 1, 0]])
        frame = cv2.warpAffine(texture, matrix, (160, 120), borderMode=cv2.BORDER_REFLECT)
        writer.write(frame)
    writer.release()


def test_video_inspection_full_decode(tmp_path):
    video = tmp_path / "synthetic.avi"
    _write_synthetic_video(video)

    metadata = inspect_video(video, full_decode=True)

    assert metadata.width == 160
    assert metadata.height == 120
    assert metadata.orientation == "landscape"
    assert metadata.declared_frames == 30
    assert metadata.decoded_frames == 30
    assert metadata.complete_decode is True
    assert metadata.duration_s == pytest.approx(3.0)
    assert len(metadata.sha256) == 64


def test_deterministic_frame_extraction_writes_report(tmp_path):
    video = tmp_path / "synthetic.avi"
    _write_synthetic_video(video)
    output = tmp_path / "frames"

    report = extract_frames(
        video,
        output,
        FrameExtractionConfig(target_fps=2.0),
    )

    payload = json.loads((output / "extraction_report.json").read_text())
    assert report.source_frame_stride == 5
    assert report.extracted_frames == 6
    assert report.achieved_fps == pytest.approx(2.0)
    assert payload["extracted_frames"] == 6
    assert (output / "frames.csv").is_file()
    assert len(list((output / "frames").glob("*.jpg"))) == 6


def test_extraction_refuses_nonempty_output(tmp_path):
    video = tmp_path / "synthetic.avi"
    _write_synthetic_video(video)
    output = tmp_path / "frames"
    output.mkdir()
    (output / "existing.txt").write_text("do not overwrite")

    with pytest.raises(FileExistsError):
        extract_frames(video, output)


def test_adaptive_keyframes_write_quantitative_report(tmp_path):
    frames = tmp_path / "candidates"
    frames.mkdir()
    rng = np.random.default_rng(12)
    texture = rng.integers(0, 256, size=(120, 160), dtype=np.uint8)
    for index in range(20):
        matrix = np.float32([[1, 0, index * 2], [0, 1, 0]])
        frame = cv2.warpAffine(texture, matrix, (160, 120), borderMode=cv2.BORDER_REFLECT)
        if index == 10:
            frame = cv2.GaussianBlur(frame, (21, 21), 0)
        cv2.imwrite(str(frames / f"frame_{index:05d}_t{index * 0.2:09.3f}.jpg"), frame)

    output = tmp_path / "keyframes"
    report = select_adaptive_keyframes(
        frames,
        output,
        AdaptiveKeyframeConfig(
            analysis_width=160,
            min_interval_s=0.4,
            max_interval_s=1.0,
            target_motion_ratio=0.005,
            min_features=20,
            max_corners=100,
        ),
    )

    assert 2 <= report["selected_keyframes"] < report["candidate_frames"]
    assert report["coverage"]["coverage_fraction"] == 1.0
    assert report["coverage"]["max_selected_gap_s"] <= 1.2
    assert report["quality"]["selected_sharpness"]["median"] > 0
    assert (output / "keyframe_report.json").is_file()
    assert (output / "keyframes.csv").is_file()
    assert len((output / "colmap_image_list.txt").read_text().splitlines()) == report[
        "selected_keyframes"
    ]
    assert len(list((output / "frames").glob("*.jpg"))) == report["selected_keyframes"]


def test_adaptive_keyframes_recover_after_tracking_loss(tmp_path):
    frames = tmp_path / "candidates"
    frames.mkdir()
    for index in range(12):
        frame = np.full((120, 160), index * 10, dtype=np.uint8)
        cv2.imwrite(str(frames / f"frame_{index:05d}_t{index * 0.2:09.3f}.jpg"), frame)

    report = select_adaptive_keyframes(
        frames,
        tmp_path / "keyframes",
        AdaptiveKeyframeConfig(
            analysis_width=160,
            min_interval_s=0.2,
            max_interval_s=0.6,
            min_features=1,
        ),
    )

    assert report["coverage"]["max_selected_gap_s"] <= 0.8
    assert any(row["reason"] == "selected_max_gap_fallback" for row in report["frames"])


def test_yellow_tape_endpoint_detection(tmp_path):
    image = np.zeros((400, 600, 3), dtype=np.uint8)
    cv2.line(image, (80, 310), (520, 330), (0, 220, 255), 12)
    path = tmp_path / "tape.jpg"
    cv2.imwrite(str(path), image)

    first, second = detect_yellow_tape_endpoints(path)
    endpoints = sorted((first, second), key=lambda item: item[0])

    assert endpoints[0][0] == pytest.approx(80, abs=8)
    assert endpoints[1][0] == pytest.approx(520, abs=8)
