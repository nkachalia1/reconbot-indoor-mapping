from __future__ import annotations

import json

import cv2
import numpy as np
import pytest

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
