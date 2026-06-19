"""Video inspection and deterministic baseline frame extraction."""

from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np


@dataclass(frozen=True)
class VideoMetadata:
    path: str
    sha256: str
    file_size_bytes: int
    width: int
    height: int
    orientation: str
    fps: float
    declared_frames: int
    decoded_frames: int | None
    duration_s: float
    complete_decode: bool | None

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class FrameExtractionConfig:
    target_fps: float = 3.0
    jpeg_quality: int = 95
    analysis_width: int = 640
    start_s: float = 0.0
    end_s: float | None = None

    def __post_init__(self) -> None:
        if self.target_fps <= 0:
            raise ValueError("target_fps must be positive")
        if not 1 <= self.jpeg_quality <= 100:
            raise ValueError("jpeg_quality must be in [1, 100]")
        if self.analysis_width < 64:
            raise ValueError("analysis_width must be at least 64")
        if self.start_s < 0:
            raise ValueError("start_s must not be negative")
        if self.end_s is not None and self.end_s <= self.start_s:
            raise ValueError("end_s must be greater than start_s")


@dataclass(frozen=True)
class ExtractedFrame:
    sequence: int
    source_frame: int
    timestamp_s: float
    sharpness: float
    brightness_mean: float
    filename: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class FrameExtractionReport:
    video: VideoMetadata
    config: dict[str, object]
    source_frame_stride: int
    extracted_frames: int
    achieved_fps: float
    sharpness: dict[str, float]
    brightness: dict[str, float]
    frames_csv: str
    frames: tuple[dict[str, object], ...]

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["video"] = self.video.to_dict()
        payload["frames"] = list(self.frames)
        return payload


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def inspect_video(path: Path, *, full_decode: bool = False) -> VideoMetadata:
    video_path = Path(path)
    if not video_path.is_file():
        raise FileNotFoundError(f"video not found: {video_path}")

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise ValueError(f"OpenCV could not open video: {video_path}")

    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    declared_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    if width <= 0 or height <= 0 or not np.isfinite(fps) or fps <= 0:
        capture.release()
        raise ValueError("video metadata is incomplete or invalid")

    decoded_frames: int | None = None
    complete_decode: bool | None = None
    if full_decode:
        decoded_frames = 0
        while True:
            ok, _ = capture.read()
            if not ok:
                break
            decoded_frames += 1
        complete_decode = declared_frames <= 0 or decoded_frames == declared_frames
    capture.release()

    frame_count = decoded_frames if decoded_frames is not None else declared_frames
    orientation = "landscape" if width >= height else "portrait"
    return VideoMetadata(
        path=str(video_path),
        sha256=_sha256(video_path),
        file_size_bytes=video_path.stat().st_size,
        width=width,
        height=height,
        orientation=orientation,
        fps=fps,
        declared_frames=declared_frames,
        decoded_frames=decoded_frames,
        duration_s=frame_count / fps,
        complete_decode=complete_decode,
    )


def _analysis_gray(frame: np.ndarray, width: int) -> np.ndarray:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    if gray.shape[1] <= width:
        return gray
    scale = width / gray.shape[1]
    return cv2.resize(
        gray,
        (width, max(1, round(gray.shape[0] * scale))),
        interpolation=cv2.INTER_AREA,
    )


def _distribution(values: list[float]) -> dict[str, float]:
    if not values:
        raise ValueError("cannot summarize an empty measurement list")
    return {
        "min": float(np.min(values)),
        "p10": float(np.percentile(values, 10)),
        "median": float(np.median(values)),
        "p90": float(np.percentile(values, 90)),
        "max": float(np.max(values)),
    }


def extract_frames(
    video_path: Path,
    output_dir: Path,
    config: FrameExtractionConfig | None = None,
) -> FrameExtractionReport:
    selected_config = config or FrameExtractionConfig()
    metadata = inspect_video(video_path, full_decode=False)
    destination = Path(output_dir)
    frames_dir = destination / "frames"
    if destination.exists() and any(destination.iterdir()):
        raise FileExistsError(f"output directory is not empty: {destination}")
    frames_dir.mkdir(parents=True, exist_ok=True)

    stride = max(1, round(metadata.fps / selected_config.target_fps))
    start_frame = round(selected_config.start_s * metadata.fps)
    end_frame = (
        round(selected_config.end_s * metadata.fps)
        if selected_config.end_s is not None
        else metadata.declared_frames
    )

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise ValueError(f"OpenCV could not open video: {video_path}")

    extracted: list[ExtractedFrame] = []
    frame_index = 0
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            if frame_index >= end_frame:
                break
            if frame_index >= start_frame and (frame_index - start_frame) % stride == 0:
                gray = _analysis_gray(frame, selected_config.analysis_width)
                sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
                brightness = float(gray.mean())
                sequence = len(extracted)
                timestamp_s = frame_index / metadata.fps
                filename = f"frame_{sequence:05d}_t{timestamp_s:09.3f}.jpg"
                output_path = frames_dir / filename
                written = cv2.imwrite(
                    str(output_path),
                    frame,
                    [cv2.IMWRITE_JPEG_QUALITY, selected_config.jpeg_quality],
                )
                if not written:
                    raise OSError(f"failed to write frame: {output_path}")
                extracted.append(
                    ExtractedFrame(
                        sequence=sequence,
                        source_frame=frame_index,
                        timestamp_s=timestamp_s,
                        sharpness=sharpness,
                        brightness_mean=brightness,
                        filename=output_path.relative_to(destination).as_posix(),
                    )
                )
            frame_index += 1
    finally:
        capture.release()

    if not extracted:
        raise ValueError("frame extraction produced no frames")

    csv_path = destination / "frames.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(ExtractedFrame.__dataclass_fields__))
        writer.writeheader()
        for item in extracted:
            writer.writerow(item.to_dict())

    covered_duration = extracted[-1].timestamp_s - extracted[0].timestamp_s
    achieved_fps = (
        (len(extracted) - 1) / covered_duration if covered_duration > 0 else metadata.fps / stride
    )
    report = FrameExtractionReport(
        video=metadata,
        config=asdict(selected_config),
        source_frame_stride=stride,
        extracted_frames=len(extracted),
        achieved_fps=achieved_fps,
        sharpness=_distribution([item.sharpness for item in extracted]),
        brightness=_distribution([item.brightness_mean for item in extracted]),
        frames_csv=csv_path.relative_to(destination).as_posix(),
        frames=tuple(item.to_dict() for item in extracted),
    )
    (destination / "extraction_report.json").write_text(
        json.dumps(report.to_dict(), indent=2) + "\n",
        encoding="utf-8",
    )
    return report
