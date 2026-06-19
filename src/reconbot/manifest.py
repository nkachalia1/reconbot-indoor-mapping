"""Versioned dataset manifests for reproducible reconstruction experiments."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class MetricReference:
    name: str
    distance_m: float
    description: str

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("metric reference name must not be empty")
        if self.distance_m <= 0:
            raise ValueError("metric reference distance_m must be positive")
        if not self.description.strip():
            raise ValueError("metric reference description must not be empty")


@dataclass(frozen=True)
class DatasetManifest:
    dataset_id: str
    video_path: str
    capture_device: str
    environment: str
    route_description: str
    metric_references: tuple[MetricReference, ...]
    schema_version: int = 1

    def __post_init__(self) -> None:
        if self.schema_version != 1:
            raise ValueError("only dataset manifest schema_version 1 is supported")
        if not self.dataset_id.strip():
            raise ValueError("dataset_id must not be empty")
        if not self.video_path.strip():
            raise ValueError("video_path must not be empty")
        if not self.capture_device.strip():
            raise ValueError("capture_device must not be empty")
        if not self.environment.strip():
            raise ValueError("environment must not be empty")
        if not self.route_description.strip():
            raise ValueError("route_description must not be empty")
        if not self.metric_references:
            raise ValueError("at least one metric reference is required")

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["metric_references"] = [asdict(item) for item in self.metric_references]
        return payload

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> DatasetManifest:
        references = tuple(
            MetricReference(**item) for item in payload["metric_references"]  # type: ignore[arg-type]
        )
        return cls(
            schema_version=int(payload.get("schema_version", 1)),
            dataset_id=str(payload["dataset_id"]),
            video_path=str(payload["video_path"]),
            capture_device=str(payload["capture_device"]),
            environment=str(payload["environment"]),
            route_description=str(payload["route_description"]),
            metric_references=references,
        )


def load_manifest(path: Path) -> DatasetManifest:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("dataset manifest root must be a JSON object")
    return DatasetManifest.from_dict(payload)


def save_manifest(path: Path, manifest: DatasetManifest) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(manifest.to_dict(), indent=2) + "\n",
        encoding="utf-8",
    )
