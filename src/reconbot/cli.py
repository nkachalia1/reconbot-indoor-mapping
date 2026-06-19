"""Command-line interface for reproducible indoor reconstruction experiments."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .manifest import DatasetManifest, MetricReference, save_manifest
from .scale import DimensionCheck, build_metric_scale_report
from .video import FrameExtractionConfig, extract_frames, inspect_video


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="reconbot")
    commands = parser.add_subparsers(dest="command", required=True)

    manifest = commands.add_parser("init-dataset", help="Create a versioned dataset manifest.")
    manifest.add_argument("--dataset-id", required=True)
    manifest.add_argument("--video", type=Path, required=True)
    manifest.add_argument("--output", type=Path, required=True)
    manifest.add_argument("--capture-device", default="iPhone 14")
    manifest.add_argument("--environment", default="one hallway and one connected room")
    manifest.add_argument(
        "--route",
        default="closed loop returning to the starting position and view",
    )
    manifest.add_argument("--reference-name", default="tape_0_to_4_ft")
    manifest.add_argument("--reference-distance-m", type=float, default=1.2192)
    manifest.add_argument(
        "--reference-description",
        default="Tape measure endpoints at 0 ft and 4 ft visible in the video.",
    )

    inspect = commands.add_parser(
        "inspect-video",
        help="Inspect and optionally fully decode video.",
    )
    inspect.add_argument("--video", type=Path, required=True)
    inspect.add_argument("--output", type=Path, required=True)
    inspect.add_argument("--full-decode", action="store_true")

    extract = commands.add_parser(
        "extract-frames",
        help="Extract a deterministic baseline frame set with quality metrics.",
    )
    extract.add_argument("--video", type=Path, required=True)
    extract.add_argument("--output-dir", type=Path, required=True)
    extract.add_argument("--target-fps", type=float, default=3.0)
    extract.add_argument("--jpeg-quality", type=int, default=95)
    extract.add_argument("--start-s", type=float, default=0.0)
    extract.add_argument("--end-s", type=float)

    scale = commands.add_parser("scale-report", help="Calculate metric scale and dimension errors.")
    scale.add_argument("--input", type=Path, required=True)
    scale.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if args.command == "init-dataset":
        manifest = DatasetManifest(
            dataset_id=args.dataset_id,
            video_path=args.video.as_posix(),
            capture_device=args.capture_device,
            environment=args.environment,
            route_description=args.route,
            metric_references=(
                MetricReference(
                    name=args.reference_name,
                    distance_m=args.reference_distance_m,
                    description=args.reference_description,
                ),
            ),
        )
        save_manifest(args.output, manifest)
        print(f"Wrote dataset manifest: {args.output}")
        print(f"Metric reference: {args.reference_distance_m:.4f} m")
        return 0

    if args.command == "inspect-video":
        metadata = inspect_video(args.video, full_decode=args.full_decode)
        _write_json(args.output, metadata.to_dict())
        print(f"Wrote video report: {args.output}")
        print(
            f"{metadata.width}x{metadata.height} {metadata.orientation}, "
            f"{metadata.fps:.3f} FPS, {metadata.duration_s:.2f} s"
        )
        if args.full_decode:
            print(f"Complete decode: {metadata.complete_decode}")
        return 0

    if args.command == "extract-frames":
        report = extract_frames(
            args.video,
            args.output_dir,
            FrameExtractionConfig(
                target_fps=args.target_fps,
                jpeg_quality=args.jpeg_quality,
                start_s=args.start_s,
                end_s=args.end_s,
            ),
        )
        print(f"Wrote frame set: {args.output_dir / 'frames'}")
        print(f"Extracted frames: {report.extracted_frames}")
        print(f"Achieved rate: {report.achieved_fps:.3f} FPS")
        print(f"Median sharpness: {report.sharpness['median']:.3f}")
        return 0

    payload = json.loads(args.input.read_text(encoding="utf-8"))
    checks = tuple(DimensionCheck(**item) for item in payload.get("checks", []))
    report = build_metric_scale_report(
        anchor_name=payload["anchor_name"],
        anchor_distance_m=float(payload["anchor_distance_m"]),
        anchor_reconstructed_units=float(payload["anchor_reconstructed_units"]),
        checks=checks,
    )
    _write_json(args.output, report.to_dict())
    print(f"Wrote metric scale report: {args.output}")
    print(f"Scale: {report.meters_per_reconstruction_unit:.6f} m/unit")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
