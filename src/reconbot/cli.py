"""Command-line interface for reproducible indoor reconstruction experiments."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .colmap import ColmapSparseConfig, run_sparse_colmap_wsl
from .dense_qa import write_dense_mesh_qa
from .keyframes import AdaptiveKeyframeConfig, select_adaptive_keyframes
from .manifest import DatasetManifest, MetricReference, save_manifest
from .rematch import prepare_cross_interval_rematch, prepare_targeted_rematch
from .scale import DimensionCheck, build_metric_scale_report
from .stitch import stitch_metric_components
from .trajectory import write_trajectory_artifacts
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

    keyframes = commands.add_parser(
        "select-keyframes",
        help="Select sharp, trackable keyframes with motion and coverage metrics.",
    )
    keyframes.add_argument("--frames-dir", type=Path, required=True)
    keyframes.add_argument("--output-dir", type=Path, required=True)
    keyframes.add_argument("--min-interval-s", type=float, default=0.4)
    keyframes.add_argument("--max-interval-s", type=float, default=1.2)
    keyframes.add_argument("--target-motion-ratio", type=float, default=0.012)
    keyframes.add_argument("--min-track-ratio", type=float, default=0.35)
    keyframes.add_argument("--min-features", type=int, default=80)
    keyframes.add_argument("--blur-percentile", type=float, default=20.0)

    scale = commands.add_parser("scale-report", help="Calculate metric scale and dimension errors.")
    scale.add_argument("--input", type=Path, required=True)
    scale.add_argument("--output", type=Path, required=True)

    sparse = commands.add_parser(
        "run-sparse-wsl",
        help="Run a reproducible CPU COLMAP sparse reconstruction through WSL.",
    )
    sparse.add_argument("--images-dir", type=Path, required=True)
    sparse.add_argument("--workspace-dir", type=Path, required=True)
    sparse.add_argument("--sequential-overlap", type=int, default=40)
    sparse.add_argument("--loop-anchor-window", type=int, default=10)
    sparse.add_argument("--max-num-features", type=int, default=4096)
    sparse.add_argument("--wsl-distribution", default="Ubuntu-22.04")
    sparse.add_argument(
        "--mapper-profile",
        choices=("baseline", "fast"),
        default="baseline",
        help="Use full baseline optimization or a faster exploratory mapper.",
    )
    sparse.add_argument(
        "--matched-database",
        type=Path,
        help="Reuse a completed feature/match database and skip those stages.",
    )

    trajectory = commands.add_parser(
        "trajectory-report",
        help="Export camera trajectory artifacts and closed-loop quality metrics.",
    )
    trajectory.add_argument("--images-txt", type=Path, required=True)
    trajectory.add_argument("--output-dir", type=Path, required=True)
    trajectory.add_argument("--expected-images", type=int)

    rematch = commands.add_parser(
        "prepare-targeted-rematch",
        help="Invalidate and list selected temporal pairs in a copied COLMAP database.",
    )
    rematch.add_argument("--database", type=Path, required=True)
    rematch.add_argument("--pair-list", type=Path, required=True)
    rematch.add_argument("--report", type=Path, required=True)
    rematch.add_argument("--start-frame", type=int, required=True)
    rematch.add_argument("--end-frame", type=int, required=True)
    rematch.add_argument("--overlap", type=int, default=12)

    cross_rematch = commands.add_parser(
        "prepare-cross-rematch",
        help="Invalidate and list all pairs between two revisit intervals.",
    )
    cross_rematch.add_argument("--database", type=Path, required=True)
    cross_rematch.add_argument("--pair-list", type=Path, required=True)
    cross_rematch.add_argument("--report", type=Path, required=True)
    cross_rematch.add_argument("--first-start-frame", type=int, required=True)
    cross_rematch.add_argument("--first-end-frame", type=int, required=True)
    cross_rematch.add_argument("--second-start-frame", type=int, required=True)
    cross_rematch.add_argument("--second-end-frame", type=int, required=True)

    stitch = commands.add_parser(
        "stitch-metric-components",
        help="Scale two sparse models from tape and constrain their consecutive seam poses.",
    )
    stitch.add_argument("--main-model-dir", type=Path, required=True)
    stitch.add_argument("--secondary-model-dir", type=Path, required=True)
    stitch.add_argument("--images-dir", type=Path, required=True)
    stitch.add_argument("--output-dir", type=Path, required=True)
    stitch.add_argument("--main-tape-frame", type=int, default=750)
    stitch.add_argument("--secondary-tape-frame", type=int, default=660)
    stitch.add_argument("--main-seam-frame", type=int, default=706)
    stitch.add_argument("--secondary-seam-frame", type=int, default=705)
    stitch.add_argument("--known-distance-m", type=float, default=1.2192)

    dense_qa = commands.add_parser(
        "dense-qa",
        help="Measure dense mesh topology, component separation, and integrity gates.",
    )
    dense_qa.add_argument("--mesh-ply", type=Path, required=True)
    dense_qa.add_argument("--output-dir", type=Path, required=True)
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

    if args.command == "run-sparse-wsl":
        report = run_sparse_colmap_wsl(
            args.images_dir,
            args.workspace_dir,
            ColmapSparseConfig(
                sequential_overlap=args.sequential_overlap,
                loop_anchor_window=args.loop_anchor_window,
                max_num_features=args.max_num_features,
                wsl_distribution=args.wsl_distribution,
                mapper_profile=args.mapper_profile,
            ),
            matched_database=args.matched_database,
        )
        ratio = report.registered_images / report.images if report.registered_images else 0.0
        print(f"Wrote sparse report: {args.workspace_dir / 'sparse_report.json'}")
        print(f"Registered images: {report.registered_images}/{report.images} ({ratio:.1%})")
        print(f"Sparse points: {report.points}")
        print(f"Mean reprojection error: {report.mean_reprojection_error_px} px")
        return 0

    if args.command == "select-keyframes":
        report = select_adaptive_keyframes(
            args.frames_dir,
            args.output_dir,
            AdaptiveKeyframeConfig(
                min_interval_s=args.min_interval_s,
                max_interval_s=args.max_interval_s,
                target_motion_ratio=args.target_motion_ratio,
                min_track_ratio=args.min_track_ratio,
                min_features=args.min_features,
                blur_percentile=args.blur_percentile,
            ),
        )
        print(f"Wrote adaptive keyframes: {args.output_dir / 'frames'}")
        print(
            f"Selected keyframes: {report['selected_keyframes']}/"
            f"{report['candidate_frames']} ({report['selection_rate']:.1%})"
        )
        print(f"Effective rate: {report['effective_keyframe_fps']:.3f} FPS")
        print(f"Maximum coverage gap: {report['coverage']['max_selected_gap_s']:.3f} s")
        return 0

    if args.command == "trajectory-report":
        report = write_trajectory_artifacts(
            args.images_txt,
            args.output_dir,
            expected_images=args.expected_images,
        )
        print(f"Wrote trajectory artifacts: {args.output_dir}")
        print(f"Registered images: {report.registered_images}")
        print(f"Position closure error: {report.position_closure_error_percent:.2f}%")
        print(f"Orientation closure error: {report.orientation_closure_error_deg:.2f} deg")
        return 0

    if args.command == "prepare-targeted-rematch":
        report = prepare_targeted_rematch(
            args.database,
            args.pair_list,
            args.report,
            start_frame=args.start_frame,
            end_frame=args.end_frame,
            overlap=args.overlap,
        )
        print(f"Prepared targeted pair list: {args.pair_list}")
        print(f"Selected images: {report.selected_images}")
        print(f"Generated pairs: {report.generated_pairs}")
        print(f"Database integrity: {report.database_integrity}")
        return 0

    if args.command == "prepare-cross-rematch":
        report = prepare_cross_interval_rematch(
            args.database,
            args.pair_list,
            args.report,
            first_start_frame=args.first_start_frame,
            first_end_frame=args.first_end_frame,
            second_start_frame=args.second_start_frame,
            second_end_frame=args.second_end_frame,
        )
        print(f"Prepared cross-interval pair list: {args.pair_list}")
        print(f"First interval images: {report.first_images}")
        print(f"Second interval images: {report.second_images}")
        print(f"Generated pairs: {report.generated_pairs}")
        print(f"Database integrity: {report.database_integrity}")
        return 0

    if args.command == "stitch-metric-components":
        report = stitch_metric_components(
            args.main_model_dir,
            args.secondary_model_dir,
            args.images_dir,
            args.output_dir,
            main_tape_frame=args.main_tape_frame,
            secondary_tape_frame=args.secondary_tape_frame,
            main_seam_frame=args.main_seam_frame,
            secondary_seam_frame=args.secondary_seam_frame,
            known_distance_m=args.known_distance_m,
        )
        print(f"Wrote continuous metric map: {args.output_dir}")
        print(f"Merged points: {report.merged_points}")
        print(f"Merged cameras: {report.merged_cameras}")
        print(f"Seam position gap: {report.seam_position_gap_m:.6f} m")
        print(f"Seam orientation gap: {report.seam_orientation_gap_deg:.6f} deg")
        return 0

    if args.command == "dense-qa":
        report = write_dense_mesh_qa(args.mesh_ply, args.output_dir)
        print(f"Wrote dense mesh QA: {args.output_dir}")
        print(f"Connected components: {report.connected_components}")
        print(f"Meaningful components: {report.meaningful_components}")
        print(f"Largest component: {report.largest_component_vertex_fraction:.1%}")
        if report.largest_two_sampled_gap_m is not None:
            print(f"Largest-component sampled gap: {report.largest_two_sampled_gap_m:.3f} m")
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
