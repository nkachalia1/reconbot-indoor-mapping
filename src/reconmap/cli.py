"""Command line entry point for indoor mapping planning and evaluation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .planning import build_mapping_plan
from .trajectory import trajectory_metrics


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="reconmap")
    commands = parser.add_subparsers(dest="command", required=True)

    plan = commands.add_parser("plan", help="Create an environment mapping run plan.")
    plan.add_argument("--session", required=True)
    plan.add_argument("--video-duration-s", type=float, required=True)
    plan.add_argument("--output", type=Path, required=True)

    trajectory = commands.add_parser("trajectory-metrics", help="Evaluate Nx3 camera positions.")
    trajectory.add_argument("--positions", type=Path, required=True, help="JSON array or .npy file.")
    trajectory.add_argument("--output", type=Path, required=True)
    return parser


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "plan":
        plan = build_mapping_plan(args.session, args.video_duration_s)
        _write_json(args.output, plan.to_dict())
        print(f"Wrote mapping plan: {args.output}")
        print(f"Target keyframes: {plan.target_keyframes}")
        print(f"Dense segments: {plan.dense_segments}")
        return 0

    if args.positions.suffix.lower() == ".npy":
        positions = np.load(args.positions)
    else:
        positions = np.asarray(json.loads(args.positions.read_text(encoding="utf-8")))
    metrics = trajectory_metrics(positions)
    _write_json(args.output, metrics.to_dict())
    print(f"Wrote trajectory metrics: {args.output}")
    print(f"Path length: {metrics.path_length:.3f} reconstruction units")
    print(f"Loop drift: {metrics.drift_ratio:.3%}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
