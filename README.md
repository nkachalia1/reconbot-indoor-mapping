# ReconBot Indoor Mapping

A portable monocular mapping system for recovering camera trajectories, sparse
environment maps, dense room geometry, and navigation-oriented occupancy layers
from hallway-and-room walkthroughs.

This is the environment-scale companion to
[ReconBot Portable 3D Reconstruction](https://github.com/nkachalia1/reconbot-portable-3d-reconstruction).
The original project focuses on object-centric orbits. This repository focuses
on long trajectories, loop closure, drift, room segmentation, and map usability.

## Hardware

- Raspberry Pi 5, 8 GB, as field coordinator
- 20,000 mAh USB-C power bank
- Laptop webcam as the monocular sensor
- Windows laptop with WSL, Python, OpenCV, and COLMAP

## Mapping Architecture

```text
walkthrough video
  -> sharp motion-distinct keyframes
  -> sequential SfM + explicit loop closures
  -> bundle-adjusted camera trajectory
  -> metric scale anchor
  -> global sparse environment map
  -> segmented room/hallway dense reconstruction
  -> merged GLB/PLY map
  -> floor plane + 2D occupancy layer
```

The sparse trajectory is treated as a quality gate. Dense reconstruction does
not begin until registration, reprojection error, and loop-closure drift pass.

## First Milestone

Plan a two-to-three minute hallway and single-room capture:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"

reconmap plan `
  --session hallway_room_001 `
  --video-duration-s 180 `
  --output outputs\hallway_room_001\mapping_plan.json
```

Run the test suite:

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

## Initial Quality Gates

- At least 90% of selected keyframes registered
- Mean sparse reprojection error at or below 1.5 px
- At least one validated return-to-start loop closure
- Start/end trajectory drift below 2% of traveled distance
- No dense processing until the sparse map passes

## Capture Protocol

1. Place a known-size fiducial or measured reference near the starting area.
2. Walk at chest height with the camera facing slightly forward and downward.
3. Preserve 70-80% overlap and translate through doorways instead of pivoting.
4. Circle each room once before returning to the hallway.
5. Return to the starting view to create an explicit loop closure.
6. Avoid blank walls, mirrors, windows, moving people, and abrupt exposure shifts.

See [docs/capture_protocol.md](docs/capture_protocol.md) and
[docs/system_architecture.md](docs/system_architecture.md).

## Roadmap

- [x] Environment-specific run profiles and deterministic mapping plans
- [x] Quantitative trajectory length and loop-drift metrics
- [x] Sparse-map acceptance gates with unit tests and CI
- [ ] Video keyframe extraction and blur/motion reports
- [ ] COLMAP sequential matching with place-recognition loop closure
- [ ] Camera trajectory and sparse-map dashboard
- [ ] Metric scale from a fiducial or measured landmark pair
- [ ] Room/hallway segmentation and chunked dense reconstruction
- [ ] Mesh registration, fusion, cleanup, and GLB/PLY export
- [ ] Floor-plane extraction and 2D occupancy-map generation

## Evaluation

Every mapped run will report registration ratio, reprojection error, loop
closures, start/end drift, trajectory length, sparse landmarks, dense coverage,
processing time, peak memory, and metric-reference error. Failed captures remain
useful experimental evidence rather than being hidden behind a visually plausible
mesh.
