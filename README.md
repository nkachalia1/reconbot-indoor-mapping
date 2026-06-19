# ReconBot Spatial Reconstruction

ReconBot is an interview-ready computer vision and robotics project that
reconstructs a metrically scaled indoor hallway and room from commodity
smartphone video.

The project is deliberately framed as an engineered spatial reconstruction
system—not “I ran a SLAM package.” COLMAP provides the first geometric baseline,
while this repository owns dataset provenance, quality analysis, orchestration,
trajectory visualization, metric calibration, dimensional evaluation, feature
experiments, and embedded profiling.

## Hardware

- iPhone 14 recording 1080p video
- Raspberry Pi 5 with a 128 GB microSD card
- laptop for reconstruction and visualization
- tape measure visible from 0 to 4 ft

The tape segment is exactly `4 ft = 1.2192 m`. It calibrates monocular scale.
Door width, hallway width, and room dimensions should be measured separately
and used only for accuracy evaluation.

## Outputs

```text
source video
  -> verified dataset + SHA-256 fingerprint
  -> frame set + quality report
  -> sparse point cloud + camera trajectory
  -> metric scale calibration
  -> dimensional accuracy report
  -> dense point cloud
  -> textured mesh (PLY/OBJ/GLB)
```

## Current Status

Phase 1 is implemented through deterministic frame extraction. COLMAP
orchestration and sparse-model evaluation are next.

## Setup

```powershell
cd "C:\Users\Neel\Documents\ReconBot Indoor Mapping"
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"
```

Copy the iPhone recording to:

```text
data\raw\hallway_room_001.mp4
```

Create or refresh the dataset manifest:

```powershell
reconbot init-dataset `
  --dataset-id hallway_room_001 `
  --video data\raw\hallway_room_001.mp4 `
  --output configs\hallway_room_001.json
```

Fully decode and fingerprint the video:

```powershell
reconbot inspect-video `
  --video data\raw\hallway_room_001.mp4 `
  --full-decode `
  --output outputs\hallway_room_001\video_report.json
```

Extract a deterministic 3 FPS baseline:

```powershell
reconbot extract-frames `
  --video data\raw\hallway_room_001.mp4 `
  --target-fps 3 `
  --output-dir data\frames\hallway_room_001_baseline
```

This produces full-resolution JPEGs, `frames.csv`, and an
`extraction_report.json` with sharpness and brightness distributions.

Run verification:

```powershell
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\ruff.exe check src tests
```

## Engineering Principles

- Preserve raw data and fingerprint every input.
- Prefer quantitative reports over visual claims.
- Keep calibration measurements separate from validation measurements.
- Do not run dense reconstruction until sparse geometry passes quality gates.
- Clearly distinguish offline SfM, SLAM concepts, and original project code.
- Treat failed captures and failed reconstructions as experimental evidence.

See [architecture](docs/architecture.md),
[evaluation plan](docs/evaluation_plan.md), and [roadmap](docs/roadmap.md).
The [engineering decisions](docs/engineering_decisions.md) explain the major
tradeoffs, and the [interview guide](docs/interview_guide.md) separates original
work from external components.
