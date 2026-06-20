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

Phase 1 includes verified video ingestion, deterministic frame extraction,
adaptive keyframe selection, feature-match preflight metrics, COLMAP orchestration,
and camera trajectory quality reporting. `hallway_room_001` exposed a measured
room-coverage failure: the hallway and room reconstruct as disconnected sparse
components. Dense processing is blocked until a continuous capture passes the
sparse gates. See [the experiment report](docs/hallway_room_001_results.md).

## Setup

```powershell
cd "C:\Users\Neel\Documents\ReconBot Indoor Mapping"
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"
```

Copy the iPhone recording to:

```text
data\raw\hallway_room_001.mov
```

Create or refresh the dataset manifest:

```powershell
reconbot init-dataset `
  --dataset-id hallway_room_001 `
  --video data\raw\hallway_room_001.mov `
  --output configs\hallway_room_001.json
```

Fully decode and fingerprint the video:

```powershell
reconbot inspect-video `
  --video data\raw\hallway_room_001.mov `
  --full-decode `
  --output outputs\hallway_room_001\video_report.json
```

Extract a deterministic 5 FPS SfM baseline:

```powershell
reconbot extract-frames `
  --video data\raw\hallway_room_001.mov `
  --target-fps 5 `
  --output-dir data\frames\hallway_room_001_sfm_5fps
```

This produces full-resolution JPEGs, `frames.csv`, and an
`extraction_report.json` with sharpness and brightness distributions.

Select a smaller frame set using blur, optical-flow motion, feature tracking,
and temporal coverage metrics:

```powershell
reconbot select-keyframes `
  --frames-dir data\frames\hallway_room_001_sfm_5fps\frames `
  --output-dir data\frames\hallway_room_001_adaptive
```

This writes the selected full-resolution frames plus `keyframes.csv` and
`keyframe_report.json`. Original frame sequence and timestamps remain traceable.

Create the provisional continuous metric map from the two sparse components:

```powershell
reconbot stitch-metric-components `
  --main-model-dir outputs\hallway_room_001\colmap_5fps_targeted_guided_4_0_4\sparse_text `
  --secondary-model-dir outputs\hallway_room_001\colmap_room_focus_250_780_global_4_0_4\sparse_text `
  --images-dir data\frames\hallway_room_001_sfm_5fps\frames `
  --output-dir outputs\hallway_room_001\continuous_metric_map_v1
```

The stitch detects the 4 ft tape, fits reconstructed floor planes, scales both
components, and constrains consecutive camera poses 705/706. It is explicitly
reported as low-confidence when sparse geometry fails the seam-overlap gate.

Run the CPU sparse baseline in WSL (the runner stages its working database on
the Linux ext4 filesystem and exports completed artifacts back to Windows):

```powershell
reconbot run-sparse-wsl `
  --images-dir data\frames\hallway_room_001_sfm_5fps\frames `
  --workspace-dir outputs\hallway_room_001\colmap_5fps_ext4_baseline `
  --mapper-profile baseline `
  --sequential-overlap 40 `
  --loop-anchor-window 10
```

Before mapping begins, the runner exports `matched_database.db`. This stage
boundary preserves feature extraction and matching so mapper experiments do
not repeat the expensive front half of the pipeline. A faster exploratory
mapping run can reuse that database:

```powershell
reconbot run-sparse-wsl `
  --images-dir data\frames\hallway_room_001_sfm_5fps\frames `
  --workspace-dir outputs\hallway_room_001\colmap_5fps_fast `
  --matched-database outputs\hallway_room_001\colmap_5fps_ext4_baseline\matched_database.db `
  --mapper-profile fast
```

The `fast` profile reduces local/global bundle-adjustment frequency and
iterations. It is intended for iteration and must be evaluated against the
`baseline` profile before its geometry is used for accuracy claims.

After COLMAP exports its text model, generate quantitative closed-loop metrics
and trajectory artifacts:

```powershell
reconbot trajectory-report `
  --images-txt outputs\hallway_room_001\colmap_5fps_ext4_baseline\sparse_text\images.txt `
  --expected-images 870 `
  --output-dir outputs\hallway_room_001\trajectory_5fps_baseline
```

The trajectory stage writes JSON quality gates, CSV camera centers, a PLY point
cloud, and a PCA-projected PNG with start/end markers.

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
