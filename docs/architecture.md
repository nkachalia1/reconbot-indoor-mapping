# System Architecture

## Project Positioning

ReconBot is an end-to-end monocular spatial reconstruction system, not a claim
that COLMAP is original work and not a relabeling of offline SfM as real-time
SLAM.

The project owns:

- dataset design and provenance
- frame-selection policy and quality reporting
- feature/matcher experiments
- COLMAP orchestration and failure handling
- camera-trajectory parsing and visualization
- metric-scale calibration
- dimensional accuracy evaluation
- dense reconstruction, mesh publication, and system profiling

COLMAP initially supplies proven geometric optimization: feature extraction,
matching, triangulation, bundle adjustment, and dense stereo. Later experiments
replace selected front-end components and compare them quantitatively.

## Pipeline

```text
iPhone 14 video + physical measurements
  -> immutable dataset manifest and video fingerprint
  -> video integrity and capture-quality report
  -> deterministic baseline frames
  -> feature/matcher benchmark
  -> COLMAP sparse SfM
  -> camera trajectory + sparse quality gates
  -> tape-based metric scale
  -> independent dimension error report
  -> dense point cloud
  -> mesh and GLB/PLY publication
  -> Raspberry Pi front-end profiling
```

## Why SfM Before SLAM

Offline SfM separates geometry problems from real-time systems problems. It
provides a strong baseline before adding online state estimation, loop closure,
latency constraints, and embedded deployment. This makes later SLAM-like work
measurable rather than theatrical.

## Coordinate and Scale Strategy

Monocular reconstruction produces coordinates up to an unknown global scale.
The visible 0–4 ft tape segment is 1.2192 m and becomes the primary scale
anchor. Door width, hallway width, and room dimensions must be measured
independently and reserved as validation checks; using them all for calibration
would hide error.
