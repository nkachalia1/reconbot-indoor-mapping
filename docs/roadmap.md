# Engineering Roadmap

## Phase 1 — Reproducible Baseline

- [x] versioned dataset manifest
- [x] source-video fingerprint and full-decode inspection
- [x] deterministic frame extraction with blur/brightness reports
- [x] validate the iPhone hallway dataset
- [x] automate COLMAP sparse reconstruction
- [x] parse sparse model metrics and camera poses
- [x] adaptive keyframe selection with blur, motion, and coverage metrics
- [x] document the `hallway_room_001` component-split failure
- [ ] capture `hallway_room_002` with continuous translational parallax

## Phase 2 — Metric Reconstruction

- [x] detect tape endpoints in registered images
- [x] calculate per-component metric scale
- [x] export a provisional constrained continuous metric map
- [ ] measure independent room dimensions
- [ ] generate an accuracy report
- [ ] enforce sparse quality gates before dense processing

## Phase 3 — Dense Products

- [ ] COLMAP dense stereo
- [ ] point-cloud cleanup
- [ ] mesh generation and texturing
- [ ] PLY/OBJ/GLB export
- [ ] trajectory and model viewer

## Phase 4 — Owned Vision Front End

- [ ] ORB/SIFT/AKAZE extraction benchmark
- [ ] geometric match verification
- [ ] feature-match visualization
- [ ] simple place-recognition loop proposals
- [ ] compare custom matches with COLMAP defaults

## Phase 5 — Embedded Mapping Front End

- [ ] Raspberry Pi 5 capture/ingest service
- [ ] real-time feature extraction benchmark
- [ ] laptop transfer protocol
- [ ] CPU, memory, temperature, bandwidth, and latency report

## Stretch

- [ ] SuperPoint + LightGlue experiment
- [ ] incremental pose visualization
- [ ] FastAPI artifact/metadata service
- [ ] 2D floor projection or occupancy representation
