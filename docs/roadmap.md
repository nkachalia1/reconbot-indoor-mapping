# Engineering Roadmap

## Phase 1 — Reproducible Baseline

- [x] versioned dataset manifest
- [x] source-video fingerprint and full-decode inspection
- [x] deterministic frame extraction with blur/brightness reports
- [ ] validate the iPhone hallway dataset
- [ ] automate COLMAP sparse reconstruction
- [ ] parse sparse model metrics and camera poses

## Phase 2 — Metric Reconstruction

- [ ] annotate tape endpoints in registered images
- [ ] calculate global metric scale
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
