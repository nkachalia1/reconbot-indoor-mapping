# Engineering Decisions

## ADR-001: Establish Offline SfM Before SLAM

**Decision:** Build a reproducible COLMAP Structure-from-Motion baseline before
adding online pose estimation.

**Why:** Offline reconstruction isolates camera geometry, frame quality, scale,
and matching failures from real-time scheduling and embedded constraints.

**Consequence:** The first milestone is not called SLAM. Later work can add
SLAM concepts—incremental pose estimation, loop closure, and pose graphs—and
compare them against a stable geometric baseline.

## ADR-002: Preserve Raw Video and Fingerprint Inputs

**Decision:** Raw recordings remain immutable and are identified by SHA-256.

**Why:** Reconstruction settings are meaningless if experiments silently use
different source videos. A fingerprint makes results traceable and prevents
accidental dataset substitution.

## ADR-003: Begin With Deterministic Time-Based Frames

**Decision:** The first baseline samples at a requested rate, defaulting to
3 FPS, and reports quality without silently dropping difficult frames.

**Why:** A deterministic baseline is easy to reproduce and diagnose. Adaptive
selection will be introduced as an experiment and compared against this
baseline rather than becoming an unmeasured heuristic.

## ADR-004: Separate Calibration From Validation

**Decision:** Use the visible 4 ft tape only to calculate scale. Use door,
hallway, and room measurements to evaluate accuracy.

**Why:** Calibrating and evaluating on the same distances produces a circular
accuracy claim.

## ADR-005: Use the Raspberry Pi as a Measured Front End

**Decision:** The Raspberry Pi 5 initially handles capture, feature extraction,
telemetry, and transfer experiments—not dense reconstruction.

**Why:** This creates a credible embedded-systems contribution with measurable
latency, throughput, memory, CPU, and temperature constraints.

## ADR-006: Separate Matching From Mapper Experiments

**Decision:** Persist the closed COLMAP feature/match database before mapping
and provide explicit `baseline` and `fast` mapper profiles.

**Why:** Feature extraction and matching are deterministic, expensive stages
that do not need to be repeated when testing bundle-adjustment tradeoffs. The
baseline profile prioritizes geometry quality. The fast profile reduces local
and global optimization work for shorter experimental feedback loops.

**Consequence:** Fast-profile outputs are exploratory until their registration
rate, reprojection error, loop-closure drift, and dimensional accuracy are
compared quantitatively against the baseline.

## ADR-007: Use Adaptive Keyframes as a Measured Optimization

**Decision:** Select keyframes using sharpness, tracked-feature overlap,
normalized optical-flow motion, and bounded temporal coverage. Preserve a
deterministic 5 FPS candidate set as the comparison baseline.

**Why:** Uniform sampling spends compute on near-duplicate and blurred frames.
Adaptive selection reduced the first dataset from 870 to 275 images and reduced
global mapping time from 11.7 minutes to 2.0 minutes.

**Consequence:** Every decision is written to CSV/JSON, including rejected
frames and forced coverage fallbacks. Speed claims must include registration
and continuity metrics, not runtime alone.

## ADR-008: Do Not Densify Disconnected Environment Geometry

**Decision:** Block dense reconstruction and metric scaling when the hallway
and room do not share one validated sparse coordinate frame.

**Why:** Independent sparse components can each look plausible while having an
unknown relative scale, rotation, and translation. Combining them visually
would not produce a defensible environment map.

**Consequence:** `hallway_room_001` is retained as a documented observability
failure. Its point clouds are diagnostic artifacts, not a completed room model.
See `hallway_room_001_results.md` for the quantitative evidence.

## ADR-009: Permit an Explicitly Constrained Visualization Stitch

**Decision:** Produce one provisional metric map by independently scaling each
component from the visible 4 ft tape and aligning consecutive boundary camera
poses.

**Why:** This preserves usable geometry and provides one viewable artifact
without inventing image correspondences or claiming joint bundle adjustment.

**Consequence:** The report includes both camera-pose continuity and local
geometry overlap. A zero camera-pose gap does not pass the map-quality gate
when nearby sparse surfaces fail to overlap.
