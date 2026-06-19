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
