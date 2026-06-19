# Interview Guide

## Thirty-Second Summary

> I built a monocular indoor spatial reconstruction system around commodity
> iPhone video. The system fingerprints and validates datasets, reconstructs a
> hallway and room with Structure-from-Motion, estimates a camera trajectory,
> converts the model into metric units using a visible 4 ft tape reference, and
> evaluates dimensions against independent physical measurements. I use
> COLMAP as a geometric baseline, then benchmark my own frame-selection and
> feature-matching decisions and profile a Raspberry Pi 5 vision front end.

## What Is Original Engineering?

- reproducible dataset and artifact model
- input integrity and frame-quality analysis
- extraction and feature experiment framework
- COLMAP orchestration and sparse quality gates
- trajectory parsing and visualization
- scale calibration and independent error evaluation
- dense artifact publication
- Raspberry Pi performance profiling

## What Is an External Component?

COLMAP supplies mature SfM and multi-view stereo algorithms. Be explicit about
that. The engineering contribution is the complete system around it, the
experiments that challenge its defaults, and the evidence used to accept or
reject results.

## Likely Technical Discussion

- Why monocular geometry has scale ambiguity
- Essential matrices, triangulation, and bundle adjustment
- Why pure rotation and motion blur hurt reconstruction
- Sequential matching versus loop-closure candidates
- Calibration leakage and honest accuracy evaluation
- Sparse versus dense reconstruction
- Offline SfM versus real-time SLAM
- Embedded throughput and thermal constraints
