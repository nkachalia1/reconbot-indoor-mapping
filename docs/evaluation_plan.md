# Evaluation Plan

## Dataset Integrity

- SHA-256 fingerprint of the source video
- resolution, orientation, FPS, duration, and complete decode status
- exact frame-extraction settings and output count

## Sparse Reconstruction

- registered images / selected images
- sparse 3D point count
- mean and median reprojection error
- track length distribution
- camera trajectory length
- start/end distance after metric scaling
- validated loop closures

## Metric Accuracy

Use the 4 ft tape as the calibration anchor. Reserve at least three independent
measurements for evaluation:

| Dimension | Role |
| --- | --- |
| 0–4 ft tape segment | scale calibration |
| door width | validation |
| hallway width | validation |
| room wall length | validation |

For each validation measurement report actual meters, estimated meters,
absolute error, and percent error. Report the mean and worst percent error.

## Feature Experiments

Compare ORB, SIFT, and eventually SuperPoint/LightGlue using:

- keypoints per image
- verified matches per image pair
- matching runtime
- registered-image ratio
- reprojection error
- dimensional error
- total processing time

## Raspberry Pi 5

The Pi is a front-end and systems experiment, not the first reconstruction
worker. Profile frame decoding, feature extraction, upload/stream throughput,
CPU utilization, memory, temperature, and end-to-end latency.
