# System Architecture

## Compute Split

### Raspberry Pi 5

- Hosts the field operator interface
- Starts and stops walkthrough recordings
- Stores session metadata and route annotations
- Reports power, temperature, memory, and network status
- Transfers completed videos to the mapping worker

### Laptop Sensor And Worker

- Captures the webcam video in native Windows Python
- Selects blur-safe, motion-distinct keyframes
- Runs COLMAP in WSL on the Linux filesystem
- Validates loop closure and trajectory drift
- Runs segmented dense reconstruction after sparse acceptance
- Publishes trajectory, point cloud, mesh, occupancy, and metrics

## Map Products

```text
trajectory.json        ordered, scaled camera poses
sparse.ply             whole-route visual landmarks
segments.json          room and hallway reconstruction chunks
dense.ply              fused environment point cloud
environment.glb        dashboard and portfolio mesh
occupancy.png/yaml     navigation-oriented 2D obstacle layer
evaluation.json        drift, coverage, accuracy, runtime, memory
```

## Why Dense Reconstruction Is Segmented

A long walkthrough can contain hundreds of registered views. Processing every
image as one dense job raises memory, runtime, and failure cost. The global
sparse solution defines one coordinate frame, while overlapping hallway/room
segments are reconstructed independently and fused back into that frame.

## Scale And Navigation

Monocular SfM recovers geometry only up to a global scale. A measured landmark
pair or known-size fiducial supplies meters per reconstruction unit. Floor-plane
estimation then aligns gravity approximately, and occupied points are projected
into a 2D grid. The occupancy product is evaluated separately from visual mesh
quality because a photorealistic surface is not automatically safe navigation
geometry.
