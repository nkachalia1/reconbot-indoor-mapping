# hallway_room_001 Dense Baseline QA

## Decision

The CUDA dense pipeline completed, but the result is **not an accepted
continuous map**. It is retained as a reproducible negative baseline. Dense
meshing generated plausible local surfaces but could not repair the missing
cross-seam image geometry identified during sparse reconstruction.

## Reconstruction Profile

| Parameter | Value |
| --- | ---: |
| Registered main-component images | 494 |
| Registered secondary-component images | 139 |
| Maximum image dimension | 720 px |
| PatchMatch iterations | 2 |
| Fusion voxel | 0.02 m |
| Poisson depth | 8 |

## Output Metrics

| Metric | Result |
| --- | ---: |
| Fused dense points | 145,116 |
| Mesh vertices | 82,959 |
| Mesh triangles | 165,022 |
| Mesh surface area | 515.28 m2 |
| Mesh bounds | 24.82 x 20.87 x 22.73 m |
| Bounding-box diagonal | 39.61 m |
| Connected components | 25 |
| Components with at least 1% of vertices | 4 |
| Largest component | 57.2% of vertices |
| Second component | 19.6% of vertices |
| Third component | 18.1% of vertices |
| Largest-two centroid separation | 5.72 m |
| Largest-two sampled surface gap | 0.042 m |
| Boundary edges | 926 |
| Non-manifold edges | 22 |
| Degenerate triangles | 0 |

The two largest surfaces approach within 4.2 cm at one sampled location, but
they are topologically disconnected. The large visual separation is therefore
not a single uniform gap: the model contains several fragmented surfaces plus
an underconstrained global component transform. The 39.61 m diagonal is not a
credible physical dimension of the hallway-and-room route.

## Gate Results

| Gate | Result |
| --- | --- |
| Largest component >= 95% | fail (57.2%) |
| Second component <= 5% | fail (19.6%) |
| No non-manifold edges | fail (22) |
| No degenerate triangles | pass (0) |
| Sparse seam median <= 0.25 m | fail (15.06 m) |
| Sparse seam samples within 0.25 m | fail (0%) |

The topology metrics independently agree with the sparse seam test and visual
inspection. The GLB is appropriate for demonstrating the pipeline and its
failure detection, not for dimension claims or navigation.

## Root Cause

The source video loses trackable parallax through blank walls and doorway
transitions. Independent sparse components have no verified cross-seam image
correspondences and cannot be jointly bundle-adjusted. Poisson reconstruction
then closes locally supported surfaces into rounded shells, which makes the
failure look denser without making it more correct.

## Artifacts

All files are under `outputs/hallway_room_001/dense_colab_v1`:

- `hallway_room_001_dense_metric.ply`
- `hallway_room_001_mesh_metric.ply`
- `hallway_room_001_mesh_metric.obj`
- `hallway_room_001_mesh_metric.glb`
- `dense_reconstruction_summary.json`
- `dense_mesh_qa_report.json`
- `dense_components.png`

The next controlled capture and acceptance gates are specified in
`docs/hallway_room_002_capture_plan.md`.
