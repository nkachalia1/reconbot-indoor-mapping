# hallway_room_001 Sparse Reconstruction Results

## Outcome

The capture produces two geometrically valid sparse components, but it does not
yet support a trustworthy unified hallway-and-room model. The primary component
contains the outbound and return hallway views. A focused solve recovers the
room as a second component. The components have no shared registered cameras,
so a similarity transform between them is underconstrained.

Dense reconstruction and metric scaling were initially blocked as accepted
deliverables for this dataset. A later, explicitly low-confidence run exercised
the complete CUDA dense pipeline and quantified the expected fragmentation. See
docs/hallway_room_001_dense_baseline.md for topology metrics and gate results.

## Quantitative Experiments

| Experiment | Input images | Runtime | Registered | Sparse points | Largest missing interval |
| --- | ---: | ---: | ---: | ---: | --- |
| Incremental COLMAP baseline | 870 | 82.1 min mapper | 467 (53.7%) in largest model | 114,905 | frames 312–714 |
| Global COLMAP baseline | 870 | 9.5 min | 494 (56.8%) | 137,260 | frames 330–705 |
| Guided rematch + global solve | 870 | 7.9 + 11.7 min | 494 (56.8%) | 135,038 | frames 330–705 |
| Adaptive global solve | 275 | 2.0 min | 183 (66.5%) | 50,514 | selected keyframes 123–214 |
| Focused room solve | 531 eligible | 35.8 s | 139 | 20,351 | frames 564–605 and 609–705 |

The adaptive solve is about 6x faster than the full global solve and 42x faster
than the original incremental mapper. It improves registration rate among the
submitted keyframes, but it does not repair missing parallax.

## Adaptive Keyframe Report

- Candidate frames: 870 at 5 FPS
- Selected keyframes: 275 (31.6%)
- Effective keyframe rate: 1.576 FPS
- Five-second coverage bins: 35/35 covered
- Maximum selected-frame gap: 1.4 s
- Median selected-frame track ratio: 0.768
- Motion-selected frames: 232
- Coverage fallback frames: 41
- Blur rejections: 115
- Low-overlap rejections: 45
- Low-feature rejections: 12

Every selected frame retains its original filename, sequence, and timestamp in
`keyframes.csv`. `colmap_image_list.txt` allows mapper experiments to reuse the
existing feature database without recomputing SIFT descriptors or matches.

## Root-Cause Evidence

Representative frames in the missing interval contain long views of blank
walls, close-up furniture, motion blur, and camera rotations with little
translation. Pairwise SIFT matches exist, but the views do not provide a stable
chain of triangulated 3D points through the room-to-hallway transition.

Three attempted remedies did not improve unified coverage:

1. Guided rematching added approximately 363,000 inlier correspondences.
2. Relaxing global rotation filtering from 10 to 30 degrees retained the exact
   same 139-camera room component.
3. Relaxed image registration added zero cameras because candidate views saw no
   existing room-model 3D points.

This is a capture-observability failure, not a compute-capacity failure.

## Next Capture Changes

1. Record in landscape and lock focus/exposure when possible.
2. Keep floor-wall, wall-ceiling, and doorway boundaries visible together.
3. Translate laterally or forward while turning; avoid pivoting in place.
4. Keep textured furniture edges in view across each doorway transition.
5. Walk more slowly near frames that would otherwise show mostly blank walls.
6. Place the 4 ft scale reference near the start without blocking the route.
7. Record independent hallway width, door width, and room dimensions for
   validation; do not reuse the scale anchor as an accuracy measurement.

The next dataset should be named `hallway_room_002` so this failed-but-useful
experiment remains reproducible.

## Provisional Continuous Metric Map

The usable components were later combined into one coordinate system as a
**low-confidence constrained stitch**:

- Automatic tape detection used registered frames 660 and 750.
- RANSAC floor planes converted the tape endpoint pixels into 3D distances.
- Main component scale: `0.6550 m/unit`.
- Return-hallway component scale: `0.8039 m/unit`.
- Relative component scale: `1.2272`.
- Consecutive boundary camera poses 705 and 706 constrain rotation and
  translation.
- Output: 155,389 colored sparse points and 633 camera poses.

The camera seam is continuous by construction. The sparse geometry near the
textureless wall does not overlap and fails the 25 cm seam-quality gate. This
artifact is useful as a unified visualization, but it is not equivalent to a
jointly optimized SfM reconstruction.

Artifacts are in `outputs/hallway_room_001/continuous_metric_map_v1`.
