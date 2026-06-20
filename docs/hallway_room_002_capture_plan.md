# hallway_room_002 Capture and Acceptance Plan

## Objective

Produce one jointly optimized, metrically scaled hallway-and-room model from a
single iPhone 14 video. This capture replaces missing doorway and blank-wall
observability; it does not change the environment-mapping project into an
object-reconstruction demo.

## Prepare the Route

1. Use the same hallway, connected room, and closed return route.
2. Remove moving people, pets, opening doors, mirrors, and reflective clutter.
3. Place distinct textured cards or printed markers every 1-1.5 m along blank
   walls and on both sides of the doorway. Avoid repeating patterns.
4. Place the 0-4 ft tape reference near the start, visible from several angles.
5. Independently measure and record the hallway width, door width, and one room
   dimension. These are validation dimensions, not scale anchors.

## Camera and Motion

- Record landscape 1080p at 30 FPS with stable lighting and a clean lens.
- Lock focus and exposure when the camera application permits it.
- Target 120-180 seconds and approximately 0.2-0.4 m/s translation.
- Keep floor-wall and wall-ceiling boundaries visible together.
- Do not pivot in place. Translate at least 0.3 m while changing heading.
- Begin and end with the same view held still for five seconds.

At the doorway, start the transition about 2 m before the threshold. Keep one
doorframe edge, a textured marker, and room furniture visible in the same views
while translating through. Repeat this overlap chain on the return pass.

## Preflight Gates

Do not start an expensive reconstruction unless all checks pass:

| Gate | Required |
| --- | ---: |
| Complete video decode | 100% |
| Five-second temporal bins represented | 100% |
| Maximum selected-keyframe gap | <= 1.2 s |
| Median adjacent-frame track ratio | >= 0.50 |
| Longest low-track interval | <= 1.5 s |
| Persistently blurred capture frames | < 10% |
| Doorway visible with translation in both directions | manual pass |

## Sparse Acceptance Gates

| Gate | Required |
| --- | ---: |
| Joint sparse models | exactly 1 |
| Selected keyframes registered | >= 80% |
| Missing registered interval | <= 2.0 s |
| Position closure error | <= 5% of path length |
| Orientation closure error | <= 15 degrees |
| Maximum camera step | <= 10x median step |
| Verified correspondences across each doorway pass | required |

Dense reconstruction remains blocked until the joint-model and doorway gates
pass. Independent components must not be pose-constrained into an accepted map.

## Dense and Metric Acceptance Gates

| Gate | Required |
| --- | ---: |
| Largest mesh component | >= 95% of vertices |
| Second mesh component | <= 5% of vertices |
| Median local seam distance | <= 0.25 m |
| Seam samples within 0.25 m | >= 50% |
| Independent dimension median error | <= 5% |
| Independent dimension maximum error | <= 10% |

The 4 ft tape sets global scale. Door width, hallway width, and room dimension
must remain held-out checks. Report every failed gate alongside the model.

## Deliverables

- Versioned capture manifest and complete video quality report
- Adaptive keyframe CSV with blur, motion, track, and coverage statistics
- One jointly optimized sparse point cloud and camera trajectory
- Metric dense cloud, mesh, GLB, and component-integrity report
- Dimension table with actual, estimated, absolute, and percentage error
- Short failure analysis if any acceptance gate does not pass
