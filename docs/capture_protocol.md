# Hallway And Room Capture Protocol

## Pilot Route

Use one hallway and one connected room before attempting a whole floor:

1. Show a known-size scale marker near the starting position.
2. Begin with a five-second stationary view of the starting scene.
3. Walk down the hallway at a steady pace with visible translation.
4. Pass through the doorway without rotating in place.
5. Circle the room while keeping corners and furniture in view.
6. Exit through the same doorway and return along the hallway.
7. End at the original camera position and orientation.

Target duration: 120-180 seconds.

## Camera Technique

- Hold the camera at approximately 1.2-1.5 meters.
- Aim slightly downward so the floor-wall boundary stays visible.
- Keep motion slow enough to avoid rolling-shutter blur.
- Preserve 70-80% visual overlap between accepted keyframes.
- Translate while turning; pure rotation provides little triangulation baseline.
- Pause briefly before and after doorways, but do not stop for long periods.

## Scene Preparation

- Use diffuse, stable lighting and avoid exposure changes near windows.
- Open doors fully and prevent people or pets from moving through the route.
- Add removable textured markers near blank walls or repetitive hallway sections.
- Avoid mirrors, glossy cabinets, active screens, and moving curtains.
- Measure one rigid reference distance for global scale validation.

## Abort Conditions

Repeat the recording when the live quality layer reports sustained blur, the
camera loses overlap through a doorway, the route does not return to its start,
or major objects move during capture.
