# Legacy demos

`run_apartment.py` and `run_single_scene.py` were the first apartment video
demos. They drive the robot with a constant dummy action (no real task), use
placeholder box prims instead of RoboTwin-OD furniture, and render at low
fixed-camera quality.

**Superseded by [`../robotwin/home_demo.py`](../robotwin/home_demo.py)**,
which performs a real planned pick-and-place task per room with narrated
video overlays and a moving track camera.

These files are kept for reference only and are not maintained.
