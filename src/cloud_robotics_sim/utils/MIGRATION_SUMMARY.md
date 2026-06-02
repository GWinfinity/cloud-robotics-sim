# Phase 2: ManiSkill Utils Integration - Migration Summary

## Overview

This document summarizes the integration of ManiSkill's `genesis_utils.py` into the `genesis-cloud-sim` framework's utils module.

## Source File

Original file from ManiSkill-main:
- `ManiSkill-main/mani_skill/utils/genesis_utils.py` (729 lines)

## Target Structure

```
genesis-cloud-sim/src/cloud_robotics_sim/utils/
├── __init__.py                      # Main exports and compatibility layer
├── genesis_compat.py                # Core Genesis utilities (~600 lines)
├── camera.py                        # Camera utilities (~400 lines)
├── rendering.py                     # Rendering utilities (~450 lines)
└── MIGRATION_SUMMARY.md             # This file
```

## Code Organization

### 1. genesis_compat.py

**Core utilities from ManiSkill's genesis_utils.py:**

- **Object Query** (lines 86-145 in original)
  - `get_obj_by_name()` - Find object by name
  - `get_objs_by_names()` - Find multiple objects by names
  - `get_obj_by_type()` - Find object by type

- **URDF Configuration** (lines 152-250 in original)
  - `check_urdf_config()` - Validate URDF config
  - `parse_urdf_config()` - Parse config dict
  - `apply_urdf_config()` - Apply config to loader

- **State Extraction** (lines 256-329 in original)
  - `get_actor_state()` - Get actor pose, velocity
  - `get_articulation_state()` - Get full articulation state
  - `get_articulation_padded_state()` - Padded state for batching

- **Contact Processing** (lines 335-463 in original)
  - `get_pairwise_contacts()` - Get contacts between two actors
  - `get_multiple_pairwise_contacts()` - Get contacts with multiple actors
  - `compute_total_impulse()` - Compute total contact impulse
  - `get_pairwise_contact_impulse()` - Get impulse between two actors
  - `get_cpu_actor_contacts()` - Get all contacts for an actor
  - `get_cpu_actors_contacts()` - Get contacts for multiple actors

- **Joint and Actor Utilities** (lines 608-680 in original)
  - `check_joint_stuck()` - Detect stuck joints
  - `check_actor_static()` - Check if actor is static
  - `is_state_dict_consistent()` - Check batch dimension consistency

- **Compatibility Types**
  - `Pose` class - Pose representation
  - `matrix_to_quaternion()` - Rotation conversion
  - `GENESIS_RENDER_SYSTEM` - Version constant

### 2. camera.py

**Camera-related utilities (lines 470-540 + enhancements from original):**

- `genesis_pose_to_opencv_extrinsic()` - Convert Genesis pose to OpenCV format
- `look_at()` - Compute camera pose from eye/target/up vectors
- `hex2rgba()` - Convert hex color to RGBA
- `rgba2hex()` - Convert RGBA to hex (new addition)
- `spherical_to_cartesian()` - Spherical to cartesian coordinates (new)
- `compute_fovy()` - Compute field of view (new)
- `get_camera_rays()` - Compute per-pixel camera rays (new)
- `create_viewer()` - Create Genesis viewer with configuration

### 3. rendering.py

**Rendering utilities (lines 561-601 + enhancements from original):**

- `set_render_material()` - Set material properties
- `set_articulation_render_material()` - Set material for entire articulation
- `set_entity_color()` - Set color for entity (new)
- `ShaderConfig` class - Shader configuration container (new)
- `configure_rendering()` - Global rendering configuration (new)
- `load_texture()` - Load texture from file (new)
- `create_checkerboard_texture()` - Generate checkerboard pattern (new)
- `save_screenshot()` - Save screenshot to file (new)
- `start_recording()` / `stop_recording()` - Video recording helpers (new)

## Key Changes

### 1. Modular Organization

**Original (ManiSkill single file):**
```python
from mani_skill.utils.genesis_utils import (
    get_obj_by_name,
    look_at,
    set_render_material,
)
```

**New (genesis-cloud-sim modular):**
```python
# Import all from main utils
from cloud_robotics_sim.utils import (
    get_obj_by_name,
    look_at,
    set_render_material,
)

# Or import from specific modules
from cloud_robotics_sim.utils.camera import look_at
from cloud_robotics_sim.utils.rendering import set_render_material
```

### 2. Enhanced Compatibility

- **Graceful Degradation**: All heavy dependencies (numpy, torch, genesis) are optional
- **Type Hints**: Added proper type hints throughout
- **Documentation**: Added comprehensive docstrings
- **Error Handling**: Added try-except blocks for optional dependencies

### 3. New Features Added

- **New camera functions**: `spherical_to_cartesian()`, `compute_fovy()`, `get_camera_rays()`
- **New rendering functions**: `set_entity_color()`, `ShaderConfig` class, texture utilities
- **Enhanced color utilities**: Added `rgba2hex()` complement to `hex2rgba()`
- **Recording utilities**: Screenshot and video recording helpers

### 4. Backward Compatibility

Maintained API compatibility with ManiSkill:

```python
# Old (ManiSkill)
from mani_skill.utils.genesis_utils import get_obj_by_name

# New (genesis-cloud-sim) - Same API
def get_entity_by_name(objs, name, is_unique=True):
    """Alias for get_obj_by_name for ManiSkill compatibility."""
    return get_obj_by_name(objs, name, is_unique)
```

## Code Statistics

| Metric | Value |
|--------|-------|
| Original file | 1 file (~729 lines) |
| New files | 4 files (~1,450 lines) |
| Test coverage | 8 validation tests |
| New functions | ~15 additional utilities |

## API Compatibility

### Maintained APIs (from ManiSkill)

All original APIs are preserved:

```python
# Object query
get_obj_by_name(objs, name, is_unique=True)
get_objs_by_names(objs, names)
get_obj_by_type(objs, target_type, is_unique=True)

# URDF config
check_urdf_config(urdf_config)
parse_urdf_config(config_dict)
apply_urdf_config(loader, urdf_config)

# State extraction
get_actor_state(actor)
get_articulation_state(articulation)
get_articulation_padded_state(articulation, max_dof)

# Contact processing
get_pairwise_contacts(contacts, actor0, actor1)
compute_total_impulse(contact_infos)
get_pairwise_contact_impulse(contacts, actor0, actor1)

# Camera
look_at(eye, target, up=(0, 0, 1), device=None)
hex2rgba(h, correction=True)
genesis_pose_to_opencv_extrinsic(genesis_pose_matrix)

# Rendering
set_render_material(material, **kwargs)
set_articulation_render_material(articulation, **kwargs)

# Utilities
check_joint_stuck(articulation, active_joint_idx, ...)
check_actor_static(actor, lin_thresh=1e-3, ang_thresh=1e-2)
create_viewer(viewer_camera_config)
```

### New APIs (genesis-cloud-sim additions)

```python
# Camera
spherical_to_cartesian(radius, azimuth, elevation, target)
compute_fovy(focal_length, sensor_height)
get_camera_rays(camera_pose, intrinsics, image_size)
rgba2hex(rgba)

# Rendering
ShaderConfig(shader_pack, ray_tracing_denoiser, ...)
configure_rendering(shader_pack, enable_ray_tracing, ...)
set_entity_color(entity, color, recursive=True)
load_texture(path, **kwargs)
create_checkerboard_texture(size, check_size, ...)
save_screenshot(camera_or_viewer, path, rgb=True)
start_recording(viewer, path, fps=30)
stop_recording(recording_handle)
```

## Testing

Validation test: `test_maniskill_validation.py`

Test results:
```
✓ All imports successful
✓ get_obj_by_name works
✓ check_urdf_config works
✓ hex2rgba works
✓ rgba2hex works
✓ look_at works
✓ ShaderConfig works
✓ is_state_dict_consistent works

==================================================
All validation tests passed!
==================================================
```

Examples: `genesis-cloud-sim/examples/migration/maniskill_utils_example.py`

## Dependencies

### Required
- None (all dependencies are optional)

### Optional (for full functionality)
- `genesis-world>=0.4.0` - Physics simulation
- `numpy>=1.20.0` - Numerical operations
- `torch>=2.0.0` - Deep learning tensors
- `Pillow` - Image I/O for screenshots

## Integration with Other Modules

The utils module integrates with:
- `dreamdojo` plugin - Uses these utilities for simulator implementation
- `maniskill` plugin - Compatible with ManiSkill's data formats
- `sky` plugin - Uses the same Genesis core

## Migration Commands

For users migrating from ManiSkill:

```python
# Old (ManiSkill standalone)
from mani_skill.utils.genesis_utils import (
    get_obj_by_name,
    look_at,
    set_render_material,
)

# New (genesis-cloud-sim)
from cloud_robotics_sim.utils import (
    get_obj_by_name,
    look_at,
    set_render_material,
)
```

## Integration with Phase 1

The ManiSkill utils module is designed to work with the DreamDojo plugin from Phase 1:

```python
# Use utils with DreamDojo simulator
from cloud_robotics_sim.utils import get_actor_state
from genesis_cloud_sim.plugins.datasets.dreamdojo import GenesisSimulator

simulator = create_genesis_simulator()
state = get_actor_state(simulator.robot)
```

## Notes

1. **Graceful Degradation**: All functions work without heavy dependencies, returning None or using fallbacks
2. **Type Safety**: Added proper type hints for better IDE support
3. **Documentation**: Added comprehensive docstrings following Google style
4. **Modularity**: Split into logical modules for better maintainability
5. **Compatibility**: Maintained API compatibility with ManiSkill for easy migration

## References

- Original ManiSkill: ManiSkill-main/mani_skill/utils/genesis_utils.py
- Target: genesis-cloud-sim/src/cloud_robotics_sim/utils/
- Examples: genesis-cloud-sim/examples/migration/maniskill_utils_example.py
- Phase 1: DreamDojo integration (see plugins/datasets/dreamdojo/)
