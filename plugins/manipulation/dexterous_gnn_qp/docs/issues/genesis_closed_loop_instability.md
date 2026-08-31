# Issue: Genesis rigid solver NaN / object slips through fingertips in LEAP hand grasp

> Cross-posted to genesis-world for investigation. This issue documents a reproducible instability when using Genesis 1.3.2 to simulate a LEAP hand grasping a small sphere, while the identical MJCF/URDF scene is stable in MuJoCo.

## Environment

| Item | Value |
|---|---|
| OS | Windows 10/11 |
| Python | 3.13 |
| `genesis-world` | 1.3.2 |
| Backend | `gs.cpu` |
| CPU | Intel Core i7-9750H |
| Scene source | `mujoco_menagerie/leap_hand/right_hand.xml` (MJCF) + `gs.morphs.Sphere` |

## Problem summary

In a dexterous-hand grasping benchmark, the same LEAP-hand / sphere configuration that is stable under MuJoCo fails under Genesis:

1. **Even with the hand held at a fixed grasp pose via `control_dofs_position`, the sphere slowly slides through the fingertips** (≈1.5–2 mm per 2 ms step).
2. When a computed-torque controller (QP-based) is applied, the object loses contact within ~10 steps, joint torques explode, and the solver throws:

```
genesis.GenesisException: Invalid constraint forces causing 'nan'.
Please decrease Rigid simulation timestep.
```

Decreasing `dt` or increasing `substeps` does not prevent the slip; it only changes the speed at which the object falls.

## Reproduction 1: minimal position-control slip

Run from the plugin root:

```bash
python docs/issues/repro_genesis_grasp_slip.py
```

The hand joints are locked at the target grasp configuration with `control_dofs_position`. The sphere is initialized inside the fingertip cage. In MuJoCo this configuration supports the 50 g sphere; in Genesis the sphere falls through the fingers within a few dozen steps.

### Observed output (Genesis 1.3.2)

```text
step=0 obj_z=0.0800
step=1 obj_z=0.0768
step=2 obj_z=0.0711
step=3 obj_z=0.0642
...
step=49 obj_z=-0.2946
```

## Reproduction 2: closed-loop benchmark NaN

Run the full benchmark with the Genesis backend:

```bash
python examples/benchmark.py --backend genesis --headless
```

It fails after ~9 steps with:

```text
genesis.GenesisException: Invalid constraint forces causing 'nan'.
Please decrease Rigid simulation timestep.
```

The MuJoCo backend completes 500 steps successfully with the same config:

```bash
python examples/benchmark.py --backend mujoco --headless
```

## What has been tried

- `dt` reduced from `0.002` to `0.001`.
- `substeps` increased to `8` (makes `timeconst` smaller but does not stop the slip).
- `RigidOptions.friction_cone` set to `gs.friction_cone.elliptic` to honor the MJCF declaration.
- Hand material friction raised to `1.0` and object friction kept at `0.8`.
- Position actuators neutralized / `set_dofs_kp(0)` to ensure the computed torque is the only control input.

None of these prevent the object from slipping through the fingertips.

## Expected behavior

With the hand locked in a closed grasp pose around the sphere, the sphere should remain supported by fingertip contacts (as it does in MuJoCo), and the closed-loop QP benchmark should complete without NaN.

## Actual behavior

The sphere slides through the fingertips, contact count drops to zero, and the subsequent control torques cause the rigid solver to produce NaN constraint forces.

## Attachments

- `repro_genesis_grasp_slip.py` — minimal standalone reproduction.
- Default grasp config: `configs/leap_sphere_grasp.yaml`.
- Full benchmark script: `examples/benchmark.py`.
