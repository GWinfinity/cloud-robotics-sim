# do-as-i-do Reproduction Plugin for genesis-cloud-sim

This plugin reproduces the end-to-end pipeline of
[Do as I Do: Dexterous Manipulation Data from Everyday Human Videos](https://github.com/GWinfinity/do-as-i-do)
on `genesis-cloud-sim`.

Because the original pipeline depends on heavy external modules
(SAM3, SAM-3D-Objects, MoGe, HaWoR, TAPIR, MuJoCo Warp) and on a
real bimanual platform (UR3e + Sharpa Wave hands), this scaffold:

* uses **locally available UR3 arms** with either **Allegro Hands** (default) or **Sharpa Wave hands**;
* keeps the real reconstruction / retargeting modules behind a **pluggable interface**;
* generates **synthetic demonstration data** so the full loop can run on CPU-only machines.

```
video / demo input
        |
        v
  reconstruction (synthetic stub by default)
        |
        v
   retargeting (Genesis IK + hand mapping)
        |
        v
  simulation replay (Genesis dual-arm digital twin)
        |
        v
   deployment stub (JSON / URScript placeholders)
```

## Quick Start

```bash
# From genesis-cloud-sim root
python plugins/do_as_i_do/examples/demo_synthetic.py --num-frames 60
```

This runs the full pipeline and writes artifacts to `outputs/do_as_i_do/`.

## Files

| Path | Purpose |
|------|---------|
| `core/data.py` | `DemoSequence`, `RobotTrajectory`, JSON/NPZ I/O |
| `core/assets.py` | UR3 + Allegro/Sharpa URDF loader / merger / path resolver |
| `core/reconstruction_stub.py` | Pluggable reconstruction stage (synthetic by default) |
| `core/retargeting.py` | IK-based arm retargeting + Allegro/Sharpa hand mapping |
| `core/env.py` | `DoAsIDoEnv` Genesis simulation environment |
| `core/deployment_stub.py` | Trajectory export / URScript placeholders |
| `core/pipeline.py` | `DoAsIDoPipeline` end-to-end orchestrator |
| `configs/do_as_i_do.yaml` | Default configuration |
| `examples/` | Runnable demos |
| `tests/` | Pytest tests |

## Running Tests

```bash
pytest plugins/do_as_i_do/tests -v
```

## Using Sharpa Wave Hands

If you have the `sharpa-urdf-usd-xml` assets under
`plugins/do_as_i_do/assets/sharpa-urdf-usd-xml-main`, switch the scaffold to
Sharpa Wave hands by setting in `configs/do_as_i_do.yaml`:

```yaml
robot:
  hand_type: sharpa
```

or pass `--hand-type sharpa` to the example script:

```bash
python plugins/do_as_i_do/examples/demo_synthetic.py --hand-type sharpa --num-frames 60
```

This auto-generates `plugins/do_as_i_do/assets/robots/dual_ur3_sharpa.urdf`
(56 DOFs) and uses the 22-DOF Sharpa joint naming convention for retargeting
and deployment export.

## Using Real Hardware Assets

You can also override the auto-generated robot with a custom URDF:

```yaml
robot:
  urdf_path: /path/to/dual_ur3e_sharpa.urdf
```

Then update the joint/link names in `core/retargeting.py` if they differ from
the defaults.

## Enabling the Original Reconstruction Pipeline

On a machine with GPU, network access, and the original repository cloned:

```yaml
reconstruction:
  backend: original
  original_repo_path: /path/to/GWinfinity/do-as-i-do
```

Then implement the external calls inside `OriginalReconstructionStage.run()`.

## Limitations

* CPU-only / no external network in the current environment: SAM3/HaWoR/TAPIR are stubs.
* The retargeting sampler is simplified; the full MPPI optimizer from the paper
  is not implemented.
