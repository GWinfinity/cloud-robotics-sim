# dexterous_gnn_qp

LIFT Phase-1 plugin for dexterous-hand grasping via hierarchical QP.

## Structure

- `core/` – QP solvers, MuJoCo scene builder, PD controllers, skeleton selection, metrics.
- `configs/` – YAML configs for LEAP Hand / sphere grasp and additional hands.
- `examples/` – Standalone demos and benchmark.
- `tests/` – Unit and smoke tests.

## Quick start

Run from the plugin directory (or add `plugins/manipulation` to `PYTHONPATH`):

```bash
python examples/full_qp_baseline.py
python examples/hierarchical_qp.py
python examples/benchmark.py --headless --output data/benchmark.csv
```

## Tests

```bash
python -m unittest discover -s tests -v
```

## Notes

- Both MuJoCo and Genesis backends are supported; select with `--backend mujoco|genesis` or `sim.backend` in the config.
- The hierarchical QP implements the full alternating iteration from the LIFT paper (skeleton-layer QP + edge-force update + friction-cone projection).
- Asset paths in configs are resolved relative to this plugin root.
