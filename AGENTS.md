# Agent Development Notes

## Environment

This project uses `uv` for dependency management. The lockfile is `uv.lock`.

```bash
# Sync dependencies (including dev tools)
uv sync --extra dev

# Run commands inside the managed venv
uv run python -m pytest tests/
uv run python -m ruff check src/ tests/
uv run python -m black --check src/ tests/
uv run python -m mypy src/cloud_robotics_sim
```

On Windows, the venv interpreter is at `.venv/Scripts/python.exe`. Avoid using the system Python or external virtual environments.

## Supported Python Versions

- `requires-python = ">=3.10,<3.14"`
- CI tests Python 3.10, 3.11, and 3.12.
- Local development may use Python 3.13, but it is not yet in CI.

## Genesis Version

The project pins `genesis-world==1.3.2` (chosen for its deterministic CPU/GPU simulation, improved ill-conditioned mass-matrix robustness, and islands support). Do not use Genesis 0.4.x APIs.
Use the compatibility helpers in `src/cloud_robotics_sim/utils/genesis_compat.py` for backend selection and light creation.

## Quality Gates

- **Lint**: `ruff check src/ tests/`
- **Format**: `black --check src/ tests/` (or `black src/ tests/` to fix)
- **Type check**: `mypy src/cloud_robotics_sim`
- **Tests**: `pytest tests/`

CI treats all of these as blocking.

## Notes

- Deployment installs should use `tools/install_torch.py` to pick the right PyTorch wheels: it auto-detects Moore Threads MUSA hardware (installs `torch_musa`), CUDA, or CPU, and switches to Aliyun mirrors when the official PyTorch index is unreachable (mainland China). Docker builds expose this via the `TORCH_BACKEND` and `CHINA_MIRROR` build arguments.
- RoboTwin 2.0 assets (~4GB, gitignored under `assets/robotwin/`) are checked and downloaded automatically on first use via `src/cloud_robotics_sim/robotwin/assets.py` (HF `TianxingChen/RoboTwin2.0`). Prefetch with `python -m cloud_robotics_sim.robotwin.assets`; set `CRS_ROBOTWIN_AUTO_DOWNLOAD=0` to disable and `HF_ENDPOINT=https://hf-mirror.com` for the China mirror. Tests that hit the network are not allowed — mock `download_component`/`ensure_for_path` instead.
- `plugins/envs/sky/core/genesis/` is a vendored old Genesis 0.3.11 copy. Do not import it from core code.
- Keep `uv.lock` in sync with `pyproject.toml` by running `uv lock` after dependency changes.
