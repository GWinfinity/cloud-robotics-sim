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

The project targets `genesis-world>=1.2.2,<1.3.0`. Do not use Genesis 0.4.x APIs.
Use the compatibility helpers in `src/cloud_robotics_sim/utils/genesis_compat.py` for backend selection and light creation.

## Quality Gates

- **Lint**: `ruff check src/ tests/`
- **Format**: `black --check src/ tests/` (or `black src/ tests/` to fix)
- **Type check**: `mypy src/cloud_robotics_sim`
- **Tests**: `pytest tests/`

CI treats all of these as blocking.

## Notes

- `plugins/envs/sky/core/genesis/` is a vendored old Genesis 0.3.11 copy. Do not import it from core code.
- Keep `uv.lock` in sync with `pyproject.toml` by running `uv lock` after dependency changes.
