# Contributing to Cloud Robotics Simulation Platform

Thank you for your interest in contributing! This document provides guidelines and instructions for contributing to the project.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Workflow](#development-workflow)
- [Pull Request Process](#pull-request-process)
- [Coding Standards](#coding-standards)
- [Testing](#testing)
- [Documentation](#documentation)

## Code of Conduct

This project adheres to a [Code of Conduct](CODE_OF_CONDUCT.md). By participating, you are expected to uphold this code.

## Getting Started

### Setting Up Development Environment

1. Fork the repository on GitHub
2. Clone your fork locally:
   ```bash
   git clone https://github.com/YOUR_USERNAME/cloud-robotics-sim.git
   cd cloud-robotics-sim
   ```

3. Install in development mode:
   ```bash
   pip install -e ".[dev]"
   ```

4. Set up pre-commit hooks:
   ```bash
   pre-commit install
   ```

## Development Workflow

1. **Create a branch** for your feature or bug fix:
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make your changes** following our coding standards

3. **Run tests** to ensure nothing is broken:
   ```bash
   pytest tests/ -v
   ```

4. **Commit your changes** with a clear commit message:
   ```bash
   git commit -m "feat: add new feature description"
   ```

5. **Push to your fork**:
   ```bash
   git push origin feature/your-feature-name
   ```

6. **Open a Pull Request** on GitHub

## Pull Request Process

1. Ensure your PR description clearly describes the problem and solution
2. Include relevant issue numbers in the PR description
3. Ensure all CI checks pass
4. Request review from maintainers
5. Address review feedback promptly

### PR Title Format

We follow [Conventional Commits](https://www.conventionalcommits.org/) for PR titles:

- `feat:` New feature
- `fix:` Bug fix
- `docs:` Documentation changes
- `style:` Code style changes (formatting, missing semicolons, etc)
- `refactor:` Code refactoring
- `test:` Adding or updating tests
- `chore:` Build process or auxiliary tool changes

Examples:
```
feat: add support for UR10 robot
fix: resolve camera rendering issue in headless mode
docs: update API reference for Composer class
```

## Coding Standards

### Python Code Style

We use:
- **Black** for code formatting
- **Ruff** for linting
- **MyPy** for type checking

### Code Formatting

```bash
# Format code with Black
black src/ tests/

# Run linter
ruff check src/ tests/

# Type checking
mypy src/
```

### Guidelines

1. **Type Hints**: Use type hints for all function parameters and return values
   ```python
   def compose(
       self,
       scene: Scene,
       robot: RobotEmbodiment,
   ) -> ComposedEnvironment:
       ...
   ```

2. **Docstrings**: Use Google-style docstrings
   ```python
   def reset(self, seed: int = 0) -> tuple[dict, dict]:
       """Reset the environment.
       
       Args:
           seed: Random seed for reproducibility.
           
       Returns:
           Tuple of (observation, info).
       """
   ```

3. **Naming Conventions**:
   - Classes: `PascalCase`
   - Functions/Variables: `snake_case`
   - Constants: `UPPER_SNAKE_CASE`
   - Private: `_leading_underscore`

4. **Imports**: Group imports in order:
   - Standard library
   - Third-party packages
   - Local modules

## Testing

### Writing Tests

Tests should be placed in the `tests/` directory, mirroring the source structure:

```
tests/
├── core/
│   ├── test_composer.py
│   ├── test_scene.py
│   └── test_embodiment.py
```

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=cloud_robotics_sim

# Run specific test file
pytest tests/core/test_composer.py -v

# Run tests matching pattern
pytest -k "test_reset"
```

### Test Guidelines

1. Use descriptive test names
2. One assertion per test (when possible)
3. Use fixtures for common setup
4. Mock external dependencies

Example:
```python
def test_environment_reset_sets_seed():
    """Test that reset properly sets the random seed."""
    env = create_test_env()
    obs, info = env.reset(seed=42)
    
    assert 'seed' in info
    assert info['seed'] == 42
```

## Documentation

### Building Documentation

```bash
cd docs
make html
```

### Documentation Guidelines

1. Update README.md for user-facing changes
2. Add docstrings to all public APIs
3. Include code examples in docstrings
4. Update architecture docs for design changes

## Questions?

- Open an issue for bugs or feature requests
- Start a discussion for questions
- Join our Discord for real-time chat

Thank you for contributing! 🎉
