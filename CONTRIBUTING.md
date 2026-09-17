# 贡献指南 / Contributing Guide

**中文** | [English](#contributing-guide)

感谢您对项目的关注！本文档提供为本项目贡献代码与文档的指南和操作说明。

## 目录

- [行为准则](#行为准则)
- [快速开始](#快速开始)
- [开发流程](#开发流程)
- [Pull Request 流程](#pull-request-流程)
- [代码规范](#代码规范)
- [测试](#测试)
- [文档](#文档)

## 行为准则

本项目遵守[行为准则](CODE_OF_CONDUCT.md)。参与本项目即表示您同意遵守该准则。

## 快速开始

### 搭建开发环境

1. Fork 本仓库
2. 克隆您的 fork 到本地：
   ```bash
   git clone https://github.com/YOUR_USERNAME/cloud-robotics-sim.git
   cd cloud-robotics-sim
   ```

3. 以开发模式安装（推荐 `uv`，可复现锁定依赖）：
   ```bash
   uv sync --extra dev
   # 或：pip install -e ".[dev]"
   ```

4. 安装 pre-commit 钩子：
   ```bash
   pre-commit install
   ```

## 开发流程

1. **创建分支**：
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **进行修改**，遵循本项目的代码规范

3. **运行测试**，确保没有破坏既有功能：
   ```bash
   uv run python -m pytest tests/ -v
   ```

4. **提交更改**，使用清晰的提交信息：
   ```bash
   git commit -m "feat: add new feature description"
   ```

5. **推送到您的 fork**：
   ```bash
   git push origin feature/your-feature-name
   ```

6. 在 GitHub 上**发起 Pull Request**

## Pull Request 流程

1. PR 描述中清晰说明问题与解决方案
2. 在描述中引用相关的 issue 编号
3. 确保所有 CI 检查通过（lint / 格式 / 类型检查 / 测试均为阻塞项）
4. 请求维护者评审
5. 及时响应评审意见

### PR 标题格式

PR 标题遵循 [Conventional Commits](https://www.conventionalcommits.org/) 规范：

- `feat:` 新功能
- `fix:` 缺陷修复
- `docs:` 文档变更
- `style:` 代码风格变更（格式、分号等，不影响逻辑）
- `refactor:` 代码重构
- `test:` 新增或更新测试
- `chore:` 构建流程或辅助工具的变更

示例：
```
feat: add support for UR10 robot
fix: resolve camera rendering issue in headless mode
docs: update API reference for Composer class
```

## 代码规范

### Python 代码风格

我们使用：
- **Black** 进行代码格式化
- **Ruff** 进行静态检查
- **MyPy** 进行类型检查

### 格式化与检查

```bash
# 格式化
uv run python -m black src/ tests/

# 静态检查
uv run python -m ruff check src/ tests/

# 类型检查
uv run python -m mypy src/cloud_robotics_sim
```

### 规范细则

1. **类型标注**：所有函数参数与返回值必须标注类型
   ```python
   def compose(
       self,
       scene: Scene,
       robot: RobotEmbodiment,
   ) -> ComposedEnvironment:
       ...
   ```

2. **Docstring**：使用 Google 风格
   ```python
   def reset(self, seed: int = 0) -> tuple[dict, dict]:
       """Reset the environment.

       Args:
           seed: Random seed for reproducibility.

       Returns:
           Tuple of (observation, info).
       """
   ```

3. **命名约定**：
   - 类：`PascalCase`
   - 函数/变量：`snake_case`
   - 常量：`UPPER_SNAKE_CASE`
   - 私有成员：`_leading_underscore`

4. **导入顺序**（分组，组间空一行）：
   - 标准库
   - 第三方包
   - 本地模块

## 测试

### 编写测试

测试放在 `tests/` 目录下，目录结构与源代码对应：

```
tests/
├── core/
│   ├── test_composer.py
│   ├── test_scene.py
│   └── test_embodiment.py
```

插件的测试随插件放在 `plugins/<category>/<name>/tests/`（如
`plugins/solvers/acoustics/tests/`）；每个插件需单独运行 pytest，
因为各插件在模块级调用 `gs.init`（Genesis 不能在同进程内重复初始化）。

### 运行测试

```bash
# 全部测试
uv run python -m pytest tests/

# 带覆盖率
uv run python -m pytest tests/ --cov=cloud_robotics_sim

# 指定测试文件
uv run python -m pytest tests/core/test_composer.py -v

# 按名称模式匹配
uv run python -m pytest tests/ -k "test_reset"

# 插件测试（示例）
uv run python -m pytest plugins/solvers/acoustics/tests -v
```

### 测试准则

1. 测试名称要有描述性
2. 单个测试尽可能只断言一件事
3. 公共准备工作使用 fixture
4. 外部依赖（网络、API）一律 mock——CI 中不允许真实网络访问
5. 涉及随机性的物理测试使用固定种子并给出明确的物理量容差

示例：
```python
def test_environment_reset_sets_seed():
    """Test that reset properly sets the random seed."""
    env = create_test_env()
    obs, info = env.reset(seed=42)

    assert 'seed' in info
    assert info['seed'] == 42
```

## 文档

### 构建文档

```bash
cd docs
make html
```

### 文档准则

1. 面向用户的变更需更新 README.md
2. 所有公开 API 必须有 docstring
3. docstring 中包含代码示例
4. 设计层面的变更需更新架构文档（`docs/architecture/`）与根目录的 `AGENTS.md`（`AGENTS.md` 是面向编码代理的权威说明，修改其所述的目录结构、约定、工作流时必须同步更新）

## 有疑问？

- 提交 issue 报告缺陷或提出功能需求
- 发起 discussion 讨论问题
- 加入 Discord 实时交流

感谢您的贡献！🎉

---
---

# Contributing Guide

[中文](#贡献指南) | **English**

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

1. Fork the repository
2. Clone your fork locally:
   ```bash
   git clone https://github.com/YOUR_USERNAME/cloud-robotics-sim.git
   cd cloud-robotics-sim
   ```

3. Install in development mode (`uv` recommended for a reproducible lockfile):
   ```bash
   uv sync --extra dev
   # or: pip install -e ".[dev]"
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
   uv run python -m pytest tests/ -v
   ```

4. **Commit your changes** with a clear commit message:
   ```bash
   git commit -m "feat: add new feature description"
   ```

5. **Push to your fork**:
   ```bash
   git push origin feature/your-feature-name
   ```

6. **Open a Pull Request**

## Pull Request Process

1. Ensure your PR description clearly describes the problem and solution
2. Include relevant issue numbers in the PR description
3. Ensure all CI checks pass (lint / format / type check / tests are all blocking)
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
# Format code
uv run python -m black src/ tests/

# Run linter
uv run python -m ruff check src/ tests/

# Type checking
uv run python -m mypy src/cloud_robotics_sim
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

4. **Imports**: Group imports in order (blank line between groups):
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

Plugin tests live with the plugin under `plugins/<category>/<name>/tests/`
(e.g. `plugins/solvers/acoustics/tests/`); run pytest per plugin, because
each plugin calls `gs.init` at module level (Genesis cannot be initialized
twice in one process).

### Running Tests

```bash
# Run all tests
uv run python -m pytest tests/

# Run with coverage
uv run python -m pytest tests/ --cov=cloud_robotics_sim

# Run specific test file
uv run python -m pytest tests/core/test_composer.py -v

# Run tests matching pattern
uv run python -m pytest tests/ -k "test_reset"

# Plugin tests (example)
uv run python -m pytest plugins/solvers/acoustics/tests -v
```

### Test Guidelines

1. Use descriptive test names
2. One assertion per test (when possible)
3. Use fixtures for common setup
4. Mock external dependencies — no real network access in CI
5. For stochastic physics tests, use fixed seeds and explicit physical tolerances

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
4. Update the architecture docs (`docs/architecture/`) and the root `AGENTS.md` for design changes — `AGENTS.md` is the authoritative note for coding agents and must stay in sync with the directories, conventions and workflows it describes

## Questions?

- Open an issue for bugs or feature requests
- Start a discussion for questions
- Join our Discord for real-time chat

Thank you for contributing! 🎉
