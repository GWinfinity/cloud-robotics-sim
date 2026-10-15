# Testing Guide

测试布局、运行方式与 CI 约定。

## 测试布局

| 位置 | 数量 | CI 覆盖 |
| --- | --- | --- |
| `tests/` | ~95 文件 / ~1100 函数 | ✅ 主 Test job（Python 3.10/3.11/3.12 矩阵 + coverage ≥70%） |
| `plugins/**/tests/` | ~50 文件 / ~450 函数 | ✅ plugin-tests job（Python 3.12，白名单制，见下） |
| 根目录 `test_genesis_works.py` | 2 函数 | 手动冒烟用：`pytest test_genesis_works.py -v` |

收集规则：`pyproject.toml` 设了 `testpaths = ["tests"]` 与
`python_files = ["test_*.py"]`——后者意味着 `*_test.py`（如
`examples/ab_test.py` 模板、`speed_test.py` 脚本）**不会被 pytest 收集**，
它们是示例/脚本，不是测试。

## 本地运行

```bash
# 主测试套件（与 CI 等价，去掉 coverage 参数）
uv run python -m pytest tests/ -q

# 冒烟测试（验证 genesis 环境）
uv run python -m pytest test_genesis_works.py -v

# 单个插件的测试（必须单独进程：多数 genesis 插件在模块级 gs.init）
uv run python -m pytest plugins/scenes/wfc_scenes/tests -q

# 一次跑多个插件（每目录一个进程，与 CI 行为一致）
bash scripts/check_plugin_tests.sh
```

## 插件测试入 CI 的流程

CI 的 `plugin-tests` job 按 `scripts/plugin_test_dirs.txt`（每行一个目录）
逐个目录起一个 pytest 进程。入选标准：

1. **本机实跑全绿**：`uv run python -m pytest <dir> -q -m "not slow"`；
2. **只依赖核心依赖**（CI 用 `pip install -e ".[dev]"`，无插件 extras——
   可选推理库如 ultralytics/sam2 必须用 `pytest.importorskip` 或
   `tests/optional_deps.py` 守护）；
3. **无网络访问**（外部 API 一律 mock，参考 `tests/asset_gen/`）；
4. CPU 可跑，或重测试已打 `pytest.mark.slow`（job 带 `-m "not slow"`）。

满足后把目录追加到 `scripts/plugin_test_dirs.txt` 即可。

已知未入 CI 的插件测试（本机实跑失败，原因记录于此；修复后可移入
`scripts/plugin_test_dirs.txt`）：

| 目录 | 失败原因（2026-09 本机实跑） |
| --- | --- |
| `plugins/teleop/vr_bridge/tests` | `test_e2e_genesis.py::test_genesis_franka_teleop_and_recording` 环境相关失败（其余全绿） |
| `plugins/do_as_i_do/tests` | 3 个 FileNotFoundError（缺 teleop/sharpa 资产） |
| `plugins/manipulation/dexterous_gnn_qp/tests` | `benchmark.py` 环境相关断言失败 |

其余未列入的控制器训练类等插件测试，按上方流程验证后可补入。

## 约定

- **可选依赖守护**：新测试文件从 `tests/optional_deps.py` 取
  `genesis_only` / `torch_only` 等 skip 标记；已有文件的
  `try/except HAS_*` 与 `pytest.importorskip` 风格继续有效，不必迁移。
- **禁网络**：测试不允许真实 HTTP/下载；用 monkeypatch/mock（资产下载
  mock `download_component`/`ensure_for_path`，参考 `tests/robotwin/`）。
- **slow marker**：超过几十秒的测试打 `@pytest.mark.slow`，CI 的
  plugin-tests job 用 `-m "not slow"` 跳过。`gpu` marker 已注册留给
  将来 GPU runner。
- **sys.path**：仓库根 `conftest.py` 已把 repo root 加入 `sys.path`，
  插件测试直接 `import plugins.x.y` / `from plugins...` 即可，不要
  再在测试文件里 `sys.path.insert`。
- **genesis 进程隔离**：`gs.init` 每个进程一次。同一插件的 genesis 测试
  保持在同一测试模块内；跨插件必须分进程（CI 已按目录分进程）。

## Quality gates（与 AGENTS.md 一致）

```bash
uv run python -m ruff check src/ tests/
uv run python -m black --check src/ tests/
uv run python -m mypy src/cloud_robotics_sim
uv run python -m pytest tests/
```
