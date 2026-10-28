# Agent Development Notes

## 环境与依赖

- 使用 `uv` 管理依赖，锁文件为 `uv.lock`；修改 `pyproject.toml` 中的依赖后必须运行 `uv lock`。
- 使用项目虚拟环境，避免使用系统 Python 或外部虚拟环境。Windows 解释器路径为 `.venv/Scripts/python.exe`。
- 支持 Python `>=3.10,<3.14`；CI 覆盖 3.10、3.11 和 3.12。
- 项目使用 `genesis-world==1.4.5`（quadrants 1.3.3），不要引入 Genesis 0.4.x API，也不要从核心代码导入 `plugins/envs/sky/core/genesis/` 中 vendored 的旧版本。
- `genesis-world 1.4.5` 移除了自定义的 `gs.Tensor`（1.4.5 #3592），宿主侧张量注解用 `torch.Tensor`；`gs.tc_float`/`gs.tc_int` 仍存在，但仅在 `gs.init()` 之后才有值（即原生 `torch.dtype`）。
- CUDA 版 torch/torchvision 经 `pyproject.toml` 中的 `pytorch-cu128` 显式索引解析（默认镜像只有 CPU 轮）；Windows 上 cu128 最高为 torch 2.11.x（2.12+ 不发 Windows CUDA 轮）。
- Genesis 后端选择、光源创建和外力施加应使用 `src/cloud_robotics_sim/utils/genesis_compat.py` 中的兼容接口。
- 可选依赖必须延迟导入或加保护，缺失时明确降级或报错，不得破坏基础包的导入。新增测试的跳过标记统一放在 `tests/optional_deps.py`。
- 安装开发依赖：`uv sync --extra dev`。

## 质量要求

所有变更必须保持以下 CI 门禁通过：

```bash
uv run python -m ruff check src/ tests/
uv run python -m black --check src/ tests/
uv run python -m mypy src/cloud_robotics_sim
uv run python -m pytest tests/
```

- 遵循现有代码结构、命名和接口约定；修复问题时同步更新失效的注释、文档和测试。
- 不得通过删除测试、放宽断言或静默跳过失败来规避问题。可选依赖缺失导致的跳过必须与无关测试失败区分。
- 测试不得访问网络；下载、HTTP API 和外部服务必须 mock。慢速测试使用 `slow` 标记。
- 插件测试按 `scripts/plugin_test_dirs.txt` 的白名单分目录、分进程运行，并使用 `-m "not slow"`；Genesis 插件需遵守每进程一次 `gs.init` 的约束。
- 测试和实现不得依赖本机 CUDA overlay、特定 GPU、历史基准结果或未纳入版本控制的资产。
- 验证应覆盖实际行为，而不只是导入或编译；无法执行的检查必须明确说明。

## 安全要求

- 不得读取、打印、提交或外传密钥、令牌、`.env`、私钥及其他凭据。凭据只能通过受支持的环境变量或安全配置注入，测试使用虚构值。
- 不得让测试、示例或默认代码路径隐式访问生产服务、提交云端任务或下载大型资产；网络与下载行为必须显式、可禁用并可 mock。
- 不得将凭据、生成的大型资产、缓存、仿真输出或实验结果提交到版本控制。
- 模拟设备的写操作必须经过类型检查和 `WritePrimitive` 安全边界验证，越界必须抛出 `SafetyViolationError`，不得绕过。
- 修改设备或故障模型时，不得削弱门锁、过温切断、过载/断裂停机、力限制等保护；安全联锁与故障注入逻辑必须有对应测试。
- 安全相关元数据和 GB 标准合规信息必须与实际行为一致，不得用元数据代替运行时保护。
- 喷嘴耦合插件只接受用户提供的仿真参数；不得加入推进剂配方、发动机制造数据或其他操作性武器设计资料。

历史排障与性能测量记录见 [`docs/troubleshooting.md`](docs/troubleshooting.md)。
