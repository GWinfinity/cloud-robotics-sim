"""仓库根 conftest：为 tests/ 与 plugins/**/tests 提供统一的运行环境。

- 把 repo root 幂等插入 sys.path[0]，使插件测试可以 `import plugins.x.y`
  而不依赖"从 repo root 运行 pytest"时 CWD 恰好入 path 的隐式行为；
- 兜底排除不应被收集的目录（构建产物、演示输出、vendored 文档）。
"""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = str(Path(__file__).resolve().parent)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

# 防御将来误收集（testpaths=["tests"] 已限制默认收集范围，这里兜显式路径场景）
collect_ignore_glob = [
    "outputs/*",
    "demos/*",
    "docs/_build/*",
]
