"""vendored scene-language 程序脚本的运行入口，不是 pytest 测试。

这些文件依赖 vendored mitsuba/drjit 环境与 `engine.constants` 初始化，
被 pytest 收集必然 ImportError；显式排除。
"""

collect_ignore = [
    "test_basic.py",
    "test_shape2mesh.py",
]
