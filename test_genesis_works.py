"""Genesis 冒烟测试：init → 建 Scene → 加实体 → build → 步进。

根目录保留此文件供"刚 clone 完验证环境"时显式运行：
``pytest test_genesis_works.py -v``（testpaths=["tests"] 使默认收集不会带上它）。
"""

from __future__ import annotations

import pytest

try:
    import genesis as gs

    HAS_GENESIS = True
except ImportError:  # pragma: no cover - 极简环境
    gs = None  # type: ignore[assignment]
    HAS_GENESIS = False

from cloud_robotics_sim.utils.genesis_compat import get_genesis_backend

pytestmark = pytest.mark.skipif(not HAS_GENESIS, reason="genesis-world not installed")

_GS_READY = False


@pytest.fixture(scope="module")
def scene():
    """进程内一次 gs.init（重复调用由 Genesis 自身去重）。"""
    global _GS_READY
    if not _GS_READY:
        gs.init(backend=get_genesis_backend("cpu"), precision="32")
        _GS_READY = True
    sc = gs.Scene(
        sim_options=gs.options.SimOptions(dt=0.01, substeps=2),
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(3.0, 0.0, 3.0),
            camera_lookat=(0.0, 0.0, 0.5),
        ),
        show_viewer=False,
        vis_options=gs.options.VisOptions(show_world_frame=True),
    )
    yield sc


def test_init_cpu_backend():
    """CPU 后端可用（get_genesis_backend 返回有效 backend）。"""
    backend = get_genesis_backend("cpu")
    assert backend is not None


def test_scene_create_build_step(scene):
    """建 Scene → 加 Box → build → 步进 10 步 → 实体位姿可读。"""
    box = scene.add_entity(
        morph=gs.morphs.Box(size=(0.5, 0.5, 0.5), pos=(0.0, 0.0, 0.5)),
    )
    assert box is not None
    scene.build()
    for _ in range(10):
        scene.step()
    pos = box.get_pos()
    assert pos is not None and len(pos) == 3
