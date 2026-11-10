"""XpbdRope 测试：自由落体解析解、不可伸长、球/地面碰撞、摩擦、弯曲约束。

解析 SDF 场全部可在 CPU 运行；xuvdb 场一致性单独由 ``xuvdb_only`` 守护。
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from tests.optional_deps import torch_only, xuvdb_only

pytestmark = torch_only

from cloud_robotics_sim.vdb import (  # noqa: E402
    XpbdRope,
    XuvdbSdfField,
    XuvdbVolume,
    ground_field,
    sphere_field,
    union_fields,
)
from cloud_robotics_sim.vdb.rope_xpbd import AnalyticSdfField  # noqa: E402

_DT = 1.0 / 60.0


def _settle(rope: XpbdRope, steps: int, field=None) -> None:
    for _ in range(steps):
        rope.step(_DT, field=field)


def test_free_fall_matches_analytic() -> None:
    """无场自由落体：末端粒子高度贴合 z(t) = z0 - g t² / 2。"""
    rope = XpbdRope(5, 0.05, origin=(0, 0, 1.0), substeps=20, device="cpu")
    n_steps = 15
    for _ in range(n_steps):
        rope.step(_DT)
    t = n_steps * _DT
    z = float(rope.positions[-1, 2])
    z_ref = 1.0 - 4 * 0.05 - 0.5 * 9.81 * t * t
    assert z == pytest.approx(z_ref, rel=0.02)


def test_pinned_hanging_rope_conserves_length() -> None:
    """钉住顶端自由下垂：静止后总拉伸 < 3%，保持竖直且数值稳定。"""
    rope = XpbdRope(40, 0.03, origin=(0, 0, 1.5), linear_damping=2.0, device="cpu")
    rope.pin(0)
    _settle(rope, 180)
    assert torch.isfinite(rope.positions).all()
    assert abs(rope.stretch_ratio()) < 0.03
    # 静止：动能趋零；竖直：水平漂移很小。
    assert rope.kinetic_energy() < 1e-6
    xy = rope.positions[:, :2].abs().max()
    assert float(xy) < 0.02


def test_rope_drapes_sphere_without_penetration() -> None:
    """绳从球上方下垂披覆：任何粒子不得穿入球面（margin 内）。"""
    r_ball = 0.3
    field = sphere_field((0.0, 0.0, 0.0), r_ball)
    # 初始绳底略高于球顶（pin 在球顶 + 绳长 - 5cm 让接触平缓发生）；
    # 初始形状与场相交会被碰撞投影猛烈弹出，属调用方责任。
    rope = XpbdRope(
        30,
        0.03,
        origin=(0.0, 0.0, r_ball + 29 * 0.03 - 0.05),
        margin=0.005,
        linear_damping=1.0,
        device="cpu",
    )
    rope.pin(0)
    _settle(rope, 240, field=field)
    assert torch.isfinite(rope.positions).all()
    diag = rope.diagnostics(field)
    # 解析场是精确的：允许数值容差 1e-3。
    assert diag["max_penetration"] < 1e-3
    # 披覆后末端应低于悬挂点（绳真的搭下去了）。
    assert float(rope.positions[-1, 2]) < 0.6


def test_ground_plane_stops_fall() -> None:
    """水平绳自由下落到地面：最低点停在 margin 处，不穿透。"""
    field = ground_field(0.0)
    rope = XpbdRope(
        40,
        0.03,
        origin=(-0.6, 0.0, 0.4),
        direction=(1.0, 0.0, 0.0),
        margin=0.01,
        linear_damping=1.0,
        device="cpu",
    )
    _settle(rope, 240, field=field)
    assert torch.isfinite(rope.positions).all()
    assert rope.diagnostics(field)["max_penetration"] < 1e-3
    assert float(rope.positions[:, 2].min()) > 0.009


def test_friction_reduces_tangential_sliding() -> None:
    """同一初速滑过地面：高摩擦的质心水平位移明显更小。"""
    field = ground_field(0.0)

    def run(mu: float) -> float:
        rope = XpbdRope(
            12,
            0.04,
            origin=(0.0, 0.0, 0.03),
            direction=(1.0, 0.0, 0.0),
            friction=mu,
            margin=0.01,
            device="cpu",
        )
        rope.velocities[:, 0] = 1.0
        _settle(rope, 60, field=field)
        return float(rope.positions[:, 0].mean())

    slide_free = run(0.0)
    slide_grip = run(2.0)
    assert slide_grip < 0.5 * slide_free


def test_bend_compliance_straightens_chain() -> None:
    """零重力下弯折的链：开弯曲约束后更接近直线（总长/端到端距离 -> 1）。"""
    n = 12
    pts = torch.zeros(n, 3)
    pts[:, 0] = torch.arange(n, dtype=torch.float32) * 0.03
    pts[6:, 2] = -(torch.arange(6, dtype=torch.float32) + 1.0) * 0.03

    def straightness(bend: float) -> float:
        rope = XpbdRope(
            n, 0.03, gravity=(0.0, 0.0, 0.0), bend_compliance=bend, device="cpu"
        )
        rope.reposition(pts)
        _settle(rope, 60)
        seg = rope.positions[1:] - rope.positions[:-1]
        end_to_end = (rope.positions[-1] - rope.positions[0]).norm()
        return float(seg.norm(dim=1).sum() / end_to_end)

    assert straightness(0.0) < straightness(-1.0)


def test_union_fields_min_semantics() -> None:
    """并集场：取最近表面，法向随胜出者。"""
    field = union_fields(
        sphere_field((-1.0, 0.0, 0.0), 0.5), sphere_field((1.0, 0.0, 0.0), 0.5)
    )
    pts = torch.tensor([[0.0, 0.0, 0.0], [0.6, 0.0, 0.0]])
    d, g = field.sample(pts)
    # 中点距两球各 0.5；右侧点 (0.6) 距右球心 0.4，球内 d = -0.1。
    assert float(d[0]) == pytest.approx(0.5, abs=1e-6)
    assert float(d[1]) == pytest.approx(-0.1, abs=1e-6)
    # 法向 = (点 - 球心)/|..|：在 (0.6,0,0) 处指向 -x。
    assert float(g[1, 0]) == pytest.approx(-1.0, abs=1e-6)


def test_reposition_shape_validation() -> None:
    """Reposition 形状不符时抛错。"""
    rope = XpbdRope(4, 0.1)
    with pytest.raises(ValueError):
        rope.reposition(torch.zeros(3, 3))


def test_analytic_field_type() -> None:
    """AnalyticSdfField 是 SdfField 协议的合法实现（运行时校验 duck-type）。"""
    field = AnalyticSdfField(lambda p: (p[:, 2], torch.ones_like(p)))
    d, g = field.sample(torch.zeros(2, 3))
    assert d.shape == (2,)
    assert g.shape == (2, 3)


@xuvdb_only
def test_xuvdb_field_matches_analytic_sphere() -> None:
    """XuvdbSdfField 与解析球场一致：值误差 < 半个体素，法向夹角小。"""
    voxel = 0.02
    vol = XuvdbVolume(
        background=3 * voxel,
        voxel_size=voxel,
        leaf_log2=4,
        name="ball",
        grid_class="level set",
    )
    vol.stamp_sphere((0.0, 0.0, 0.0), radius=0.3)
    xf = XuvdbSdfField(vol)
    ref = sphere_field((0.0, 0.0, 0.0), 0.3)

    rng = np.random.default_rng(0)
    pts = rng.uniform(-0.45, 0.45, size=(64, 3)).astype(np.float32)
    d_x, g_x = xf.sample(torch.from_numpy(pts))
    d_r, g_r = ref.sample(torch.from_numpy(pts))
    # 窄带 SDF 在带外饱和为背景值（3 voxel），只在带内比较。
    in_band = d_r.abs() < 2.5 * voxel
    assert in_band.any()
    assert (d_x[in_band] - d_r[in_band]).abs().max() < 0.5 * voxel
    dots = torch.nn.functional.cosine_similarity(g_x[in_band], g_r[in_band], dim=1)
    assert float(dots.min()) > 0.98
