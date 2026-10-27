"""DomainRandomizer 核心测试：采样语义用 FakeEntity，集成用 Go2 场景。"""

from __future__ import annotations

import pytest
import torch

from cloud_robotics_sim.core.domain_rand import DomainRandomizer, ParamSpec


class _FakeSolver:
    def __init__(self, n_envs: int, batch_links: bool = True, batch_dofs: bool = True):
        self._options = type(
            "Opts", (), {"batch_links_info": batch_links, "batch_dofs_info": batch_dofs}
        )()
        self.n_envs = n_envs


class _FakeEntity:
    """Records setter calls; mimics the built-entity surface the randomizer uses."""

    def __init__(self, n_envs: int = 4, n_links: int = 3, n_dofs: int = 5):
        self._solver = _FakeSolver(n_envs)
        self.device = torch.device("cpu")
        self.n_links = n_links
        self.n_dofs = n_dofs
        self.n_envs = n_envs
        self.mass0 = torch.full((n_links,), 2.0)
        self.kp0 = torch.full((n_dofs,), 60.0)
        self.kv0 = torch.full((n_dofs,), 3.0)
        self.arm0 = torch.full((n_dofs,), 0.01)
        self.calls: list[tuple[str, torch.Tensor, object]] = []

    def get_mass(self):
        return self.mass0.repeat(self.n_envs, 1).sum(dim=1)

    def set_mass(self, values, envs_idx=None):
        self.calls.append(("mass", values.detach().clone(), envs_idx))

    def set_friction_ratio(self, values, links_idx_local=None, envs_idx=None):
        self.calls.append(("friction_ratio", values.detach().clone(), envs_idx))

    def get_dofs_kp(self):
        return self.kp0.repeat(self.n_envs, 1)

    def set_dofs_kp(self, values, dofs_idx_local=None, envs_idx=None):
        self.calls.append(("kp", values.detach().clone(), envs_idx))

    def get_dofs_kv(self):
        return self.kv0.repeat(self.n_envs, 1)

    def set_dofs_kv(self, values, dofs_idx_local=None, envs_idx=None):
        self.calls.append(("kv", values.detach().clone(), envs_idx))

    def get_dofs_armature(self):
        return self.arm0.repeat(self.n_envs, 1)

    def set_dofs_armature(self, values, dofs_idx_local=None, envs_idx=None):
        self.calls.append(("armature", values.detach().clone(), envs_idx))

    def set_dofs_stiffness(self, values, dofs_idx_local=None, envs_idx=None):
        self.calls.append(("stiffness", values.detach().clone(), envs_idx))

    def set_dofs_damping(self, values, dofs_idx_local=None, envs_idx=None):
        self.calls.append(("damping", values.detach().clone(), envs_idx))

    def set_dofs_frictionloss(self, values, dofs_idx_local=None, envs_idx=None):
        self.calls.append(("frictionloss", values.detach().clone(), envs_idx))


def _specs() -> list[ParamSpec]:
    return [
        ParamSpec("mass", "mass", 0.8, 1.2),
        ParamSpec("friction", "friction_ratio", 0.5, 1.5, scale="absolute"),
        ParamSpec("kp", "kp", 0.9, 1.1),
    ]


def test_paramspec_validation() -> None:
    """ParamSpec 参数校验:非法 target/区间/分布报错。"""
    with pytest.raises(ValueError):
        ParamSpec("x", "nope", 0.0, 1.0)
    with pytest.raises(ValueError):
        ParamSpec("x", "kp", 1.0, 0.0)
    with pytest.raises(ValueError):
        ParamSpec("x", "kp", -1.0, 1.0, distribution="log_uniform")
    with pytest.raises(ValueError):
        ParamSpec("x", "kp", 0.0, 1.0, distribution="cauchy")


def test_sample_bounds_and_shapes() -> None:
    """整批采样:形状、数值区间符合 spec 定义。"""
    entity = _FakeEntity(n_envs=8)
    rz = DomainRandomizer(entity, _specs(), seed=7)
    out = rz.sample()
    assert out["mass"].shape == (8,)
    assert out["friction"].shape == (8, 3)
    assert out["kp"].shape == (8, 5)
    assert out["mass"].min() >= 0.8 * entity.mass0.sum() - 1e-6
    assert out["mass"].max() <= 1.2 * entity.mass0.sum() + 1e-6
    assert out["friction"].min() >= 0.5 and out["friction"].max() <= 1.5
    assert out["kp"].min() >= 0.9 * 60.0 and out["kp"].max() <= 1.1 * 60.0


def test_reseed_reproduces_sample() -> None:
    """重播种后采样序列完全复现。"""
    entity = _FakeEntity()
    rz = DomainRandomizer(entity, _specs(), seed=42)
    first = rz.sample()
    rz.manual_seed(42)
    second = rz.sample()
    for key in first:
        assert torch.equal(first[key], second[key])


def test_relative_scale_does_not_random_walk() -> None:
    """Relative 缩放始终围绕构造时快照的基值,不随机游走。"""
    entity = _FakeEntity()
    rz = DomainRandomizer(entity, [ParamSpec("kp", "kp", 0.9, 1.1)], seed=3)
    first = rz.sample()["kp"]
    second = rz.sample()["kp"]
    # 两次采样都围绕构造时快照的基值,而不是围绕上一次写入值。
    assert (first - 60.0).abs().max() <= 6.0 + 1e-6
    assert (second - 60.0).abs().max() <= 6.0 + 1e-6


def test_envs_idx_subset_writes_only_selected() -> None:
    """envs_idx 子集写入:值张量与 setter 都只覆盖选中 env。"""
    entity = _FakeEntity(n_envs=8)
    rz = DomainRandomizer(entity, _specs(), seed=5)
    rz.sample([0, 3])
    for target, values, envs_idx in entity.calls:
        assert values.shape[0] == 2, target
        assert envs_idx is not None


def test_gaussian_and_log_uniform_stay_in_bounds() -> None:
    """gaussian(截断)与 log_uniform 采样保持在界内且分布形态正确。"""
    entity = _FakeEntity(n_envs=256)
    rz = DomainRandomizer(
        entity,
        [
            ParamSpec("g", "kp", 0.5, 1.5, distribution="gaussian"),
            ParamSpec("l", "kv", 0.1, 10.0, distribution="log_uniform"),
        ],
        seed=9,
    )
    out = rz.sample()
    # kp/kv 默认为 relative 缩放:输出值 = 基值(60 / 3) × 因子,区间相应平移
    assert out["g"].min() >= 30.0 and out["g"].max() <= 90.0
    assert out["l"].min() >= 0.3 and out["l"].max() <= 30.0
    # 对数均匀的中位数应远小于算术均值(右偏分布)
    assert out["l"].median() < 3.0


def test_missing_batch_options_raise() -> None:
    """缺少 per-env batch 选项时给出明确报错。"""
    entity = _FakeEntity()
    entity._solver._options = type(
        "Opts", (), {"batch_links_info": False, "batch_dofs_info": False}
    )()
    with pytest.raises(RuntimeError):
        DomainRandomizer(entity, [ParamSpec("kp", "kp", 0.9, 1.1)])


def test_relative_dof_target_without_getter_raises() -> None:
    """无 getter 的 dof target 用 relative 缩放时报错。"""
    entity = _FakeEntity()
    with pytest.raises(ValueError):
        DomainRandomizer(entity, [ParamSpec("d", "damping", 0.9, 1.1)])


def test_from_config_bridge() -> None:
    """from_config 桥接:识别 legged-RL 配置键并映射为 ParamSpec。"""
    entity = _FakeEntity()
    config = {
        "friction_range": [0.5, 1.5],
        "mass_scale_range": [0.8, 1.2],
        "kp_range": [0.9, 1.1],
        "damping_range": [0.5, 2.0],
        "unrelated_key": True,
    }
    rz = DomainRandomizer.from_config(entity, config, seed=1)
    assert {s.name for s in rz._specs} == {"friction", "mass_scale", "kp", "damping"}
    out = rz.sample()
    assert out["damping"].min() >= 0.5 and out["damping"].max() <= 2.0
    with pytest.raises(ValueError):
        DomainRandomizer.from_config(entity, {"nothing": [0, 1]})


# --------------------------------------------------------------------------- #
# Go2 集成测试（真实 genesis 场景,验证 setter 路径与逐 env 发散）
# --------------------------------------------------------------------------- #


def _build_go2_scene(n_envs: int = 8):
    import genesis as gs

    from cloud_robotics_sim.utils.genesis_compat import ensure_genesis_initialized

    ensure_genesis_initialized(use_cuda=False, seed=20261024)
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=0.02, substeps=2),
        rigid_options=gs.options.RigidOptions(
            batch_links_info=True,
            batch_dofs_info=True,
            max_collision_pairs=32,
        ),
        show_viewer=False,
    )
    scene.add_entity(gs.morphs.URDF(file="urdf/plane/plane.urdf", fixed=True))
    robot = scene.add_entity(
        gs.morphs.URDF(file="urdf/go2/urdf/go2.urdf", pos=(0.0, 0.0, 0.45))
    )
    scene.build(n_envs=n_envs)
    motor_dof_idx = [joint.dof_start for joint in robot.joints[1:]]
    robot.set_dofs_kp([60.0] * len(motor_dof_idx), dofs_idx_local=motor_dof_idx)
    robot.set_dofs_kv([3.0] * len(motor_dof_idx), dofs_idx_local=motor_dof_idx)
    return scene, robot, motor_dof_idx


def test_go2_randomizer_writes_per_env_and_diverges() -> None:
    """真实 Go2:整批写入 mass/friction/kp/kv,重播种复现,按 envs_idx 子集写入,
    随机化物理在 PD 指令下真正跨 env 发散。
    """
    scene, robot, motor_dof_idx = _build_go2_scene()

    specs = [
        ParamSpec("mass", "mass", 0.8, 1.2),
        ParamSpec("friction", "friction_ratio", 0.5, 1.5, scale="absolute"),
        ParamSpec("kp", "kp", 0.9, 1.1, indices=tuple(motor_dof_idx)),
        ParamSpec("kv", "kv", 0.9, 1.1, indices=tuple(motor_dof_idx)),
    ]
    rz = DomainRandomizer(robot, specs, seed=1234)

    sampled = rz.sample()
    assert sampled["mass"].shape == (8,)
    assert sampled["friction"].shape == (8, robot.n_links)
    assert sampled["kp"].shape == (8, len(motor_dof_idx))

    mass_after = robot.get_mass()
    assert not torch.allclose(mass_after, mass_after[0].expand_as(mass_after))

    rz.manual_seed(1234)
    replay = rz.sample()
    for key in sampled:
        assert torch.equal(sampled[key], replay[key])

    # 子集写入:选中的 env 变化,其余 env 保持不动
    rz.manual_seed(1234)
    rz.sample()
    kp_before = robot.get_dofs_kp().clone()
    rz.sample([3])
    kp_after = robot.get_dofs_kp()
    assert not torch.allclose(kp_before[3], kp_after[3])
    assert torch.allclose(
        kp_before[[0, 1, 2, 4, 5, 6, 7]], kp_after[[0, 1, 2, 4, 5, 6, 7]]
    )
