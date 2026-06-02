"""
Tests for MPC + WBC Controller Plugin

测试策略:
- 单元测试: 每个子组件独立测试
- 集成测试: MPC+WBC组合运行
- 边界测试: 极端输入、维度不匹配
- 确定性测试: 相同输入产生相同输出
"""

import numpy as np
import pytest

# 导入被测组件
from plugins.controllers.mpc_wbc.core.mpc_controller import MPCController
from plugins.controllers.mpc_wbc.core.wbc_controller import WBCController
from plugins.controllers.mpc_wbc.core.gait_scheduler import GaitScheduler
from plugins.controllers.mpc_wbc.core.combined_controller import MPCWBCController
from plugins.controllers.mpc_wbc.core.config import MPCConfig, WBCConfig, GaitConfig


# =============================================================================
# MPCController Tests
# =============================================================================

class TestMPCController:
    """MPC 控制器单元测试"""

    def test_init_default(self):
        """测试默认初始化"""
        mpc = MPCController()
        assert mpc.dt == 0.005
        assert mpc.N == 10
        assert mpc.nx == 12
        assert mpc.nu == 13
        assert mpc.m == 77.35
        assert mpc.EN is False  # 默认未启用

    def test_init_custom(self):
        """测试自定义参数初始化"""
        mpc = MPCController(dt=0.01, horizon=20, mass=80.0)
        assert mpc.dt == 0.01
        assert mpc.N == 20
        assert mpc.m == 80.0

    def test_enable_disable(self):
        """测试启用/禁用"""
        mpc = MPCController()
        assert mpc.EN is False
        
        mpc.enable()
        assert mpc.EN is True
        
        mpc.disable()
        assert mpc.EN is False

    def test_set_state(self):
        """测试状态设置"""
        mpc = MPCController()
        
        base_pos = np.array([0.0, 0.0, 0.8])
        base_rpy = np.array([0.1, 0.2, 0.3])
        base_lin_vel = np.array([0.5, 0.0, 0.0])
        base_ang_vel = np.array([0.0, 0.0, 0.1])
        
        mpc.set_state(base_pos, base_rpy, base_lin_vel, base_ang_vel)
        
        assert np.allclose(mpc.X_cur[:3], base_rpy)
        assert np.allclose(mpc.X_cur[3:6], base_pos)
        assert np.allclose(mpc.X_cur[6:9], base_ang_vel)
        assert np.allclose(mpc.X_cur[9:12], base_lin_vel)

    def test_set_foot_positions(self):
        """测试足端位置设置"""
        mpc = MPCController()
        
        left = np.array([0.1, 0.15, 0.0])
        right = np.array([0.1, -0.15, 0.0])
        com = np.array([0.0, 0.0, 0.8])
        
        mpc.set_foot_positions(left, right, com)
        
        assert np.allclose(mpc.pf2com[:3], left - com)
        assert np.allclose(mpc.pf2com[3:], right - com)

    def test_compute_disabled(self):
        """测试禁用状态下的计算"""
        mpc = MPCController()
        # EN=False，应返回默认力
        forces = mpc.compute()
        
        assert len(forces) == 12
        assert forces[2] == pytest.approx(-mpc.m * mpc.g / 2, rel=1e-5)
        assert forces[8] == pytest.approx(-mpc.m * mpc.g / 2, rel=1e-5)

    def test_compute_enabled(self):
        """测试启用状态下的计算"""
        mpc = MPCController()
        mpc.enable()
        
        # 设置状态
        mpc.set_state(
            np.array([0.0, 0.0, 0.8]),
            np.array([0.0, 0.0, 0.0]),
            np.array([0.0, 0.0, 0.0]),
            np.array([0.0, 0.0, 0.0])
        )
        mpc.set_foot_positions(
            np.array([0.1, 0.15, 0.0]),
            np.array([0.1, -0.15, 0.0]),
            np.array([0.0, 0.0, 0.8])
        )
        
        forces = mpc.compute()
        assert len(forces) == 12
        # 垂直力应为正（向上）
        assert forces[2] >= 0
        assert forces[8] >= 0

    def test_set_weight(self):
        """测试权重设置"""
        mpc = MPCController()
        
        L_new = np.ones(12) * 2.0
        K_new = np.ones(13) * 0.5
        
        mpc.set_weight(1e-5, L_new, K_new)
        
        assert mpc.alpha == 1e-5
        assert np.allclose(mpc.L_diag, L_new)
        assert np.allclose(mpc.K_diag, K_new)

    def test_set_weight_wrong_dim(self):
        """测试错误维度的权重设置"""
        mpc = MPCController()
        
        # 错误维度应被忽略或调整
        L_wrong = np.ones(5)
        K_wrong = np.ones(5)
        
        mpc.set_weight(1e-5, L_wrong, K_wrong)
        # 不应崩溃
        assert mpc.alpha == 1e-5

    def test_system_matrices_shape(self):
        """测试系统矩阵维度"""
        mpc = MPCController()
        
        assert mpc.A.shape == (mpc.nx, mpc.nx)
        assert mpc.B.shape == (mpc.nx, mpc.nu)

    def test_deterministic(self):
        """测试确定性：相同输入产生相同输出"""
        mpc1 = MPCController()
        mpc1.enable()
        mpc2 = MPCController()
        mpc2.enable()
        
        state = (
            np.array([0.0, 0.0, 0.8]),
            np.array([0.0, 0.0, 0.0]),
            np.array([0.0, 0.0, 0.0]),
            np.array([0.0, 0.0, 0.0])
        )
        feet = (
            np.array([0.1, 0.15, 0.0]),
            np.array([0.1, -0.15, 0.0]),
            np.array([0.0, 0.0, 0.8])
        )
        
        mpc1.set_state(*state)
        mpc1.set_foot_positions(*feet)
        mpc2.set_state(*state)
        mpc2.set_foot_positions(*feet)
        
        f1 = mpc1.compute()
        f2 = mpc2.compute()
        
        assert np.allclose(f1, f2)


# =============================================================================
# WBCController Tests
# =============================================================================

class TestWBCController:
    """WBC 控制器单元测试"""

    def test_init_default(self):
        """测试默认初始化"""
        wbc = WBCController()
        assert wbc.num_dofs == 19
        assert wbc.dt == 0.005
        assert len(wbc.tasks) == 0

    def test_init_custom(self):
        """测试自定义参数"""
        wbc = WBCController(num_dofs=25, dt=0.01)
        assert wbc.num_dofs == 25
        assert wbc.dt == 0.01

    def test_add_task(self):
        """测试添加任务"""
        wbc = WBCController()
        
        J = np.random.randn(3, 19)
        xdd = np.random.randn(3)
        
        wbc.add_task('com', J, xdd, weight=1.0, priority=0)
        
        assert len(wbc.tasks) == 1
        assert wbc.tasks[0]['name'] == 'com'
        assert np.allclose(wbc.tasks[0]['J'], J)

    def test_clear_tasks(self):
        """测试清除任务"""
        wbc = WBCController()
        wbc.add_task('test', np.random.randn(3, 19), np.random.randn(3))
        wbc.clear_tasks()
        assert len(wbc.tasks) == 0

    def test_compute_torques_empty(self):
        """测试无任务时的力矩计算"""
        wbc = WBCController()
        wbc.clear_tasks()
        
        tau = wbc.compute_torques(
            np.zeros(19),
            np.zeros(19),
            np.zeros(12)
        )
        
        assert len(tau) == 19
        assert np.allclose(tau, 0)

    def test_compute_torques_with_tasks(self):
        """测试有任务时的力矩计算"""
        wbc = WBCController()
        wbc.clear_tasks()
        
        # 添加简单任务
        J = np.eye(3, 19)
        xdd = np.array([0.0, 0.0, 9.81])
        wbc.add_task('com', J, xdd, priority=0)
        
        tau = wbc.compute_torques(
            np.zeros(19),
            np.zeros(19),
            np.zeros(12)
        )
        
        assert len(tau) == 19
        # 有力矩输出
        assert not np.allclose(tau, 0)

    def test_compute_com_task(self):
        """测试质心任务计算"""
        wbc = WBCController()
        
        J, xdd = wbc.compute_com_task(
            com_pos=np.array([0.0, 0.0, 0.8]),
            com_vel=np.array([0.0, 0.0, 0.0]),
            desired_com_pos=np.array([0.0, 0.0, 0.85]),
            desired_com_vel=np.array([0.0, 0.0, 0.0]),
            contact_forces=np.zeros(12)
        )
        
        assert J.shape == (3, 19)
        assert len(xdd) == 3
        # PD 修正应产生向上加速度
        assert xdd[2] > 0

    def test_compute_swing_foot_task(self):
        """测试摆动脚任务计算"""
        wbc = WBCController()
        
        J, xdd = wbc.compute_swing_foot_task(
            swing_foot_pos=np.array([0.1, 0.15, 0.0]),
            swing_foot_vel=np.zeros(3),
            desired_pos=np.array([0.2, 0.15, 0.1]),
            desired_vel=np.zeros(3)
        )
        
        assert J.shape == (3, 19)
        assert len(xdd) == 3

    def test_pseudoinverse(self):
        """测试伪逆计算"""
        wbc = WBCController()
        
        J = np.random.randn(3, 19)
        J_pinv = wbc._pseudoinverse(J)
        
        # 验证伪逆性质: J @ J_pinv @ J ≈ J
        reconstructed = J @ J_pinv @ J
        assert np.allclose(J, reconstructed, atol=1e-3)


# =============================================================================
# GaitScheduler Tests
# =============================================================================

class TestGaitScheduler:
    """步态调度器单元测试"""

    def test_init_default(self):
        """测试默认初始化"""
        gs = GaitScheduler()
        assert gs.frequency == 1.25
        assert gs.gait_type == 'trot'
        assert gs.duty_factor == 0.5
        assert gs.phase == 0.0

    def test_init_custom(self):
        """测试自定义参数"""
        gs = GaitScheduler(frequency=2.0, gait_type='walk', duty_factor=0.75)
        assert gs.frequency == 2.0
        assert gs.gait_type == 'walk'
        assert gs.duty_factor == 0.75

    def test_update(self):
        """测试相位更新"""
        gs = GaitScheduler(frequency=1.0)
        
        phase = gs.update(0.1)  # 0.1秒
        assert phase == pytest.approx(0.1, abs=1e-6)
        
        phase = gs.update(0.95)  # 再更新0.95秒，应回绕
        assert phase < 0.1  # 回绕到接近0

    def test_get_swing_leg_trot(self):
        """测试 trot 步态摆动腿"""
        gs = GaitScheduler(gait_type='trot')
        
        gs.phase = 0.1
        assert gs.get_swing_leg() == 'left'
        
        gs.phase = 0.6
        assert gs.get_swing_leg() == 'right'

    def test_get_swing_leg_walk(self):
        """测试 walk 步态摆动腿"""
        gs = GaitScheduler(gait_type='walk')
        
        gs.phase = 0.1
        assert gs.get_swing_leg() == 'left'
        
        gs.phase = 0.3
        assert gs.get_swing_leg() == 'right'

    def test_get_leg_state(self):
        """测试腿状态"""
        gs = GaitScheduler(gait_type='trot')
        gs.phase = 0.1
        
        # 当前摆动腿
        swing = gs.get_swing_leg()
        assert gs.get_leg_state(swing) == 0  # 摆动
        
        # 支撑腿
        stance = 'right' if swing == 'left' else 'left'
        assert gs.get_leg_state(stance) >= 1  # 支撑

    def test_get_swing_height(self):
        """测试摆动高度"""
        gs = GaitScheduler()
        
        # 相位为0或0.5时高度为0
        gs.phase = 0.0
        assert gs.get_swing_height() == 0.0
        
        gs.phase = 0.5
        assert gs.get_swing_height() == 0.0
        
        # 相位为0.25时高度最大
        gs.phase = 0.25
        height = gs.get_swing_height()
        assert height > 0
        assert height <= 0.08

    def test_is_swing_start(self):
        """测试摆动开始检测"""
        gs = GaitScheduler()
        
        gs.phase = 0.01
        assert gs.is_swing_start() is True
        
        gs.phase = 0.5
        assert gs.is_swing_start() is True
        
        gs.phase = 0.3
        assert gs.is_swing_start() is False

    def test_is_swing_end(self):
        """测试摆动结束检测"""
        gs = GaitScheduler()
        
        gs.phase = 0.49
        assert gs.is_swing_end() is True
        
        gs.phase = 0.99
        assert gs.is_swing_end() is True

    def test_period(self):
        """测试周期计算"""
        gs = GaitScheduler(frequency=2.0)
        assert gs.period == pytest.approx(0.5, abs=1e-6)


# =============================================================================
# MPCWBCController (Combined) Tests
# =============================================================================

class TestMPCWBCController:
    """组合控制器集成测试"""

    def test_init_default(self):
        """测试默认初始化"""
        ctrl = MPCWBCController()
        assert ctrl.num_dofs == 19
        assert ctrl.dt == 0.005
        assert ctrl.use_mpc is True
        assert ctrl.use_wbc is True
        assert ctrl.time == 0.0

    def test_init_no_mpc(self):
        """测试禁用 MPC"""
        ctrl = MPCWBCController(use_mpc=False)
        assert ctrl.use_mpc is False
        assert not hasattr(ctrl, 'mpc') or ctrl.mpc is None

    def test_init_no_wbc(self):
        """测试禁用 WBC"""
        ctrl = MPCWBCController(use_wbc=False)
        assert ctrl.use_wbc is False
        assert not hasattr(ctrl, 'wbc') or ctrl.wbc is None

    def test_update_full(self):
        """测试完整更新流程"""
        ctrl = MPCWBCController()
        
        # 模拟输入
        base_pos = np.array([0.0, 0.0, 0.8])
        base_rpy = np.array([0.0, 0.0, 0.0])
        base_lin_vel = np.array([0.5, 0.0, 0.0])
        base_ang_vel = np.array([0.0, 0.0, 0.0])
        left_foot = np.array([0.1, 0.15, 0.0])
        right_foot = np.array([0.1, -0.15, 0.0])
        joint_pos = np.zeros(19)
        joint_vel = np.zeros(19)
        target_vel = np.array([1.0, 0.0, 0.0])
        
        tau = ctrl.update(
            base_pos, base_rpy,
            base_lin_vel, base_ang_vel,
            left_foot, right_foot,
            joint_pos, joint_vel,
            target_vel
        )
        
        assert len(tau) == 19
        assert not np.any(np.isnan(tau))
        assert not np.any(np.isinf(tau))

    def test_update_mpc_only(self):
        """测试仅 MPC 模式"""
        ctrl = MPCWBCController(use_wbc=False)
        
        tau = ctrl.update(
            np.array([0.0, 0.0, 0.8]),
            np.array([0.0, 0.0, 0.0]),
            np.array([0.0, 0.0, 0.0]),
            np.array([0.0, 0.0, 0.0]),
            np.array([0.1, 0.15, 0.0]),
            np.array([0.1, -0.15, 0.0]),
            np.zeros(19),
            np.zeros(19),
            np.array([0.5, 0.0, 0.0])
        )
        
        assert len(tau) == 19

    def test_update_wbc_only(self):
        """测试仅 WBC 模式"""
        ctrl = MPCWBCController(use_mpc=False)
        
        tau = ctrl.update(
            np.array([0.0, 0.0, 0.8]),
            np.array([0.0, 0.0, 0.0]),
            np.array([0.0, 0.0, 0.0]),
            np.array([0.0, 0.0, 0.0]),
            np.array([0.1, 0.15, 0.0]),
            np.array([0.1, -0.15, 0.0]),
            np.zeros(19),
            np.zeros(19),
            np.array([0.5, 0.0, 0.0])
        )
        
        assert len(tau) == 19

    def test_time_progression(self):
        """测试时间推进"""
        ctrl = MPCWBCController(dt=0.01)
        
        for _ in range(10):
            ctrl.update(
                np.array([0.0, 0.0, 0.8]),
                np.array([0.0, 0.0, 0.0]),
                np.array([0.0, 0.0, 0.0]),
                np.array([0.0, 0.0, 0.0]),
                np.array([0.1, 0.15, 0.0]),
                np.array([0.1, -0.15, 0.0]),
                np.zeros(19),
                np.zeros(19),
                np.array([0.5, 0.0, 0.0])
            )
        
        assert ctrl.time == pytest.approx(0.1, abs=1e-6)

    def test_swing_leg_alternation(self):
        """测试摆动腿交替"""
        ctrl = MPCWBCController(dt=0.01)
        ctrl.gait_scheduler.frequency = 1.0  # 1Hz
        
        legs = []
        for _ in range(200):  # 2秒
            ctrl.update(
                np.array([0.0, 0.0, 0.8]),
                np.array([0.0, 0.0, 0.0]),
                np.array([0.0, 0.0, 0.0]),
                np.array([0.0, 0.0, 0.0]),
                np.array([0.1, 0.15, 0.0]),
                np.array([0.1, -0.15, 0.0]),
                np.zeros(19),
                np.zeros(19),
                np.array([0.5, 0.0, 0.0])
            )
            legs.append(ctrl.swing_leg)
        
        # 应该有左右交替
        assert 'left' in legs
        assert 'right' in legs

    def test_deterministic(self):
        """测试确定性"""
        ctrl1 = MPCWBCController()
        ctrl2 = MPCWBCController()
        
        inputs = {
            'base_pos': np.array([0.0, 0.0, 0.8]),
            'base_rpy': np.array([0.1, 0.0, 0.0]),
            'base_lin_vel': np.array([0.5, 0.0, 0.0]),
            'base_ang_vel': np.array([0.0, 0.0, 0.1]),
            'left_foot_pos': np.array([0.1, 0.15, 0.0]),
            'right_foot_pos': np.array([0.1, -0.15, 0.0]),
            'joint_pos': np.zeros(19),
            'joint_vel': np.zeros(19),
            'target_vel': np.array([1.0, 0.0, 0.0])
        }
        
        tau1 = ctrl1.update(**inputs)
        tau2 = ctrl2.update(**inputs)
        
        assert np.allclose(tau1, tau2)


# =============================================================================
# Config Tests
# =============================================================================

class TestConfig:
    """配置类测试"""

    def test_mpc_config_default(self):
        """测试 MPC 配置默认值"""
        cfg = MPCConfig()
        assert cfg.dt == 0.005
        assert cfg.horizon == 10
        assert cfg.mass == 77.35
        assert cfg.max_force is not None
        assert cfg.min_force is not None

    def test_wbc_config_default(self):
        """测试 WBC 配置默认值"""
        from plugins.controllers.mpc_wbc.core.config import WBCJointConfig
        cfg = WBCJointConfig()
        assert cfg.num_dofs == 19
        assert cfg.dt == 0.005
        assert cfg.kp_com == 100.0
        assert cfg.kd_com == 20.0

    def test_gait_config_default(self):
        """测试步态配置默认值"""
        cfg = GaitConfig()
        assert cfg.frequency == 1.25
        assert cfg.gait_type == 'trot'
        assert cfg.swing_height == 0.08

    def test_wbc_full_config(self):
        """测试完整配置"""
        from plugins.controllers.mpc_wbc.core.config import WBCConfig as FullWBCConfig
        
        cfg = FullWBCConfig()
        assert cfg.use_mpc is True
        assert cfg.use_wbc is True
        assert cfg.mpc is not None
        assert cfg.wbc is not None
        assert cfg.gait is not None


# =============================================================================
# Edge Cases & Error Handling
# =============================================================================

class TestEdgeCases:
    """边界情况和错误处理测试"""

    def test_mpc_zero_mass(self):
        """测试零质量（应处理或报错）"""
        # 质量为0可能导致除零
        mpc = MPCController(mass=1e-6)
        mpc.enable()
        mpc.set_state(
            np.array([0.0, 0.0, 0.8]),
            np.zeros(3),
            np.zeros(3),
            np.zeros(3)
        )
        forces = mpc.compute()
        assert not np.any(np.isnan(forces))
        assert not np.any(np.isinf(forces))

    def test_wbc_zero_jacobian(self):
        """测试零雅可比矩阵"""
        wbc = WBCController()
        wbc.add_task('zero', np.zeros((3, 19)), np.zeros(3))
        
        tau = wbc.compute_torques(np.zeros(19), np.zeros(19), np.zeros(12))
        assert len(tau) == 19
        assert not np.any(np.isnan(tau))

    def test_gait_scheduler_zero_frequency(self):
        """测试零频率步态"""
        gs = GaitScheduler(frequency=0.0)
        phase = gs.update(1.0)
        assert phase == 0.0  # 不应变化

    def test_large_target_velocity(self):
        """测试极大目标速度"""
        ctrl = MPCWBCController()
        
        tau = ctrl.update(
            np.array([0.0, 0.0, 0.8]),
            np.zeros(3),
            np.zeros(3),
            np.zeros(3),
            np.array([0.1, 0.15, 0.0]),
            np.array([0.1, -0.15, 0.0]),
            np.zeros(19),
            np.zeros(19),
            np.array([100.0, 0.0, 0.0])  # 极大速度
        )
        
        assert len(tau) == 19
        assert not np.any(np.isnan(tau))
        assert not np.any(np.isinf(tau))

    def test_negative_dt(self):
        """测试负时间步长"""
        # 负 dt 可能导致相位倒退
        gs = GaitScheduler(frequency=1.0)
        gs.update(-0.1)
        # 不应崩溃
        assert gs.phase >= 0.0


# =============================================================================
# Performance Tests
# =============================================================================

class TestPerformance:
    """性能测试（可选，用于基准）"""

    @pytest.mark.slow
    def test_mpc_compute_speed(self):
        """测试 MPC 计算速度"""
        mpc = MPCController()
        mpc.enable()
        mpc.set_state(
            np.array([0.0, 0.0, 0.8]),
            np.zeros(3),
            np.zeros(3),
            np.zeros(3)
        )
        mpc.set_foot_positions(
            np.array([0.1, 0.15, 0.0]),
            np.array([0.1, -0.15, 0.0]),
            np.array([0.0, 0.0, 0.8])
        )
        
        import time
        start = time.time()
        for _ in range(100):
            mpc.compute()
        elapsed = time.time() - start
        
        # 100次计算应在1秒内完成
        assert elapsed < 1.0

    @pytest.mark.slow
    def test_full_update_speed(self):
        """测试完整更新速度"""
        ctrl = MPCWBCController()
        
        import time
        start = time.time()
        for _ in range(100):
            ctrl.update(
                np.array([0.0, 0.0, 0.8]),
                np.zeros(3),
                np.zeros(3),
                np.zeros(3),
                np.array([0.1, 0.15, 0.0]),
                np.array([0.1, -0.15, 0.0]),
                np.zeros(19),
                np.zeros(19),
                np.array([1.0, 0.0, 0.0])
            )
        elapsed = time.time() - start
        
        # 100次更新应在2秒内完成
        assert elapsed < 2.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
