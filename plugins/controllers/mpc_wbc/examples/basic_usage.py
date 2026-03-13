"""
Basic Usage Example: MPC + WBC Controller

演示如何使用 MPC + WBC 控制器控制人形机器人
"""

import numpy as np
import genesis as gs
from cloud_robotics_sim.plugins.controllers.mpc_wbc import MPCWBCController


def main():
    """主函数"""
    
    # 1. 初始化 Genesis
    gs.init(backend=gs.cpu)
    
    # 2. 创建场景
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=0.005),
        show_viewer=True
    )
    
    # 3. 添加地面
    ground = scene.add_entity(gs.morphs.Plane())
    
    # 4. 添加人形机器人
    robot = scene.add_entity(
        gs.morphs.MJCF(file='xml/humanoid/humanoid.xml'),
        surface=gs.surfaces.Default(color=(0.8, 0.6, 0.4, 1.0))
    )
    
    # 5. 构建场景
    scene.build()
    
    # 6. 创建 MPC + WBC 控制器
    controller = MPCWBCController(
        num_dofs=19,
        dt=0.005,
        gait_frequency=1.25,
        mass=77.35,
        use_mpc=True,
        use_wbc=True
    )
    
    # 7. 设置目标速度
    target_velocity = np.array([0.5, 0.0, 0.0])  # 向前 0.5 m/s
    
    print("开始仿真...")
    print(f"目标速度: {target_velocity}")
    
    # 8. 主循环
    for step in range(5000):
        # 获取当前状态
        base_pos = robot.get_pos().cpu().numpy()
        base_quat = robot.get_quat().cpu().numpy()
        base_lin_vel = robot.get_vel().cpu().numpy()
        base_ang_vel = robot.get_ang().cpu().numpy()
        
        # 简化的姿态计算 (实际需要四元数转欧拉角)
        base_rpy = np.array([0.0, 0.0, 0.0])
        
        # 获取关节状态
        joint_pos = robot.get_dofs_position().cpu().numpy()
        joint_vel = robot.get_dofs_velocity().cpu().numpy()
        
        # 简化的足端位置 (实际需要运动学)
        left_foot_pos = base_pos + np.array([0.0, 0.1, -0.6])
        right_foot_pos = base_pos + np.array([0.0, -0.1, -0.6])
        
        # 更新控制器
        tau = controller.update(
            base_pos=base_pos,
            base_rpy=base_rpy,
            base_lin_vel=base_lin_vel,
            base_ang_vel=base_ang_vel,
            left_foot_pos=left_foot_pos,
            right_foot_pos=right_foot_pos,
            joint_pos=joint_pos,
            joint_vel=joint_vel,
            target_vel=target_velocity
        )
        
        # 应用力矩 (简化为位置控制)
        # 实际应该使用 robot.control_dofs_force(tau)
        target_positions = joint_pos + tau * 0.001
        robot.control_dofs_position(target_positions[:19])
        
        # 步进仿真
        scene.step()
        
        # 打印状态
        if step % 500 == 0:
            print(f"Step {step}: pos={base_pos[:2]}, vel={base_lin_vel[:2]}")
    
    print("仿真结束")


if __name__ == '__main__':
    main()
