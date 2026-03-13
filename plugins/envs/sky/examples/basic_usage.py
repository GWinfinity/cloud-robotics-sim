"""
Sky (Genesis) Plugin - 基础使用示例

演示如何使用 Genesis 物理引擎进行机器人物理仿真
"""

import numpy as np
from pathlib import Path

# 导入插件
try:
    from cloud_robotics_sim.plugins.envs.sky import Scene
except ImportError:
    # 直接导入 (开发模式)
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))


def example_rigid_body():
    """示例1: 刚体仿真"""
    print("=" * 60)
    print("示例1: 刚体仿真")
    print("=" * 60)
    
    print("""
Genesis 刚体求解器:
- 支持任意形状的几何体
- 碰撞检测和响应
- 摩擦和恢复系数
- 关节约束

基本流程:
  1. 创建场景
  2. 添加刚体实体
  3. 设置材质属性
  4. 运行仿真

代码示例:
  ```python
  import genesis as gs
  
  # 初始化
  gs.init(backend=gs.backends.CUDA)
  
  # 创建场景
  scene = gs.Scene()
  
  # 添加地面
  plane = scene.add_entity(
      morph=gs.morphs.Plane(),
      material=gs.materials.Rigid()
  )
  
  # 添加球体
  sphere = scene.add_entity(
      morph=gs.morphs.Sphere(radius=0.1),
      material=gs.materials.Rigid(),
      pos=(0, 0, 1.0)
  )
  
  # 构建并运行
  scene.build()
  for _ in range(1000):
      scene.step()
  ```

材质属性:
  - 密度: 影响质量计算
  - 摩擦: 表面摩擦系数
  - 恢复: 弹性碰撞系数
""")
    
    print("\n✓ 刚体仿真说明完成\n")


def example_robot_simulation():
    """示例2: 机器人仿真"""
    print("=" * 60)
    print("示例2: 机器人仿真")
    print("=" * 60)
    
    print("""
Genesis 支持多种机器人模型:
- MJCF (.xml) - MuJoCo格式
- URDF - 通用机器人描述格式
- 支持机械臂、腿式机器人、无人机等

加载机器人:
  ```python
  # 加载Franka机器人
  robot = scene.add_entity(
      morph=gs.morphs.MJCF(
          file='xml/franka_emika_panda/panda.xml'
      ),
      pos=(0, 0, 0.5)
  )
  
  # 构建场景
  scene.build()
  
  # 控制机器人
  # 位置控制
  target_pos = [0.5, -0.5, 0, 0, 0, 0, 0]
  robot.control_dofs_position(target_pos)
  
  # 速度控制
  robot.control_dofs_velocity([0.1, 0, 0, 0, 0, 0, 0])
  
  # 力矩控制
  robot.control_dofs_force([10, 0, 0, 0, 0, 0, 0])
  ```

获取状态:
  - 关节位置: robot.get_dofs_position()
  - 关节速度: robot.get_dofs_velocity()
  - 末端位置: robot.get_link('hand').get_pos()
""")
    
    print("\n支持的控制模式:")
    modes = [
        ("位置控制", "PD控制到目标位置"),
        ("速度控制", "直接控制关节速度"),
        ("力矩控制", "直接控制力矩"),
    ]
    for mode, desc in modes:
        print(f"  {mode}: {desc}")
    
    print("\n✓ 机器人仿真说明完成\n")


def example_mpm_deformable():
    """示例3: MPM可变形物体"""
    print("=" * 60)
    print("示例3: MPM可变形物体仿真")
    print("=" * 60)
    
    print("""
MPM (Material Point Method) - 材料点法:
- 适合模拟可变形物体
- 支持弹塑性材料
- 雪、沙子、粘土等材料

双网格方法:
  1. 粒子到网格 (P2G)
     将粒子质量、动量转移到背景网格
  
  2. 网格更新
     在网格上求解动量方程
     计算速度更新
  
  3. 网格到粒子 (G2P)
     将网格速度插值回粒子
     更新粒子位置和速度

代码示例:
  ```python
  # 创建可变形立方体
  elastic_cube = scene.add_entity(
      morph=gs.morphs.Box(size=(0.1, 0.1, 0.1)),
      material=gs.materials.MPM.Elastic(
          E=1e5,      # 杨氏模量
          nu=0.3,     # 泊松比
          density=1000
      ),
      pos=(0, 0, 1.0)
  )
  
  # 创建雪
  snow = scene.add_entity(
      morph=gs.morphs.Box(size=(0.2, 0.2, 0.2)),
      material=gs.materials.MPM.Snow(
          E=1e4,
          nu=0.2,
          density=200
      )
  )
  ```

材料类型:
  - Elastic: 弹性体 (橡胶、果冻)
  - Snow: 雪
  - Sand: 沙子
  - Clay: 粘土
""")
    
    print("\nMPM参数:")
    params = [
        ("E (杨氏模量)", "刚度，越大越硬"),
        ("nu (泊松比)", "横向变形，0-0.5"),
        ("density", "密度"),
    ]
    for param, desc in params:
        print(f"  {param}: {desc}")
    
    print("\n✓ MPM可变形物体说明完成\n")


def example_sph_fluid():
    """示例4: SPH流体仿真"""
    print("=" * 60)
    print("示例4: SPH流体仿真")
    print("=" * 60)
    
    print("""
SPH (Smoothed Particle Hydrodynamics):
- 无网格拉格朗日方法
- 适合液体和气体
- 水、油、血液、烟雾

基本原理:
  通过核函数在粒子间插值
  A(r) = sum(m_j / rho_j * A_j * W(r - r_j, h))

其中:
  - W: 光滑核函数
  - h: 光滑长度
  - m: 粒子质量
  - rho: 密度

代码示例:
  ```python
  # 创建水
  water = scene.add_entity(
      morph=gs.morphs.Box(size=(0.5, 0.5, 0.5)),
      material=gs.materials.SPH.Liquid(
          density=1000,     # kg/m³
          viscosity=0.001,  # Pa·s
          surface_tension=0.07
      ),
      pos=(0, 0, 1.0)
  )
  ```

流体属性:
  - 密度: 质量/体积
  - 粘度: 流动阻力
  - 表面张力: 液面收缩力

应用场景:
  - 水池、杯子倒水
  - 液体飞溅
  - 流体与固体交互
""")
    
    print("\n不同液体参数:")
    fluids = [
        ("水", "1000", "0.001"),
        ("油", "800", "0.1"),
        ("蜂蜜", "1400", "10"),
        ("血液", "1060", "0.003"),
    ]
    print(f"  {'液体':10s} {'密度(kg/m³)':15s} {'粘度(Pa·s)'}")
    print("  " + "-" * 40)
    for fluid, density, viscosity in fluids:
        print(f"  {fluid:10s} {density:15s} {viscosity}")
    
    print("\n✓ SPH流体仿真说明完成\n")


def example_rendering():
    """示例5: 渲染"""
    print("=" * 60)
    print("示例5: 渲染系统")
    print("=" * 60)
    
    print("""
Genesis 渲染系统:

1. 光栅化渲染 (Rasterization)
   - 实时性能
   - OpenGL/Vulkan后端
   - 适合交互式可视化

2. 光追渲染 (Ray Tracing)
   - 照片级质量
   - 全局光照
   - 软阴影
   - 反射/折射

代码示例:
  ```python
  # 创建带相机的场景
  scene = gs.Scene(
      viewer_options=gs.options.ViewerOptions(
          camera_pos=(3, 3, 3),
          camera_lookat=(0, 0, 0),
          res=(1280, 720)
      ),
      show_viewer=True
  )
  
  # 添加相机
  camera = scene.add_camera(
      res=(640, 480),
      pos=(2, 2, 2),
      lookat=(0, 0, 0)
  )
  
  # 渲染
  rgb, depth, segmentation = camera.render()
  ```

渲染输出:
  - RGB图像
  - 深度图
  - 语义分割
  - 法线图
  - 光流 (可选)
""")
    
    print("\n渲染对比:")
    renderers = [
        ("光栅化", "实时", "中等", "交互"),
        ("光追", "较慢", "高", "最终输出"),
    ]
    print(f"  {'渲染器':10s} {'速度':10s} {'质量':10s} {'用途'}")
    print("  " + "-" * 45)
    for name, speed, quality, use in renderers:
        print(f"  {name:10s} {speed:10s} {quality:10s} {use}")
    
    print("\n✓ 渲染系统说明完成\n")


def example_performance():
    """示例6: 性能优势"""
    print("=" * 60)
    print("示例6: Genesis性能优势")
    print("=" * 60)
    
    print("""
Genesis 性能特点:

1. 极速仿真
   - 单RTX 4090: 4300万FPS (Franka机器人)
   - 比实时快43万倍
   - 适合大规模并行训练

2. GPU加速
   - CUDA内核优化
   - Taichi自动并行化
   - 内存高效管理

3. 多后端支持
   - NVIDIA CUDA
   - AMD ROCm
   - Apple Metal
   - CPU (fallback)

性能对比 (Franka机器人):
  ┌─────────────┬────────────┬─────────────┐
  │  仿真器      │  FPS       │  相对速度    │
  ├─────────────┼────────────┼─────────────┤
  │  Genesis    │  43M       │  1.0x       │
  │  Isaac Gym  │  1M        │  0.02x      │
  │  PyBullet   │  10K       │  0.0002x    │
  │  MuJoCo     │  50K       │  0.001x     │
  └─────────────┴────────────┴─────────────┘

适用场景:
  - 大规模RL训练 (数千并行环境)
  - 域随机化
  - 快速原型开发
  - 生成数据
""")
    
    print("\n✓ 性能优势说明完成\n")


def main():
    """运行所有示例"""
    print("\n" + "=" * 60)
    print("Sky (Genesis) Plugin - 使用示例")
    print("=" * 60 + "\n")
    
    # 检查依赖
    try:
        import taichi
        print(f"Taichi 版本: {taichi.__version__}")
    except ImportError:
        print("警告: Taichi 未安装 (pip install taichi)")
    
    try:
        import torch
        print(f"PyTorch 版本: {torch.__version__}")
    except ImportError:
        print("警告: PyTorch 未安装")
    
    print()
    
    # 运行示例
    examples = [
        ("刚体仿真", example_rigid_body),
        ("机器人仿真", example_robot_simulation),
        ("MPM可变形物体", example_mpm_deformable),
        ("SPH流体仿真", example_sph_fluid),
        ("渲染系统", example_rendering),
        ("性能优势", example_performance),
    ]
    
    for name, func in examples:
        try:
            func()
        except Exception as e:
            print(f"{name} 示例失败: {e}\n")
    
    print("=" * 60)
    print("所有示例运行完成!")
    print("=" * 60)


if __name__ == "__main__":
    main()
