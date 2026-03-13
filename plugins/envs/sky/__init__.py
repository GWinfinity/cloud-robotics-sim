"""
sky Plugin - Genesis: 通用物理引擎和机器人仿真平台

来源: genesis-sky (https://github.com/Genesis-Embodied-AI/Genesis)
核心实现: 高性能通用物理引擎，支持多种物理求解器和渲染

Genesis 是:
1. 通用物理引擎 - 从头重建，支持多种材料和物理现象
2. 机器人仿真平台 - 轻量、超快、Pythonic、用户友好
3. 照片级渲染系统 - 强大的实时光追渲染
4. 生成数据引擎 - 将自然语言描述转换为多模态数据

核心特性:
- 速度: 单RTX 4090上Franka机器人仿真超过4300万FPS
- 跨平台: Linux, macOS, Windows, 支持CPU/NVIDIA/AMD/Apple Metal
- 多物理求解器: 刚体、MPM、SPH、FEM、PBD、稳定流体
- 材质模型: 刚体、液体、气体、可变形物体、薄壳、颗粒材料
- 机器人兼容: 机械臂、腿式机器人、无人机、软体机器人
- 可微性: 支持MPM和Tool Solver的可微仿真

支持的求解器:
- Rigid: 刚体和关节体求解器
- MPM: Material Point Method (材料点法)
- SPH: Smoothed Particle Hydrodynamics (光滑粒子流体动力学)
- FEM: Finite Element Method (有限元法)
- PBD: Position-Based Dynamics (基于位置的动力学)
- Stable Fluid: 稳定流体求解器

支持的渲染器:
- Rasterization: 光栅化渲染
- Ray Tracing: 实时光追渲染
"""

__version__ = "0.3.11"
__source__ = "genesis-sky"

# 尝试导入核心组件
try:
    from .core.genesis import (
        Scene, Entity, Material,
        RigidSolver, MPMSolver, SPHSolver, FEMSolver, PBDSolver,
        Render, Camera, Light
    )
    
    __all__ = [
        'Scene',
        'Entity',
        'Material',
        'RigidSolver',
        'MPMSolver',
        'SPHSolver',
        'FEMSolver',
        'PBDSolver',
        'Render',
        'Camera',
        'Light',
    ]
except ImportError:
    # 开发模式，部分依赖可能未安装
    __all__ = []
