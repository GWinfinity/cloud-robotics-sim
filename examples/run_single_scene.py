#!/usr/bin/env python3
"""
单场景运行示例

展示如何运行单个房间的仿真
"""

import sys
from pathlib import Path

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent.parent))

import genesis as gs
from scenes import create_scene


def main():
    print("=" * 60)
    print("Genesis Single Scene Example")
    print("=" * 60)
    
    # 初始化 Genesis
    print("\n1. Initializing Genesis...")
    gs.init(backend=gs.backends.CUDA)
    
    # 创建场景
    print("\n2. Creating scene...")
    scene = gs.Scene(
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(8, 8, 8),
            camera_lookat=(0, 0, 0),
            camera_fov=45,
        ),
        show_viewer=True,
    )
    
    # 添加地面
    scene.add_entity(
        morph=gs.morphs.Plane(),
        surface=gs.surfaces.Default(color=(0.9, 0.9, 0.9, 1.0))
    )
    
    # 创建客厅
    print("\n3. Building living room...")
    living_room = create_scene('living_room', position=(0, 0, 0))
    living_room.initialize(scene)
    living_room.build()
    
    print(f"   Created {len(living_room.furniture_entities)} furniture items")
    
    # 构建场景
    print("\n4. Building simulation...")
    scene.build()
    
    # 运行仿真
    print("\n5. Running simulation (press Ctrl+C to stop)...")
    print("-" * 60)
    
    try:
        for i in range(10000):
            scene.step()
            
            if i % 100 == 0:
                print(f"   Frame {i}")
                
    except KeyboardInterrupt:
        print("\n   Simulation stopped by user")
    
    print("\n" + "=" * 60)
    print("Simulation completed!")
    print("=" * 60)


if __name__ == '__main__':
    main()
