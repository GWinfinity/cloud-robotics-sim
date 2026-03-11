#!/usr/bin/env python3
"""
公寓多场景运行示例

展示如何运行包含多个房间的公寓仿真
"""

import sys
from pathlib import Path

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent.parent))

import genesis as gs
from scenes import create_scene


def main():
    print("=" * 60)
    print("Genesis Apartment Simulation Example")
    print("=" * 60)
    
    # 初始化 Genesis
    print("\n1. Initializing Genesis...")
    gs.init(backend=gs.backends.CUDA)
    
    # 创建场景
    print("\n2. Creating scene...")
    scene = gs.Scene(
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(15, 15, 15),
            camera_lookat=(0, 0, 0),
            camera_fov=60,
        ),
        show_viewer=True,
    )
    
    # 添加地面
    scene.add_entity(
        morph=gs.morphs.Plane(),
        surface=gs.surfaces.Default(color=(0.9, 0.9, 0.9, 1.0))
    )
    
    # 创建所有房间
    print("\n3. Building apartment rooms...")
    
    rooms = [
        ('living_room', (0, 0, 0)),
        ('bedroom', (-8, 0, 0)),
        ('kitchen', (8, 0, 0)),
        ('bathroom', (0, -6, 0)),
        ('laundry_room', (8, -4, 0)),
    ]
    
    total_furniture = 0
    for room_type, position in rooms:
        print(f"   Building {room_type} at {position}...")
        room = create_scene(room_type, position=position)
        room.initialize(scene)
        room.build()
        total_furniture += len(room.furniture_entities)
    
    print(f"\n   Total rooms: {len(rooms)}")
    print(f"   Total furniture items: {total_furniture}")
    
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
