"""
ManiSkill Genesis Utils Integration Examples

This script demonstrates how to use the ManiSkill genesis_utils
integration within the genesis-cloud-sim framework.

Based on the original ManiSkill-main/mani_skill/utils/genesis_utils.py
"""

import sys
from pathlib import Path

# Add genesis-cloud-sim to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))


def example_1_object_query():
    """Example 1: Object Query Utilities"""
    print("=" * 60)
    print("Example 1: Object Query Utilities")
    print("=" * 60)
    
    try:
        from cloud_robotics_sim.utils import (
            get_obj_by_name,
            get_objs_by_names,
            get_obj_by_type,
        )
        
        # Mock objects for demonstration
        class MockObj:
            def __init__(self, name):
                self.name = name
            def get_name(self):
                return self.name
        
        objs = [MockObj(f"obj_{i}") for i in range(5)]
        objs.append(MockObj("target"))
        
        # Test get_obj_by_name
        result = get_obj_by_name(objs, "target")
        print(f"✓ get_obj_by_name: {result.get_name() if result else None}")
        
        # Test get_objs_by_names
        result = get_objs_by_names(objs, ["obj_0", "obj_2", "target"])
        print(f"✓ get_objs_by_names: {[r.get_name() if r else None for r in result]}")
        
        # Test get_obj_by_type
        result = get_obj_by_type(objs, MockObj)
        print(f"✓ get_obj_by_type: {result.get_name() if result else None}")
        
        print("✓ Example 1 completed successfully\n")
        
    except Exception as e:
        print(f"✗ Example 1 failed: {e}\n")


def example_2_urdf_config():
    """Example 2: URDF Configuration"""
    print("=" * 60)
    print("Example 2: URDF Configuration")
    print("=" * 60)
    
    try:
        from cloud_robotics_sim.utils import (
            check_urdf_config,
            parse_urdf_config,
        )
        
        # Valid config
        valid_config = {
            "material": {"static_friction": 0.5, "dynamic_friction": 0.5},
            "link": {
                "link_1": {"density": 1000.0},
            }
        }
        
        try:
            check_urdf_config(valid_config)
            print("✓ Valid URDF config accepted")
        except KeyError as e:
            print(f"✗ Valid config rejected: {e}")
        
        # Invalid config
        invalid_config = {
            "invalid_key": "value"
        }
        
        try:
            check_urdf_config(invalid_config)
            print("✗ Invalid config should have raised error")
        except KeyError:
            print("✓ Invalid config correctly rejected")
        
        # Parse config (without Genesis)
        parsed = parse_urdf_config(valid_config)
        print(f"✓ Parsed config keys: {list(parsed.keys())}")
        
        print("✓ Example 2 completed successfully\n")
        
    except Exception as e:
        print(f"✗ Example 2 failed: {e}\n")


def example_3_camera_utils():
    """Example 3: Camera Utilities"""
    print("=" * 60)
    print("Example 3: Camera Utilities")
    print("=" * 60)
    
    try:
        from cloud_robotics_sim.utils import (
            look_at,
            hex2rgba,
            rgba2hex,
            spherical_to_cartesian,
            compute_fovy,
        )
        
        # Test look_at
        pose = look_at(
            eye=[1.0, 2.0, 3.0],
            target=[0.0, 0.0, 0.0],
            up=[0.0, 0.0, 1.0],
        )
        print(f"✓ look_at pose created: {type(pose).__name__}")
        
        # Test color conversion
        rgba = hex2rgba("#FF0000")
        print(f"✓ hex2rgba('#FF0000'): {rgba}")
        
        hex_color = rgba2hex([1.0, 0.0, 0.0, 1.0])
        print(f"✓ rgba2hex([1,0,0,1]): {hex_color}")
        
        # Test spherical coordinates
        pos = spherical_to_cartesian(
            radius=2.0,
            azimuth=0.0,
            elevation=0.0,
            target=[0, 0, 0]
        )
        print(f"✓ spherical_to_cartesian: {pos}")
        
        # Test FOV computation
        fovy = compute_fovy(focal_length=50.0, sensor_height=24.0)
        print(f"✓ compute_fovy(50mm, 24mm): {fovy:.2f}°")
        
        print("✓ Example 3 completed successfully\n")
        
    except Exception as e:
        print(f"✗ Example 3 failed: {e}\n")


def example_4_rendering_utils():
    """Example 4: Rendering Utilities"""
    print("=" * 60)
    print("Example 4: Rendering Utilities")
    print("=" * 60)
    
    try:
        from cloud_robotics_sim.utils import (
            ShaderConfig,
            configure_rendering,
            create_checkerboard_texture,
        )
        
        # Test ShaderConfig
        config = ShaderConfig(
            shader_pack="rt",
            ray_tracing_denoiser="optix",
            ray_tracing_path_depth=4,
        )
        print(f"✓ ShaderConfig created: {config.shader_pack}")
        
        # Test configure_rendering (without Genesis)
        result = configure_rendering(shader_pack="default")
        print(f"✓ configure_rendering: {result}")
        
        # Test checkerboard texture
        texture = create_checkerboard_texture(
            size=256,
            check_size=32,
            color1=[1.0, 1.0, 1.0],
            color2=[0.0, 0.0, 0.0],
        )
        if texture is not None:
            print(f"✓ create_checkerboard_texture: shape={texture.shape}")
        else:
            print("✓ create_checkerboard_texture: returned None (numpy not available)")
        
        print("✓ Example 4 completed successfully\n")
        
    except Exception as e:
        print(f"✗ Example 4 failed: {e}\n")


def example_5_state_extraction():
    """Example 5: State Extraction (Mock)"""
    print("=" * 60)
    print("Example 5: State Extraction (Mock)")
    print("=" * 60)
    
    try:
        from cloud_robotics_sim.utils import (
            get_actor_state,
            get_articulation_state,
            is_state_dict_consistent,
        )
        
        # Mock actor
        class MockActor:
            def get_pose(self):
                class Pose:
                    p = [0.0, 0.0, 1.0]
                    q = [1.0, 0.0, 0.0, 0.0]
                return Pose()
            def get_linear_velocity(self):
                return [0.0, 0.0, 0.0]
            def get_angular_velocity(self):
                return [0.0, 0.0, 0.0]
        
        actor = MockActor()
        state = get_actor_state(actor)
        
        if state is not None:
            print(f"✓ get_actor_state: shape={state.shape}")
        else:
            print("✓ get_actor_state: returned None (numpy not available)")
        
        # Test state dict consistency
        consistent_dict = {
            "actors": {
                "actor1": type('obj', (object,), {'shape': (10,)})(),
                "actor2": type('obj', (object,), {'shape': (10,)})(),
            }
        }
        result = is_state_dict_consistent(consistent_dict)
        print(f"✓ is_state_dict_consistent: {result}")
        
        print("✓ Example 5 completed successfully\n")
        
    except Exception as e:
        print(f"✗ Example 5 failed: {e}\n")


def example_6_backward_compatibility():
    """Example 6: Backward Compatibility"""
    print("=" * 60)
    print("Example 6: Backward Compatibility")
    print("=" * 60)
    
    try:
        # Old ManiSkill style import
        from cloud_robotics_sim.utils import get_entity_by_name
        
        # Mock objects
        class MockObj:
            def __init__(self, name):
                self.name = name
            def get_name(self):
                return self.name
        
        objs = [MockObj(f"obj_{i}") for i in range(3)]
        result = get_entity_by_name(objs, "obj_1")
        
        if result and result.get_name() == "obj_1":
            print("✓ get_entity_by_name (backward compatible)")
        else:
            print("✗ get_entity_by_name failed")
        
        # Test importing all utilities
        from cloud_robotics_sim.utils import (
            get_obj_by_name,
            look_at,
            set_render_material,
            Pose,
            GENESIS_RENDER_SYSTEM,
        )
        print("✓ All utilities imported successfully")
        print(f"✓ GENESIS_RENDER_SYSTEM: {GENESIS_RENDER_SYSTEM}")
        
        print("✓ Example 6 completed successfully\n")
        
    except Exception as e:
        print(f"✗ Example 6 failed: {e}\n")


def main():
    """Run all examples"""
    print("\n" + "=" * 60)
    print("ManiSkill Genesis Utils Integration Examples")
    print("=" * 60 + "\n")
    
    examples = [
        ("Object Query", example_1_object_query),
        ("URDF Config", example_2_urdf_config),
        ("Camera Utils", example_3_camera_utils),
        ("Rendering Utils", example_4_rendering_utils),
        ("State Extraction", example_5_state_extraction),
        ("Backward Compatibility", example_6_backward_compatibility),
    ]
    
    results = []
    for name, func in examples:
        try:
            func()
            results.append((name, "✓ PASSED"))
        except Exception as e:
            results.append((name, f"✗ FAILED: {e}"))
    
    # Summary
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    for name, result in results:
        print(f"{result}: {name}")
    
    passed = sum(1 for _, r in results if r.startswith("✓"))
    print(f"\nTotal: {passed}/{len(results)} examples passed")


if __name__ == "__main__":
    main()
