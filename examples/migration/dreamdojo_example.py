"""
DreamDojo Plugin Usage Examples

This script demonstrates how to use the DreamDojo Genesis dataset plugin
within the genesis-cloud-sim framework.

Based on the original DreamDojo genesis_dreams module.
"""

import os
import sys
from pathlib import Path

# Add genesis-cloud-sim to path if needed
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


def example_1_basic_simulator():
    """Example 1: Basic Genesis Simulator Usage"""
    print("=" * 60)
    print("Example 1: Basic Genesis Simulator")
    print("=" * 60)
    
    try:
        from plugins.datasets.dreamdojo import (
            GenesisSimulator,
            GenesisSimulatorConfig,
            GenesisRobotType,
        )
        
        # Create configuration
        config = GenesisSimulatorConfig(
            robot_type=GenesisRobotType.HUMANOID,
            headless=True,  # Set to False to see the viewer
            device="cuda",
        )
        
        # Initialize simulator
        print("Initializing Genesis simulator...")
        simulator = GenesisSimulator(config)
        simulator.initialize()
        
        # Run a few steps
        print("Running simulation...")
        for i in range(10):
            # Random action
            import numpy as np
            action = np.random.randn(21) * 0.1
            
            obs, info = simulator.step(action)
            print(f"  Step {i+1}: obs shape={obs.shape}, "
                  f"joint_pos shape={info['joint_positions'].shape if info['joint_positions'] is not None else None}")
        
        # Get final state
        state = simulator.get_state()
        print(f"Final state keys: {list(state.keys())}")
        
        # Cleanup
        simulator.close()
        print("✓ Example 1 completed successfully\n")
        
    except ImportError as e:
        print(f"⚠ Genesis not installed: {e}")
    except Exception as e:
        print(f"✗ Example 1 failed: {e}\n")


def example_2_dataset_offline():
    """Example 2: Using Pre-generated Dataset"""
    print("=" * 60)
    print("Example 2: Pre-generated Dataset (Offline)")
    print("=" * 60)
    
    try:
        from plugins.datasets.dreamdojo import GenesisDataset
        import tempfile
        import h5py
        
        # Create a dummy dataset file
        print("Creating dummy dataset...")
        with tempfile.NamedTemporaryFile(suffix=".hdf5", delete=False) as f:
            dummy_path = f.name
        
        with h5py.File(dummy_path, "w") as f:
            # Create a few episodes
            for i in range(3):
                grp = f.create_group(f"episode_{i}")
                # Random observations and actions
                observations = np.random.randint(0, 255, (100, 480, 640, 3), dtype=np.uint8)
                actions = np.random.randn(100, 21).astype(np.float32)
                grp.create_dataset("observations", data=observations)
                grp.create_dataset("actions", data=actions)
        
        # Load dataset
        print("Loading dataset...")
        dataset = GenesisDataset(
            num_frames=81,
            robot_type="humanoid",
            pre_generated_path=dummy_path,
            use_online_sim=False,
        )
        
        print(f"Dataset length: {len(dataset)}")
        
        # Get a sample
        sample = dataset[0]
        print(f"Sample video shape: {sample['video'].shape}")
        print(f"Sample action shape: {sample['action'].shape}")
        
        # Cleanup
        dataset.close()
        os.unlink(dummy_path)
        print("✓ Example 2 completed successfully\n")
        
    except Exception as e:
        print(f"✗ Example 2 failed: {e}\n")


def example_3_dataset_online():
    """Example 3: Online Simulation Dataset"""
    print("=" * 60)
    print("Example 3: Online Simulation Dataset")
    print("=" * 60)
    
    try:
        import genesis as gs
        from plugins.datasets.dreamdojo import GenesisDataset
        
        print("Creating online dataset (requires Genesis)...")
        dataset = GenesisDataset(
            num_frames=10,  # Small for demo
            robot_type="humanoid",
            use_online_sim=True,
            num_episodes=2,
            device="cpu",  # Use CPU for demo
        )
        
        print(f"Dataset length: {len(dataset)}")
        
        # Get a sample
        sample = dataset[0]
        print(f"Sample video shape: {sample['video'].shape}")
        print(f"Sample action shape: {sample['action'].shape}")
        
        # Cleanup
        dataset.close()
        print("✓ Example 3 completed successfully\n")
        
    except ImportError:
        print("⚠ Genesis not installed, skipping online simulation\n")
    except Exception as e:
        print(f"✗ Example 3 failed: {e}\n")


def example_4_dataset_wrapper():
    """Example 4: Dataset Wrapper for Data Generation"""
    print("=" * 60)
    print("Example 4: Dataset Wrapper")
    print("=" * 60)
    
    try:
        import genesis as gs
        from plugins.datasets.dreamdojo import (
            GenesisSimulator,
            GenesisSimulatorConfig,
            GenesisRobotType,
            GenesisDatasetWrapper,
        )
        import tempfile
        
        # Create simulator
        print("Creating simulator...")
        config = GenesisSimulatorConfig(
            robot_type=GenesisRobotType.HUMANOID,
            headless=True,
            device="cpu",
        )
        simulator = GenesisSimulator(config)
        simulator.initialize()
        
        # Create wrapper
        print("Creating dataset wrapper...")
        wrapper = GenesisDatasetWrapper(
            simulator=simulator,
            num_episodes=2,  # Small for demo
            episode_length=20,  # Short episodes for demo
        )
        
        # Generate a single episode
        print("Generating episode...")
        episode = wrapper.generate_episode()
        print(f"Episode observations shape: {episode['observations'].shape}")
        print(f"Episode actions shape: {episode['actions'].shape}")
        
        # Generate full dataset
        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = os.path.join(tmpdir, "genesis_data.hdf5")
            print(f"Generating dataset to {save_path}...")
            wrapper.generate_dataset(save_path)
            print(f"Dataset saved, size: {os.path.getsize(save_path)} bytes")
        
        # Cleanup
        simulator.close()
        print("✓ Example 4 completed successfully\n")
        
    except ImportError:
        print("⚠ Genesis not installed, skipping dataset wrapper\n")
    except Exception as e:
        print(f"✗ Example 4 failed: {e}\n")


def example_5_factory_functions():
    """Example 5: Using Factory Functions"""
    print("=" * 60)
    print("Example 5: Factory Functions")
    print("=" * 60)
    
    try:
        from plugins.datasets.dreamdojo import (
            create_genesis_simulator,
            create_genesis_dataset,
            is_genesis_dataset,
        )
        
        # Create simulator using factory
        print("Creating simulator with factory...")
        simulator = create_genesis_simulator(
            robot_type="humanoid",
            headless=True,
            device="cpu",
        )
        print(f"Simulator created: {type(simulator).__name__}")
        
        # Test a step
        import numpy as np
        action = np.random.randn(21) * 0.1
        obs, info = simulator.step(action)
        print(f"Step successful, obs shape: {obs.shape}")
        
        simulator.close()
        
        # Test dataset detection
        test_paths = [
            "datasets/genesis_synthetic/data.hdf5",
            "datasets/gr1_unified",
            "datasets/g1",
        ]
        print("\nDataset path detection:")
        for path in test_paths:
            is_genesis = is_genesis_dataset(path)
            print(f"  {path}: {'✓ Genesis' if is_genesis else '✗ Other'}")
        
        print("✓ Example 5 completed successfully\n")
        
    except ImportError:
        print("⚠ Genesis not installed, skipping factory functions\n")
    except Exception as e:
        print(f"✗ Example 5 failed: {e}\n")


def example_6_robot_types():
    """Example 6: Different Robot Types"""
    print("=" * 60)
    print("Example 6: Robot Type Configuration")
    print("=" * 60)
    
    from plugins.datasets.dreamdojo import GenesisRobotType
    
    print("Available robot types:")
    for robot_type in GenesisRobotType:
        print(f"  - {robot_type.name}: {robot_type.value}")
    
    # Show action dimensions
    action_dims = {
        "humanoid": 21,
        "ur5": 6,
        "franka": 7,
        "g1": 29,
        "gr1": 32,
        "custom": 7,
    }
    
    print("\nAction dimensions:")
    for robot, dim in action_dims.items():
        print(f"  {robot}: {dim} DOF")
    
    print("✓ Example 6 completed successfully\n")


def main():
    """Run all examples"""
    print("\n" + "=" * 60)
    print("DreamDojo Plugin Examples")
    print("=" * 60 + "\n")
    
    examples = [
        ("Basic Simulator", example_1_basic_simulator),
        ("Offline Dataset", example_2_dataset_offline),
        ("Online Dataset", example_3_dataset_online),
        ("Dataset Wrapper", example_4_dataset_wrapper),
        ("Factory Functions", example_5_factory_functions),
        ("Robot Types", example_6_robot_types),
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
