"""Test genesis-world init and Scene creation.

This test is focused on getting the project running.
"""
import genesis as gs

from cloud_robotics_sim.utils.genesis_compat import get_genesis_backend

# Test 1: init with CPU backend
print("Test 1: gs.init with cpu backend...")
try:
    backend = get_genesis_backend("cpu")
    gs.init(backend=backend, precision="32")
    print("✅ gs.init OK")
    print(f"   device: {gs.device}")
except Exception as e:
    print(f"❌ gs.init failed: {e}")

# Test 2: basic scene
print("\nTest 2: Scene creation...")
try:
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=0.01,
            substeps=2,
        ),
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(3.0, 0.0, 3.0),
            camera_lookat=(0.0, 0.0, 0.5),
        ),
        show_viewer=False,
        vis_options=gs.options.VisOptions(
            show_world_frame=True,
        ),
    )
    print("✅ Scene created")
    
    # Add a box
    box = scene.add_entity(
        morph=gs.morphs.Box(size=(0.5, 0.5, 0.5), pos=(0.0, 0.0, 0.5)),
    )
    print(f"✅ Box entity added: {box}")
    
    # Build scene
    scene.build()
    print("✅ Scene built")
    
    # Run a few steps
    for i in range(10):
        scene.step()
    print("✅ 10 simulation steps completed")
    
    gs.destroy()
    print("✅ gs.destroy() OK")
except Exception as e:
    print(f"❌ Failed: {e}")
    import traceback
    traceback.print_exc()
    gs.destroy()
