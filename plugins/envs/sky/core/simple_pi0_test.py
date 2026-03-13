import torch
from lerobot.policies.factory import make_policy, make_policy_config

# Simple test to check PI 0.6 model creation
print("Testing PI 0.6 model creation...")

# Create minimal dataset metadata
ds_meta = {
    "stats": None,
    "observation_space": {"shape": (14,)},
    "action_space": {"shape": (6,)},
}

# Create PI 0.6 configuration object
pi0_config = make_policy_config(
    "pi0",
    paligemma_variant="gemma_2b",
    action_expert_variant="gemma_300m",
    dtype="float32",
    n_obs_steps=1,
    chunk_size=50,
    n_action_steps=50,
    max_state_dim=14,
    max_action_dim=6,
    num_inference_steps=10,
    image_resolution=(224, 224),
    gradient_checkpointing=True,
    compile_model=False,
)

try:
    print("Creating PI 0.6 policy...")
    policy = make_policy(cfg=pi0_config, ds_meta=ds_meta, rename_map=None)
    print(f"✓ PI 0.6 policy created successfully!")
    print(f"Policy type: {type(policy)}")
    print(f"Number of parameters: {sum(p.numel() for p in policy.parameters())}")
    
    # Test with dummy data
    print("\nTesting policy inference...")
    batch_size = 2
    dummy_images = torch.randn(batch_size, 6, 224, 224)  # Stereo images
    dummy_state = torch.randn(batch_size, 14)  # State vector
    
    pi0_obs = {
        "observation.images": dummy_images,
        "observation.state": dummy_state,
    }
    
    with torch.no_grad():
        actions = policy(**pi0_obs)
    
    print(f"✓ Policy inference successful!")
    print(f"Actions shape: {actions.shape}")
    print(f"Actions dtype: {actions.dtype}")
    
except Exception as e:
    print(f"✗ Error: {e}")
    import traceback
    traceback.print_exc()

print("\nTest completed!")