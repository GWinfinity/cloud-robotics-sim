import argparse
import pickle
from pathlib import Path

import torch
import genesis as gs
from dataclasses import dataclass

# Import lerobot components
from lerobot.policies.factory import make_policy, make_policy_config
from lerobot.envs.configs import EnvConfig
from lerobot.configs.types import PolicyFeature, FeatureType

from grasp_env import GraspEnv

# Custom EnvConfig for GraspEnv
@dataclass
class GraspEnvConfig(EnvConfig):
    task: str = "grasp_env"
    fps: int = 30
    
    def __post_init__(self):
        # Define features for PI 0.5
        self.features = {
            "observation.images": PolicyFeature(
                type=FeatureType.VISUAL,
                shape=(6, 224, 224),  # 6 channels for stereo RGB
            ),
            "observation.state": PolicyFeature(
                type=FeatureType.STATE,
                shape=(14,),  # Our observation space
            ),
            "action": PolicyFeature(
                type=FeatureType.ACTION,
                shape=(6,),  # Our action space
            ),
        }
        super().__post_init__()
    
    def gym_kwargs(self):
        """Required abstract method"""
        return {}


def load_rl_policy(env, train_cfg, log_dir):
    """Load reinforcement learning policy using lerobot."""
    # Find the latest checkpoint
    checkpoint_files = sorted([f for f in log_dir.iterdir() if f.name.startswith('checkpoint_') and f.is_dir()])
    if not checkpoint_files:
        raise FileNotFoundError(f"No checkpoint files found in {log_dir}")

    last_ckpt_dir = checkpoint_files[-1]
    checkpoint_path = last_ckpt_dir / "policy.bin"
    
    if not checkpoint_path.exists():
        checkpoint_path = last_ckpt_dir / "policy.pt"
    
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"No policy checkpoint found in {last_ckpt_dir}")
    
    print(f"Loaded RL checkpoint from {checkpoint_path}")
    
    # Create environment configuration for PI 0.5
    env_cfg_pi05 = GraspEnvConfig()
    
    # Create PI 0.5 configuration object
    pi05_config = make_policy_config(
        "pi05",
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
    
    # Create policy and load checkpoint
    policy = make_policy(
        cfg=pi05_config,
        env_cfg=env_cfg_pi05,
        rename_map=None,
    )
    
    # Load the checkpoint state
    state_dict = torch.load(checkpoint_path, map_location=gs.device)
    policy.load_state_dict(state_dict)
    policy.to(gs.device)
    policy.eval()

    return policy


def load_bc_policy(env, bc_cfg, log_dir):
    """Load behavior cloning policy using lerobot."""
    # Find the latest checkpoint
    checkpoint_files = sorted([f for f in log_dir.iterdir() if f.name.startswith('checkpoint_') and f.is_dir()])
    if not checkpoint_files:
        raise FileNotFoundError(f"No checkpoint files found in {log_dir}")

    last_ckpt_dir = checkpoint_files[-1]
    checkpoint_path = last_ckpt_dir / "policy.bin"
    
    if not checkpoint_path.exists():
        checkpoint_path = last_ckpt_dir / "policy.pt"
    
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"No policy checkpoint found in {last_ckpt_dir}")
    
    print(f"Loaded BC checkpoint from {checkpoint_path}")
    
    # Create environment configuration for PI 0.5
    env_cfg_pi05 = GraspEnvConfig()
    
    # Create PI 0.5 configuration object
    pi05_config = make_policy_config(
        "pi05",
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
    
    # Create policy and load checkpoint
    policy = make_policy(
        cfg=pi05_config,
        env_cfg=env_cfg_pi05,
        rename_map=None,
    )
    
    # Load the checkpoint state
    state_dict = torch.load(checkpoint_path, map_location=gs.device)
    policy.load_state_dict(state_dict)
    policy.to(gs.device)
    policy.eval()

    return policy


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--exp_name", type=str, default="grasp")
    parser.add_argument(
        "--stage",
        type=str,
        default="rl",
        choices=["rl", "bc"],
        help="Model type: 'rl' for reinforcement learning, 'bc' for behavior cloning",
    )
    parser.add_argument(
        "--record",
        action="store_true",
        help="Record stereo images as video during evaluation",
    )
    parser.add_argument(
        "--video_path",
        type=str,
        default=None,
        help="Path to save the video file (default: auto-generated)",
    )
    args = parser.parse_args()

    # Set PyTorch default dtype to float32 for better performance
    torch.set_default_dtype(torch.float32)

    gs.init()

    log_dir = Path("logs") / f"{args.exp_name + '_' + args.stage}"

    # Load configurations
    if args.stage == "rl":
        # For RL, load the standard configs
        env_cfg, reward_cfg, robot_cfg, rl_train_cfg, bc_train_cfg = pickle.load(open(log_dir / "cfgs.pkl", "rb"))
    else:
        # For BC, we need to load the configs and create BC config
        env_cfg, reward_cfg, robot_cfg, rl_train_cfg, bc_train_cfg = pickle.load(open(log_dir / "cfgs.pkl", "rb"))

    # set the max FPS for visualization
    env_cfg["max_visualize_FPS"] = 60
    # set the box collision
    env_cfg["box_collision"] = True
    # set the box fixed
    env_cfg["box_fixed"] = False
    # set the number of envs for evaluation
    env_cfg["num_envs"] = 10
    # for video recording
    env_cfg["visualize_camera"] = args.record

    env = GraspEnv(
        env_cfg=env_cfg,
        reward_cfg=reward_cfg,
        robot_cfg=robot_cfg,
        show_viewer=True,
    )

    # Load the appropriate policy based on model type
    if args.stage == "rl":
        policy = load_rl_policy(env, rl_train_cfg, log_dir)
    else:
        policy = load_bc_policy(env, bc_train_cfg, log_dir)
        policy.eval()

    obs, _ = env.reset()

    max_sim_step = int(env_cfg["episode_length_s"] * env_cfg["max_visualize_FPS"])

    with torch.no_grad():
        if args.record:
            print("Recording video...")
            env.vis_cam.start_recording()
            env.left_cam.start_recording()
            env.right_cam.start_recording()
        for step in range(max_sim_step):
            if args.stage == "rl":
                # Get PI 0.5 formatted observations for RL
                pi0_obs = env.get_pi0_observations()
                actions = policy(**pi0_obs)
            else:
                # Get PI 0.5 formatted observations for BC
                pi0_obs = env.get_pi0_observations()
                actions = policy(**pi0_obs)

                # Collect frame for video recording
                if args.record:
                    env.vis_cam.render()  # render the visualization camera

            obs, rews, dones, infos = env.step(actions)
        env.grasp_and_lift_demo()
        if args.record:
            print("Stopping video recording...")
            env.vis_cam.stop_recording(save_to_filename="video.mp4", fps=env_cfg["max_visualize_FPS"])
            env.left_cam.stop_recording(save_to_filename="left_cam.mp4", fps=env_cfg["max_visualize_FPS"])
            env.right_cam.stop_recording(save_to_filename="right_cam.mp4", fps=env_cfg["max_visualize_FPS"])


if __name__ == "__main__":
    main()

"""
# evaluation
# For reinforcement learning model:
python examples/manipulation/grasp_eval.py --stage=rl

# For behavior cloning model:
python examples/manipulation/grasp_eval.py --stage=bc

# With video recording:
python examples/manipulation/grasp_eval.py --stage=bc --record
"""
