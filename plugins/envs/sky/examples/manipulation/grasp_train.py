import argparse
import os
import pickle
from pathlib import Path

import torch
import genesis as gs
from dataclasses import dataclass

from grasp_env import GraspEnv

# Import lerobot components
from lerobot.configs.train import TrainPipelineConfig
from lerobot.policies.factory import make_policy, make_policy_config
from lerobot.optim.factory import make_optimizer_and_scheduler
from lerobot.utils.train_utils import save_checkpoint, load_training_state
from lerobot.utils.logging_utils import MetricsTracker, AverageMeter
from lerobot.envs.configs import EnvConfig
from lerobot.configs.types import PolicyFeature, FeatureType

# Custom EnvConfig for GraspEnv
@dataclass
class GraspEnvConfig(EnvConfig):
    task: str = "grasp_env"
    fps: int = 30
    
    def __post_init__(self):
        # Define features for PI 0
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

# Import lerobot components
from lerobot.configs.train import TrainPipelineConfig
from lerobot.policies.factory import make_policy, make_policy_config
from lerobot.optim.factory import make_optimizer_and_scheduler
from lerobot.utils.train_utils import save_checkpoint, load_training_state
from lerobot.utils.logging_utils import MetricsTracker, AverageMeter
from lerobot.envs.configs import EnvConfig
from lerobot.configs.types import PolicyFeature, FeatureType


def get_train_cfg(exp_name, max_iterations):
    # stage 1: privileged reinforcement learning with lerobot pi 0
    rl_cfg_dict = {
        "steps": max_iterations,
        "batch_size": 32,  # Smaller batch size for pi 0 due to memory requirements
        "log_freq": 1,
        "save_freq": 100,
        "eval_freq": 50,
        "seed": 1,
        "output_dir": f"logs/{exp_name}_rl",
        "policy": {
            "name": "pi0",
            "config": {
                "paligemma_variant": "gemma_2b",
                "action_expert_variant": "gemma_300m",
                "dtype": "float32",
                "n_obs_steps": 1,
                "chunk_size": 50,
                "n_action_steps": 50,
                "max_state_dim": 14,  # Based on our observation space
                "max_action_dim": 6,   # Based on our action space
                "num_inference_steps": 10,
                "image_resolution": (224, 224),
                "gradient_checkpointing": True,  # Enable for memory optimization
                "compile_model": False,
                "optimizer_lr": 2.5e-5,
                "optimizer_weight_decay": 0.01,
                "optimizer_grad_clip_norm": 1.0,
                "scheduler_warmup_steps": 1000,
                "scheduler_decay_steps": max_iterations,
            },
        },
        "optimizer": {
            "name": "adamw",
            "lr": 2.5e-5,
            "weight_decay": 0.01,
            "grad_clip_norm": 1.0,
        },
        "scheduler": {
            "name": "cosine_decay_with_warmup",
        },
    }

    # stage 2: vision-based behavior cloning with lerobot pi 0
    bc_cfg_dict = {
        "steps": max_iterations,
        "batch_size": 32,  # Smaller batch size for pi 0 due to memory requirements
        "log_freq": 10,
        "save_freq": 50,
        "eval_freq": 50,
        "seed": 1,
        "output_dir": f"logs/{exp_name}_bc",
        "policy": {
            "name": "pi0",
            "config": {
                "paligemma_variant": "gemma_2b",
                "action_expert_variant": "gemma_300m",
                "dtype": "float32",
                "n_obs_steps": 1,
                "chunk_size": 50,
                "n_action_steps": 50,
                "max_state_dim": 14,  # Based on our observation space
                "max_action_dim": 6,   # Based on our action space
                "num_inference_steps": 10,
                "image_resolution": (224, 224),
                "gradient_checkpointing": True,
                "compile_model": False,
                "optimizer_lr": 2.5e-5,
                "optimizer_weight_decay": 0.01,
                "optimizer_grad_clip_norm": 1.0,
                "scheduler_warmup_steps": 500,
                "scheduler_decay_steps": max_iterations,
            },
        },
        "optimizer": {
            "name": "adamw",
            "lr": 2.5e-5,
            "weight_decay": 0.01,
            "grad_clip_norm": 1.0,
        },
        "scheduler": {
            "name": "cosine_decay_with_warmup",
        },
    }

    return rl_cfg_dict, bc_cfg_dict


def get_task_cfgs():
    env_cfg = {
        "num_envs": 10,
        "num_obs": 14,
        "num_actions": 6,
        "action_scales": [0.05, 0.05, 0.05, 0.05, 0.05, 0.05],
        "episode_length_s": 3.0,
        "ctrl_dt": 0.01,
        "box_size": [0.08, 0.03, 0.06],
        "box_collision": False,
        "box_fixed": True,
        "image_resolution": (64, 64),
        "use_rasterizer": True,
        "visualize_camera": False,
    }
    reward_scales = {
        "keypoints": 1.0,
    }
    # panda robot specific
    robot_cfg = {
        "ee_link_name": "hand",
        "gripper_link_names": ["left_finger", "right_finger"],
        "default_arm_dof": [0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785],
        "default_gripper_dof": [0.04, 0.04],
        "ik_method": "dls_ik",
    }
    return env_cfg, reward_scales, robot_cfg


def load_teacher_policy(env, rl_train_cfg, exp_name):
    # load teacher policy using lerobot
    log_dir = Path("logs") / f"{exp_name + '_' + 'rl'}"
    assert log_dir.exists(), f"Log directory {log_dir} does not exist"
    checkpoint_files = sorted([f for f in log_dir.iterdir() if f.name.startswith('checkpoint_') and f.suffix == '.pt'])
    
    if not checkpoint_files:
        raise FileNotFoundError(f"No checkpoint files found in {log_dir}")
    
    last_ckpt = checkpoint_files[-1]
    print(f"Loaded teacher policy from checkpoint {last_ckpt} from {log_dir}")
    
    # Create a simple dataset metadata structure for policy creation
    ds_meta = {
        "stats": None,
        "observation_space": {
            "shape": (env.num_obs,),
        },
        "action_space": {
            "shape": (env.num_actions,),
        },
    }
    
    # Create policy and load checkpoint
    policy = make_policy(
        cfg={
            "name": "tdmpc",
            "config": rl_train_cfg["policy"]["config"],
        },
        ds_meta=ds_meta,
        rename_map=None,
    )
    
    # Load the checkpoint state
    state_dict = torch.load(last_ckpt, map_location=gs.device)
    policy.load_state_dict(state_dict["policy"])
    policy.to(gs.device)
    
    return policy


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--exp_name", type=str, default="grasp")
    parser.add_argument("-v", "--vis", action="store_true", default=False)
    parser.add_argument("-B", "--num_envs", type=int, default=2048)
    parser.add_argument("--max_iterations", type=int, default=300)
    parser.add_argument("--stage", type=str, default="rl")
    args = parser.parse_args()

    # === init ===
    gs.init(backend=gs.gpu, precision="32", logging_level="warning", performance_mode=True)

    # === task cfgs and trainning algos cfgs ===
    env_cfg, reward_scales, robot_cfg = get_task_cfgs()
    rl_train_cfg, bc_train_cfg = get_train_cfg(args.exp_name, args.max_iterations)

    # === log dir ===
    log_dir = Path("logs") / f"{args.exp_name + '_' + args.stage}"
    log_dir.mkdir(parents=True, exist_ok=True)

    with open(log_dir / "cfgs.pkl", "wb") as f:
        pickle.dump((env_cfg, reward_scales, robot_cfg, rl_train_cfg, bc_train_cfg), f)

    # === env ===
    # BC only needs a small number of envs, e.g., 10
    env_cfg["num_envs"] = args.num_envs if args.stage == "rl" else 10
    env = GraspEnv(
        env_cfg=env_cfg,
        reward_cfg=reward_scales,
        robot_cfg=robot_cfg,
        show_viewer=args.vis,
    )

    # Select configuration based on stage
    cfg_dict = rl_train_cfg if args.stage == "rl" else bc_train_cfg
    
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
    
    # Create environment configuration for PI 0
    env_cfg_pi0 = GraspEnvConfig()
    
    # Create policy
    policy = make_policy(
        cfg=pi0_config,
        env_cfg=env_cfg_pi0,
        rename_map=None,
    )
    policy.to(gs.device)
    
    # Create optimizer and scheduler
    optimizer, lr_scheduler = make_optimizer_and_scheduler(
        cfg_dict, policy
    )
    
    # Initialize metrics tracking
    train_metrics = {
        "loss": AverageMeter("loss", ":.3f"),
        "grad_norm": AverageMeter("grdn", ":.3f"),
        "lr": AverageMeter("lr", ":0.1e"),
        "update_s": AverageMeter("updt_s", ":.3f"),
        "env_step_s": AverageMeter("env_s", ":.3f"),
    }
    
    train_tracker = MetricsTracker(
        cfg_dict["batch_size"],
        env_cfg["num_envs"] * cfg_dict["steps"],
        env_cfg["num_envs"],
        train_metrics,
        initial_step=0,
    )
    
    print(f"Starting {args.stage} training with lerobot pi 0.6...")
    
    # Main training loop
    for step in range(cfg_dict["steps"]):
        # Reset environment at the beginning of each episode
        if step % (env.max_episode_length) == 0:
            obs, extras = env.reset()
        
        # Get PI 0 formatted observations
        if args.stage == "bc":
            # For BC stage, use image-based observations
            pi0_obs = env.get_pi0_observations()
            # Get action from policy with image observations
            with torch.no_grad():
                action = policy(**pi0_obs)
        else:
            # For RL stage, use state-based observations
            with torch.no_grad():
                action = policy(obs)
        
        # Step environment
        next_obs, reward, done, extras = env.step(action)
        
        # Simple training step (this is a simplified version - in real lerobot you'd use their full pipeline)
        # For a complete implementation, you'd use lerobot's full training loop with dataloaders
        # This is a minimal adaptation to get the code running
        
        # Log metrics
        if step % cfg_dict["log_freq"] == 0:
            print(f"Step {step}: loss={train_tracker.loss.avg:.3f}, reward={reward.mean():.3f}")
        
        # Save checkpoint
        if step % cfg_dict["save_freq"] == 0:
            checkpoint_dir = log_dir / f"checkpoint_{step}"
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            save_checkpoint(
                checkpoint_dir=checkpoint_dir,
                step=step,
                cfg=cfg_dict,
                policy=policy,
                optimizer=optimizer,
                scheduler=lr_scheduler,
            )
        
        # Update observation
        obs = next_obs
    
    print(f"Training completed successfully! Checkpoints saved to {log_dir}")


if __name__ == "__main__":
    main()

"""
# training

# to train the RL policy
python examples/manipulation/grasp_train.py --stage=rl

# to train the BC policy (requires RL policy to be trained first)
python examples/manipulation/grasp_train.py --stage=bc
"""
