"""Minimal environment creation test."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root / "src"))

from cloud_robotics_sim.runtime import make_env_from_config
import yaml

logging.basicConfig(level=logging.INFO)

with open("configs/franka_pickplace.yaml") as f:
    config = yaml.safe_load(f)

# Force headless to avoid GUI and reduce resolution for speed
config["environment"]["simulation"]["headless"] = True
config["environment"]["simulation"]["resolution"] = [320, 240]
config["environment"]["simulation"]["substeps"] = 1

print("Creating environment...")
env = make_env_from_config(config)
print("Environment created. Resetting...")
obs, info = env.reset()
print("Reset done. Steping once...")
obs, reward, terminated, truncated, info = env.step(env.action_space["low"])
print(f"Step done. reward={reward}, terminated={terminated}, truncated={truncated}, info={info}")
print("SUCCESS")
