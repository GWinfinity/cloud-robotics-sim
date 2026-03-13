import lerobot

# Explore the RL module
print("\n=== Exploring lerobot.rl module ===")
try:
    from lerobot.rl import actor, buffer, learner, eval_policy
    print("Found RL components in lerobot.rl")
    print(f"Actor module contents: {dir(actor)[:10]}...")
    print(f"Buffer module contents: {dir(buffer)[:10]}...")
    print(f"Learner module contents: {dir(learner)[:10]}...")
    print(f"Eval Policy module contents: {dir(eval_policy)[:10]}...")
except Exception as e:
    print(f"Error exploring lerobot.rl: {e}")

# Check for PPO specifically
print("\n=== Checking for PPO implementation ===")
try:
    from lerobot.policies import PPO
    print("Found PPO in lerobot.policies")
except ImportError:
    print("No PPO in lerobot.policies")

try:
    from lerobot.rl.ppo import PPO
    print("Found PPO in lerobot.rl.ppo")
except ImportError:
    print("No PPO in lerobot.rl.ppo")

try:
    from lerobot.policies.ppo import configuration_ppo, modeling_ppo, processor_ppo
    print("Found PPO implementation in lerobot.policies.ppo")
    print(f"PPO Configuration: {dir(configuration_ppo)[:10]}...")
    print(f"PPO Modeling: {dir(modeling_ppo)[:10]}...")
except ImportError as e:
    print(f"No PPO directory: {e}")

# Check available policies
print("\n=== Available Policies ===")
print(f"Available policies: {lerobot.available_policies}")

# Explore learner interface
print("\n=== Exploring Learner interface ===")
try:
    from lerobot.rl.learner import Learner
    print("Found Learner class")
    print(f"Learner methods: {[m for m in dir(Learner) if not m.startswith('_')][:15]}...")
except Exception as e:
    print(f"Error exploring Learner: {e}")

# Check training script
print("\n=== Checking training script structure ===")
try:
    import inspect
    from lerobot.scripts.lerobot_train import main
    print("Found lerobot_train.py main function")
    print(f"Train script signature: {inspect.signature(main)}")
except Exception as e:
    print(f"Error exploring train script: {e}")
