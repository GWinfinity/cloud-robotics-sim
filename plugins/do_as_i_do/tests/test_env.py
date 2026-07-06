import numpy as np
from do_as_i_do.core.env import DoAsIDoEnv


def test_env_create_reset_step():
    """Check environment creation, reset, and a single step."""
    env = DoAsIDoEnv(headless=True)
    assert env.n_dofs == 44
    assert env.action_dim == 44

    obs = env.reset(seed=0)
    assert obs["proprioception"].shape == (env.obs_dim,)

    action = np.zeros(env.action_dim, dtype=np.float32)
    obs, reward, terminated, truncated, info = env.step(action)
    assert obs["proprioception"].shape == (env.obs_dim,)
    assert isinstance(reward, float)
    assert isinstance(terminated, bool)
    env.close()


def test_env_random_actions():
    """Run a few random actions in the environment."""
    env = DoAsIDoEnv(headless=True)
    env.reset()
    for _ in range(5):
        action = np.random.uniform(-0.1, 0.1, size=env.action_dim).astype(np.float32)
        env.step(action)
    env.close()
