import numpy as np
from do_as_i_do.core.env import DoAsIDoEnv
from do_as_i_do.core.reconstruction_stub import SyntheticReconstructionStage


def test_env_create_sharpa():
    """Check that the Sharpa Wave robot loads and runs in Genesis."""
    config = {"robot": {"hand_type": "sharpa"}}
    env = DoAsIDoEnv(config=config, headless=True)
    assert env.n_dofs == 56
    assert env.action_dim == 56

    obs = env.reset(seed=0)
    assert obs["proprioception"].shape == (env.obs_dim,)

    action = np.zeros(env.action_dim, dtype=np.float32)
    obs, reward, terminated, truncated, info = env.step(action)
    assert obs["proprioception"].shape == (env.obs_dim,)
    assert isinstance(reward, float)
    assert isinstance(terminated, bool)
    env.close()


def test_retarget_sharpa():
    """Check retargeting against the Sharpa hand DOF count."""
    stage = SyntheticReconstructionStage(num_frames=10, hand_dof=22)
    demo = stage.run("test")

    config = {"robot": {"hand_type": "sharpa"}}
    env = DoAsIDoEnv(config=config, headless=True)
    traj = env.retarget(demo)

    assert len(traj) == len(demo)
    assert traj.left_arm_q.shape == (10, 6)
    assert traj.right_arm_q.shape == (10, 6)
    assert traj.left_hand_q.shape == (10, 22)
    assert traj.right_hand_q.shape == (10, 22)
    env.close()
