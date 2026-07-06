from do_as_i_do.core.env import DoAsIDoEnv
from do_as_i_do.core.reconstruction_stub import SyntheticReconstructionStage


def test_retarget_shapes():
    """Check that retargeting produces trajectories of the expected shape."""
    stage = SyntheticReconstructionStage(num_frames=10)
    demo = stage.run("test")

    env = DoAsIDoEnv(headless=True)
    traj = env.retarget(demo)

    assert len(traj) == len(demo)
    assert traj.left_arm_q.shape == (10, 6)
    assert traj.right_arm_q.shape == (10, 6)
    assert traj.left_hand_q.shape == (10, 16)
    assert traj.right_hand_q.shape == (10, 16)
    env.close()
