import numpy as np
from do_as_i_do.core.data import DemoSequence, RobotTrajectory
from do_as_i_do.core.reconstruction_stub import SyntheticReconstructionStage


def test_demo_sequence_roundtrip(tmp_path):
    """Save and reload a DemoSequence."""
    stage = SyntheticReconstructionStage(num_frames=10)
    demo = stage.run("test_video")
    demo.save(tmp_path / "demo")
    loaded = DemoSequence.load(tmp_path / "demo")

    assert len(loaded) == len(demo)
    np.testing.assert_allclose(loaded.object_trajectory.positions, demo.object_trajectory.positions)
    np.testing.assert_allclose(loaded.left_hand.joints, demo.left_hand.joints)


def test_robot_trajectory_roundtrip(tmp_path):
    """Save and reload a RobotTrajectory."""
    traj = RobotTrajectory(
        left_arm_q=np.zeros((5, 6), dtype=np.float32),
        right_arm_q=np.zeros((5, 6), dtype=np.float32),
        left_hand_q=np.ones((5, 16), dtype=np.float32),
        right_hand_q=np.ones((5, 16), dtype=np.float32),
    )
    traj.save(tmp_path / "traj")
    loaded = RobotTrajectory.load(tmp_path / "traj")
    np.testing.assert_allclose(loaded.full_q(), traj.full_q())
