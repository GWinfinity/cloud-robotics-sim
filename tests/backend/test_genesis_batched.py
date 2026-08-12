"""Real-Genesis tests for batched envs and camera parameter alignment.

Covers migration doc section 3.1/5.1 (``n_envs`` batched build) and
section 7 (camera intrinsics must match the pinhole derivation within
1e-6, extrinsics are camera-to-world).
"""

from __future__ import annotations

import numpy as np
import pytest

from cloud_robotics_sim.robotwin.recorder import EpisodeRecorder
from cloud_robotics_sim.utils.camera import intrinsics_from_fov
from examples.robotwin.aloha_demo import _write_synthetic_arm

pytestmark = pytest.mark.slow


def _build_scene(tmp_path, n_envs: int):
    """Build a real Genesis CPU scene with the synthetic arm."""
    pytest.importorskip("genesis")
    from cloud_robotics_sim.backends.genesis_backend import GenesisBackend

    urdf = _write_synthetic_arm(tmp_path)
    backend = GenesisBackend()
    backend.initialize(headless=True, device="cpu")
    scene = backend.create_scene(dt=0.01, substeps=1, headless=True)
    robot = backend.load_urdf(str(urdf), pos=(0.0, 0.0, 0.0), fixed=True)
    scene.add_articulation(robot)
    camera = scene.renderer.add_camera(
        name="head",
        pos=(0.0, -0.8, 0.6),
        lookat=(0.0, 0.0, 0.2),
        resolution=(640, 480),
        fov=60.0,
    )
    if n_envs > 1:
        scene.build(n_envs=n_envs, env_spacing=(2.0, 2.0))
    else:
        scene.build()
    return scene, robot, camera


@pytest.fixture()
def built_scene(tmp_path):
    """A real Genesis CPU scene built for 2 parallel envs."""
    return _build_scene(tmp_path, n_envs=2)


@pytest.fixture()
def single_env_scene(tmp_path):
    """A real Genesis CPU scene built for a single env (no env offset)."""
    return _build_scene(tmp_path, n_envs=1)


class TestCameraParamAlignment:
    """Doc section 7: intrinsics/extrinsics alignment with 1e-6 tolerance."""

    def test_intrinsics_match_pinhole_derivation(self, single_env_scene) -> None:
        _, _, camera = single_env_scene
        intrinsic, _ = camera.get_camera_params()
        expected = intrinsics_from_fov(640, 480, 60.0)
        np.testing.assert_allclose(intrinsic, expected, rtol=0, atol=1e-6)

    def test_extrinsic_is_camera_to_world(self, single_env_scene) -> None:
        _, _, camera = single_env_scene
        _, extrinsic = camera.get_camera_params()
        assert extrinsic.shape == (4, 4)
        # Translation part = camera world position; rotation is orthonormal.
        np.testing.assert_allclose(extrinsic[:3, 3], [0.0, -0.8, 0.6], atol=1e-6)
        rotation = extrinsic[:3, :3]
        np.testing.assert_allclose(rotation @ rotation.T, np.eye(3), atol=1e-6)


class TestBatchedEnvs:
    """Doc section 3.1/5.1: n_envs batch semantics, env dim first."""

    def test_batched_qpos_shape(self, built_scene) -> None:
        _, robot, _ = built_scene
        qpos = robot.get_qpos()
        assert qpos.shape[-1] == robot.n_dofs
        assert qpos.reshape(2, -1).shape == (2, robot.n_dofs)

    def test_batched_capture_and_split(self, built_scene, tmp_path) -> None:
        scene, robot, _ = built_scene
        rec = EpisodeRecorder(task_name="batched", n_envs=2)
        for step in range(3):
            scene.step()
            rec.capture(step, qpos=np.asarray(robot.get_qpos()).reshape(2, -1))

        paths = rec.save_all_per_env(tmp_path / "episodes", prefix="ep")
        assert len(paths) == 2

        import h5py

        for env_idx, path in enumerate(paths):
            with h5py.File(path) as f:
                assert f.attrs["env_idx"] == env_idx
                assert f["qpos"].shape == (3, robot.n_dofs)

    def test_batched_domain_randomization(self, built_scene) -> None:
        _, robot, _ = built_scene
        n_links = len(robot._entity.links)
        ratios = np.full((2, n_links), 1.1)
        robot.set_friction_ratio(ratios, envs_idx=[0, 1])
        robot.set_mass_shift(np.zeros((2, n_links)))
        robot.set_com_shift(np.zeros((2, n_links, 3)))
