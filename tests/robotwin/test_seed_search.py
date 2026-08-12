"""Tests for batched parallel seed search (doc section 5.1)."""

from __future__ import annotations

import numpy as np
import pytest

from cloud_robotics_sim.robotwin.seed_search import batched_seed_search


class TestBatchedSeedSearch:
    """Pure logic with scripted callbacks (no Genesis required)."""

    def test_batches_and_collects_mask(self) -> None:
        applied: list[tuple[int, int]] = []
        rollouts = 0
        current_batch: list[int] = []

        def apply_seed(seed: int, env_idx: int) -> None:
            applied.append((seed, env_idx))
            current_batch.append(seed)

        def rollout() -> None:
            nonlocal rollouts
            rollouts += 1

        def evaluate() -> np.ndarray:
            mask = np.array([s % 2 == 0 for s in current_batch])
            current_batch.clear()
            return mask

        result = batched_seed_search(
            seeds=list(range(10)),
            n_envs=4,
            apply_seed=apply_seed,
            rollout=rollout,
            evaluate=evaluate,
        )

        assert result.n_batches == 3  # ceil(10 / 4)
        assert rollouts == 3
        assert applied[:4] == [(0, 0), (1, 1), (2, 2), (3, 3)]
        assert result.accepted == [0, 2, 4, 6, 8]
        assert result.acceptance_rate == pytest.approx(0.5)
        assert [o.env_idx for o in result.outcomes[:4]] == [0, 1, 2, 3]

    def test_early_stop_at_target(self) -> None:
        def evaluate() -> np.ndarray:
            return np.array([True, True, False, False])

        result = batched_seed_search(
            seeds=list(range(12)),
            n_envs=4,
            apply_seed=lambda s, i: None,
            rollout=lambda: None,
            evaluate=evaluate,
            target_accepted=3,
        )

        assert result.n_batches == 2  # stops after accepting 4 >= 3
        assert result.accepted == [0, 1, 4, 5]

    def test_evaluate_length_mismatch_rejected(self) -> None:
        with pytest.raises(ValueError, match="evaluate"):
            batched_seed_search(
                seeds=[1, 2, 3],
                n_envs=2,
                apply_seed=lambda s, i: None,
                rollout=lambda: None,
                evaluate=lambda: np.array([True]),
            )

    def test_invalid_n_envs(self) -> None:
        with pytest.raises(ValueError, match="n_envs"):
            batched_seed_search(
                seeds=[1],
                n_envs=0,
                apply_seed=lambda s, i: None,
                rollout=lambda: None,
                evaluate=lambda: np.array([True]),
            )

    def test_empty_result_properties(self) -> None:
        result = batched_seed_search(
            seeds=[],
            n_envs=4,
            apply_seed=lambda s, i: None,
            rollout=lambda: None,
            evaluate=lambda: np.array([]),
        )
        assert result.accepted == []
        assert result.acceptance_rate == 0.0
        assert result.n_batches == 0


@pytest.mark.slow
class TestRealGenesisSeedSearch:
    """Smoke test: seed search on a real batched Genesis scene."""

    def test_seed_search_on_batched_scene(self, tmp_path) -> None:
        pytest.importorskip("genesis")
        from cloud_robotics_sim.backends.genesis_backend import GenesisBackend
        from examples.robotwin.aloha_demo import _write_synthetic_arm

        n_envs = 3
        urdf = _write_synthetic_arm(tmp_path)
        backend = GenesisBackend()
        backend.initialize(headless=True, device="cpu")
        scene = backend.create_scene(dt=0.01, substeps=1, headless=True)
        robot = backend.load_urdf(str(urdf), pos=(0.0, 0.0, 0.0), fixed=True)
        scene.add_articulation(robot)
        scene.build(n_envs=n_envs, env_spacing=(2.0, 2.0))

        entity = robot._entity

        def apply_seed(seed: int, env_idx: int) -> None:
            rng = np.random.default_rng(seed)
            entity.set_qpos(
                rng.uniform(-0.5, 0.5, size=robot.n_dofs),
                envs_idx=[env_idx],
            )

        def rollout() -> None:
            scene.step()

        def evaluate() -> np.ndarray:
            # Stand-in success criterion: joint 0 within limits after a step.
            qpos = np.asarray(entity.get_qpos()).reshape(n_envs, -1)
            return np.abs(qpos[:, 0]) <= 1.0

        result = batched_seed_search(
            seeds=list(range(6)),
            n_envs=n_envs,
            apply_seed=apply_seed,
            rollout=rollout,
            evaluate=evaluate,
            target_accepted=2,
        )

        assert len(result.accepted) >= 2
        assert result.n_batches >= 1
