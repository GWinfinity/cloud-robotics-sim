"""Tests for the improvement_loop_main compatibility shim."""

from __future__ import annotations


class TestImprovementLoopMainShim:
    """Verify the shim re-exports the public API from runtime.main."""

    def test_import_improvement_loop(self):
        from cloud_robotics_sim.runtime.improvement_loop_main import ImprovementLoop
        from cloud_robotics_sim.runtime.main import ImprovementLoop as MainLoop

        assert ImprovementLoop is MainLoop

    def test_import_loop_config(self):
        from cloud_robotics_sim.runtime.improvement_loop_main import LoopConfig
        from cloud_robotics_sim.runtime.main import LoopConfig as MainConfig

        assert LoopConfig is MainConfig

    def test_import_make_env_from_config(self):
        from cloud_robotics_sim.runtime.improvement_loop_main import (
            make_env_from_config,
        )
        from cloud_robotics_sim.runtime.main import (
            make_env_from_config as main_make_env,
        )

        assert make_env_from_config is main_make_env

    def test_all_exports(self):
        import cloud_robotics_sim.runtime.improvement_loop_main as mod

        assert set(mod.__all__) == {
            "ImprovementLoop",
            "LoopConfig",
            "make_env_from_config",
        }
