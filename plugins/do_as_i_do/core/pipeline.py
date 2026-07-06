"""End-to-end pipeline orchestrator for the do-as-i-do reproduction."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

from .data import DemoSequence, RobotTrajectory
from .deployment_stub import DeploymentStub, build_deployment_stage
from .env import DoAsIDoEnv
from .reconstruction_stub import ReconstructionStage, build_reconstruction_stage
from .retargeting import Retargeter, build_retargeter


class DoAsIDoPipeline:
    """High-level pipeline that wires together all do-as-i-do stages.

    This class combines reconstruction, retargeting, Genesis simulation replay,
    and deployment export into a single callable pipeline.
    """

    def __init__(self, config: Optional[dict[str, Any]] = None) -> None:
        self.config = config or self.default_config()
        self.reconstruction: ReconstructionStage = build_reconstruction_stage(self.config)
        self.env: Optional[DoAsIDoEnv] = None
        self.retargeter: Optional[Retargeter] = None
        self.deployment: DeploymentStub = build_deployment_stage(self.config)
        self.demo: Optional[DemoSequence] = None
        self.robot_traj: Optional[RobotTrajectory] = None

    def _require_robot_traj(self, robot_traj: Optional[RobotTrajectory] = None) -> RobotTrajectory:
        if robot_traj is not None:
            self.robot_traj = robot_traj
        if self.robot_traj is None:
            raise RuntimeError("No robot trajectory. Run retarget() first or pass a RobotTrajectory.")
        return self.robot_traj

    @staticmethod
    def default_config() -> dict[str, Any]:
        return {
            "headless": True,
            "robot": {"urdf_path": None, "hand_type": "allegro"},
            "scene": {
                "table_height": 0.75,
                "table_size": [0.8, 1.2, 0.02],
                "object_size": 0.05,
                "object_mesh_path": None,
            },
            "simulation": {
                "dt": 0.01,
                "substeps": 10,
                "backend": "cpu",
            },
            "env": {"max_steps": 500},
            "reconstruction": {"backend": "synthetic", "num_frames": 150, "fps": 30.0},
            "retargeting": {"method": "ik", "hand_open_degrees": 10.0},
            "deployment": {"frequency_hz": 50.0},
        }

    def reconstruct(self, video_path: str | Path, **kwargs: Any) -> DemoSequence:
        """Run the reconstruction stage."""
        self.demo = self.reconstruction.run(video_path, **kwargs)
        return self.demo

    def retarget(self, demo: Optional[DemoSequence] = None) -> RobotTrajectory:
        """Run the retargeting stage using the simulation robot."""
        if demo is not None:
            self.demo = demo
        if self.demo is None:
            raise RuntimeError("No demo available. Run reconstruct() first or pass a DemoSequence.")

        self.env = DoAsIDoEnv(config=self.config, headless=self.config.get("headless", True))
        self.retargeter = build_retargeter(self.config, self.env.robot)
        self.robot_traj = self.retargeter.retarget(self.demo)
        return self.robot_traj

    def simulate(
        self,
        robot_traj: Optional[RobotTrajectory] = None,
        close_on_finish: bool = False,
    ) -> list[dict[str, Any]]:
        """Replay the retargeted trajectory in Genesis."""
        self._require_robot_traj(robot_traj)
        if self.env is None:
            self.env = DoAsIDoEnv(config=self.config, headless=self.config.get("headless", True))

        obs_list = self.env.replay(
            self.robot_traj,
            object_trajectory=self.demo.object_trajectory if self.demo else None,
            close_on_finish=False,
        )
        metrics = {
            "num_steps": len(obs_list),
            "final_object_pos": obs_list[-1]["proprioception"][self.env.n_dofs * 2 : self.env.n_dofs * 2 + 3].tolist()
            if obs_list
            else None,
        }
        if close_on_finish:
            self.close()
        return metrics

    def deploy(
        self,
        robot_traj: Optional[RobotTrajectory] = None,
        output_dir: str | Path = "outputs/do_as_i_do",
    ) -> dict[str, Path]:
        """Export the trajectory to deployment formats."""
        self._require_robot_traj(robot_traj)

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        json_path = self.deployment.export_trajectory(self.robot_traj, output_dir / "robot_trajectory.json")
        urscript_path = self.deployment.export_to_urscript(self.robot_traj, output_dir / "demo.script")
        return {"trajectory_json": json_path, "urscript": urscript_path}

    def run(
        self,
        video_path: str | Path = "synthetic_demo",
        output_dir: str | Path = "outputs/do_as_i_do",
        **recon_kwargs: Any,
    ) -> dict[str, Any]:
        """Run the full pipeline end-to-end."""
        demo = self.reconstruct(video_path, **recon_kwargs)
        traj = self.retarget(demo)
        metrics = self.simulate(traj)

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        demo.save(output_dir / "demo_sequence")
        traj.save(output_dir / "robot_trajectory")

        paths = self.deploy(traj, output_dir=output_dir)
        self.close()
        return {
            "demo": demo,
            "robot_trajectory": traj,
            "simulation_metrics": metrics,
            "deployment_paths": paths,
        }

    def close(self) -> None:
        if self.env is not None:
            self.env.close()
            self.env = None
