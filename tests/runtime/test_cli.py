"""Tests for cloud_robotics_sim.__main__ CLI."""

from pathlib import Path
from unittest.mock import MagicMock, patch

from cloud_robotics_sim.__main__ import main


class TestTrainCommand:
    """Tests for train subcommand."""

    def test_train_missing_config(self, tmp_path):
        missing = tmp_path / "missing.yaml"
        code = main(["train", "--config", str(missing)])
        assert code == 1

    def test_train_existing_config(self, tmp_path):
        config = tmp_path / "config.yaml"
        config.write_text("scene: empty_room")
        code = main(["train", "--config", str(config)])
        assert code == 0


class TestEvalCommand:
    """Tests for eval subcommand."""

    def test_eval_missing_checkpoint(self, tmp_path):
        missing = tmp_path / "missing.pt"
        code = main(["eval", "--checkpoint", str(missing)])
        assert code == 1

    def test_eval_existing_checkpoint(self, tmp_path):
        checkpoint = tmp_path / "model.pt"
        checkpoint.write_text("dummy")
        code = main(["eval", "--checkpoint", str(checkpoint)])
        assert code == 0


class TestAgentCommand:
    """Tests for agent subcommand."""

    def test_agent_no_args(self):
        code = main(["agent"])
        assert code == 1

    def test_agent_list_skills(self, capsys):
        code = main(["agent", "--list-skills"])
        captured = capsys.readouterr()
        assert code == 0
        assert "run_patent" in captured.out

    def test_agent_run_skill(self, capsys):
        code = main(["agent", "--skill", "list_patents"])
        captured = capsys.readouterr()
        assert code == 0
        assert '"status": "ok"' in captured.out

    def test_agent_run_skill_bad_param(self):
        code = main(["agent", "--skill", "list_patents", "--param", "hack=1"])
        assert code == 1

    def test_agent_goal_no_match(self):
        code = main(["agent", "--goal", "zzzqqq nothing matches"])
        assert code == 1

    def test_agent_goal_resolves_and_runs(self, capsys, monkeypatch):
        import cloud_robotics_sim.patents as patents

        class FakeState:
            time = 2.0
            parameters = {}
            metrics = {}

        monkeypatch.setattr(
            patents, "run_patent_simulation", lambda *a, **k: FakeState()
        )
        code = main(["agent", "--goal", "run patent US821393 headlessly"])
        captured = capsys.readouterr()
        assert code == 0
        assert '"skill": "run_patent"' in captured.out


class TestTestCommand:
    """Tests for test subcommand."""

    def test_test_command(self):
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = "pytest output"
        mock_result.stderr = ""
        with patch("subprocess.run", return_value=mock_result) as mock_run:
            code = main(["test"])
        assert code == 0
        mock_run.assert_called_once()
        args = mock_run.call_args[0][0]
        assert args[1:4] == ["-m", "pytest", str(Path(__file__).parent.parent)]
        assert "pytest" in args and "-m" in args

    def test_test_command_failure(self):
        mock_result = MagicMock()
        mock_result.returncode = 3
        mock_result.stdout = ""
        mock_result.stderr = "errors"
        with patch("subprocess.run", return_value=mock_result):
            code = main(["test"])
        assert code == 3


class TestNoSubcommand:
    """Tests for CLI with no subcommand."""

    def test_no_subcommand(self, capsys):
        code = main([])
        captured = capsys.readouterr()
        assert code == 1
        assert "usage:" in captured.out
