
from do_as_i_do.core.pipeline import DoAsIDoPipeline


def test_pipeline_runs(tmp_path):
    """Run the full pipeline with synthetic data."""
    config = DoAsIDoPipeline.default_config()
    config["headless"] = True
    config["reconstruction"]["num_frames"] = 10

    pipeline = DoAsIDoPipeline(config)
    result = pipeline.run(video_path="test_demo", output_dir=tmp_path / "out")

    assert "robot_trajectory" in result
    assert len(result["robot_trajectory"]) == 10
    assert (tmp_path / "out" / "robot_trajectory.json").exists()
    assert (tmp_path / "out" / "demo_sequence.json").exists()
