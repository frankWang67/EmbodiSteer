import json
from pathlib import Path

import numpy as np

from embodisteer.evaluation import (
    aggregate_workflow_results,
    render_results_markdown,
    workflow_result_dir,
    write_workflow_results,
)


def _write_result(root: Path, profile: str, robot: str, success: list[float]) -> Path:
    result_dir = workflow_result_dir(root, profile, robot)
    result_dir.mkdir(parents=True)
    success_array = np.asarray(success, dtype=np.float32)
    collision = np.asarray([0, 2][: len(success)], dtype=np.float32)
    metrics = {
        "success_once_rate": float(success_array.mean()),
        "success_at_end_rate": float(success_array.mean()),
        "avg_collision_per_episode": float(collision.mean()),
        "std_collision_per_episode": float(collision.std()),
        "jm2d_ik_pose_success_rate": 0.75,
    }
    (result_dir / "metrics.json").write_text(
        json.dumps({"schema_version": 1, "metrics": metrics}), encoding="utf-8"
    )
    np.savez_compressed(
        result_dir / "episode_metrics.npz",
        success_once=success_array,
        success_at_end=success_array,
        collision_count=collision,
        max_reward=success_array,
        first_success_step=np.where(success_array > 0, 10, -1),
        fail_once=collision > 0,
    )
    return result_dir


def test_aggregation_preserves_custom_metrics_and_pools_episodes(tmp_path):
    first = _write_result(tmp_path, "jm2d_n16", "panda", [1, 0])
    second = _write_result(tmp_path, "jm2d_n16", "ur5", [1])
    jobs = [
        {"profile": "jm2d_n16", "robot": "panda", "result_dir": first, "status": "completed"},
        {"profile": "jm2d_n16", "robot": "ur5", "result_dir": second, "status": "completed"},
    ]

    results = aggregate_workflow_results(tmp_path, jobs)
    profile = results["profiles"]["jm2d_n16"]

    assert profile["macro_average"]["success_once_rate"] == 0.75
    assert np.isclose(
        profile["pooled_episode_metrics"]["success_once_rate"], 2 / 3
    )
    assert profile["pooled_episode_metrics"]["episode_count"] == 3
    assert profile["robots"]["panda"]["metrics"]["jm2d_ik_pose_success_rate"] == 0.75
    assert "std_collision_per_episode" not in profile["macro_average"]


def test_report_writer_creates_json_and_markdown_with_failure_status(tmp_path):
    completed = _write_result(tmp_path, "embodisteer", "panda", [1])
    jobs = [
        {"profile": "embodisteer", "robot": "panda", "result_dir": completed, "status": "completed"},
        {
            "profile": "embodisteer",
            "robot": "ur5",
            "result_dir": workflow_result_dir(tmp_path, "embodisteer", "ur5"),
            "status": "failed",
            "error": "test failure",
        },
    ]

    results = write_workflow_results(tmp_path, jobs)
    markdown = (tmp_path / "results.md").read_text(encoding="utf-8")

    assert (tmp_path / "results.json").is_file()
    assert results["profiles"]["embodisteer"]["robots"]["ur5"]["status"] == "failed"
    assert "Macro average" in markdown
    assert "All episodes (pooled)" in markdown
    assert "failed" in markdown


def test_markdown_keeps_method_specific_metric_table():
    text = render_results_markdown(
        {
            "profiles": {
                "jm2d": {
                    "robots": {
                        "panda": {
                            "status": "completed",
                            "metrics": {"jm2d_ik_trajectory_success_rate": 0.5},
                        }
                    },
                    "macro_average": {},
                    "pooled_episode_metrics": {},
                }
            }
        }
    )

    assert "jm2d_ik_trajectory_success_rate" in text
