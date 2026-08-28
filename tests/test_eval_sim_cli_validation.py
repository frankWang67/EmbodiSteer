from pathlib import Path

from click.testing import CliRunner

from eval_sim_single_robot import main


ROOT = Path(__file__).resolve().parents[1]
MISSING_CHECKPOINT_ROOT = "/definitely/missing/embodisteer-checkpoint"


def _base_args():
    return [
        "--input",
        MISSING_CHECKPOINT_ROOT,
        "--ckpt_filename",
        "missing",
        "--env_id",
        "Dummy-v0",
        "--robot_uids",
        "panda_robotiq_wristcam",
        "--obstacle",
        "--policy-config",
        str(ROOT / "configs/policy/embodisteer.yaml"),
    ]


def test_invalid_control_mode_fails_before_checkpoint_access():
    result = CliRunner().invoke(
        main,
        _base_args() + ["--control_mode", "pd_ee_pose"],
    )

    assert result.exit_code == 2
    assert "require a pd_joint* control mode" in result.output
    assert "Checkpoint" not in result.output


def test_negative_obstacle_noise_fails_before_checkpoint_access():
    result = CliRunner().invoke(
        main,
        _base_args()
        + ["--obstacle-observation-noise", "-0.1", "0.1", "0.1"],
    )

    assert result.exit_code == 2
    assert "must be non-negative" in result.output
    assert "Checkpoint" not in result.output
