from pathlib import Path

from click.testing import CliRunner

from embodisteer.evaluation import evaluation_subdir
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


def test_result_subdirs_distinguish_guidance_methods_and_reverse_cbf():
    common = {
        "inference_space": "joint",
        "baseline_method": "",
        "obstacle": True,
        "obstacle_observation_noise": None,
    }

    cbf = evaluation_subdir(
        guidance="cbf", reverse_cbf_task_threshold=None, **common
    )
    gd = evaluation_subdir(
        guidance="gd", reverse_cbf_task_threshold=None, **common
    )
    reverse_cbf = evaluation_subdir(
        guidance="cbf", reverse_cbf_task_threshold=0.05, **common
    )

    assert cbf == "obstacle_joint_space_guidance_cbf"
    assert gd == "obstacle_joint_space_guidance_gd"
    assert reverse_cbf == (
        "obstacle_joint_space_guidance_cbf_reverse_cbf_task_threshold_0.05"
    )
    assert len({cbf, gd, reverse_cbf}) == 3


def test_result_subdir_preserves_noise_provenance():
    subdir = evaluation_subdir(
        inference_space="ee",
        guidance="gd",
        baseline_method="",
        reverse_cbf_task_threshold=None,
        obstacle=True,
        obstacle_observation_noise=(0.01, 0.02, 0.03),
    )

    assert subdir == (
        "obstacle_ee_space_guidance_gd/"
        "obs_noise_pos0.01_size0.02_rot0.03"
    )
