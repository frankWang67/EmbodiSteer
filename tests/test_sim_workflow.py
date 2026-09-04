from copy import deepcopy
import json
from pathlib import Path
import subprocess
import tempfile

import numpy as np
import h5py
import pytest
import yaml
import zarr
from hydra.utils import get_class

from run_sim_workflow import (
    ROOT_DIR,
    WorkflowConfigError,
    apply_evaluation_overrides,
    build_collection_command,
    build_evaluation_commands,
    build_evaluation_jobs,
    build_training_command,
    load_workflow_config,
    run_evaluation_stage,
    workflow_paths,
)
from scripts_maniskill.validate_umi_dataset import (
    DatasetValidationError,
    validate_dataset,
)
from convert_hdf5_to_umi_zarr import convert_dataset


CONFIG_PATH = ROOT_DIR / "configs" / "workflows" / "simulation.yaml"
COFFEE_CONFIG_PATH = (
    ROOT_DIR / "configs" / "workflows" / "make_iced_coffee" / "full_pipeline.yaml"
)
ALL_METHODS_CONFIG_PATH = (
    ROOT_DIR
    / "configs"
    / "workflows"
    / "make_iced_coffee"
    / "evaluation_all_methods.yaml"
)
PLACE_TOAST_CONFIG_PATH = (
    ROOT_DIR / "configs" / "workflows" / "place_toast" / "full_pipeline.yaml"
)
PLACE_TOAST_METHODS_CONFIG_PATH = (
    ROOT_DIR
    / "configs"
    / "workflows"
    / "place_toast"
    / "evaluation_all_methods.yaml"
)
TURN_FAUCET_CONFIG_PATH = (
    ROOT_DIR / "configs" / "workflows" / "turn_faucet" / "full_pipeline.yaml"
)
TURN_FAUCET_METHODS_CONFIG_PATH = (
    ROOT_DIR
    / "configs"
    / "workflows"
    / "turn_faucet"
    / "evaluation_all_methods.yaml"
)
FLOATING_ROBOT_UID = "floating_robotiq_2f_85_gripper_wristcam"
TOAST_COLLECTION_ROBOTS = [
    "panda_robotiq_wristcam",
    "xarm6_robotiq_wristcam",
    "xarm7_robotiq_wristcam",
    "ur5_robotiq_wristcam",
    FLOATING_ROBOT_UID,
]


def test_checked_in_workflow_resolves_pinned_fork_and_artifacts():
    config = load_workflow_config(CONFIG_PATH)
    paths = workflow_paths(config)

    assert paths["maniskill_root"] == ROOT_DIR / "third_party" / "src" / "maniskill"
    assert paths["dataset"].name.endswith(".zarr.zip")
    assert paths["checkpoint"] == paths["train_output"] / "checkpoints" / "latest.ckpt"
    assert paths["evaluation_output"] == ROOT_DIR / "data/outputs/evaluation/pickplace_local"
    assert get_class("embodisteer.training.NoOpImageRunner").__name__ == "NoOpImageRunner"


def test_workflow_commands_connect_collection_training_and_evaluation(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "3,5")
    config = load_workflow_config(CONFIG_PATH)
    paths = workflow_paths(config)

    collect, collect_cwd, collect_env = build_collection_command(config)
    assert collect_cwd == paths["maniskill_root"]
    assert collect[collect.index("--traj-num") + 1] == "100"
    robot_arg = collect.index("--robot-uids")
    assert collect[robot_arg + 1] == FLOATING_ROBOT_UID
    assert collect[robot_arg + 2] == "--obs-mode"
    assert collect_env["CUDA_VISIBLE_DEVICES"] == "3,5"

    train = build_training_command(config)
    assert f"task.dataset_path={paths['dataset']}" in train
    assert f"hydra.run.dir={paths['train_output']}" in train
    assert "training.resume=false" in train

    evaluations = build_evaluation_commands(config)
    assert len(evaluations) == 2
    policy_configs = {
        Path(command[command.index("--policy-config") + 1]).name
        for command in evaluations
    }
    assert policy_configs == {"ee.yaml", "embodisteer.yaml"}
    for evaluation in evaluations:
        assert evaluation[evaluation.index("--input") + 1] == str(paths["checkpoint"])
        assert evaluation[evaluation.index("--ckpt_filename") + 1] == "latest"
        assert evaluation[evaluation.index("--env_id") + 1] == config["task"]["env_id"]
        assert "--obstacle" in evaluation
        assert "--output-dir" in evaluation
        assert "--profile-name" in evaluation


def test_evaluation_jobs_isolate_profile_and_robot_results():
    config = load_workflow_config(CONFIG_PATH, stage="eval")
    jobs = build_evaluation_jobs(config)

    assert {(job["profile"], job["robot"]) for job in jobs} == {
        ("ee", "panda_robotiq_wristcam"),
        ("embodisteer", "panda_robotiq_wristcam"),
    }
    assert len({job["result_dir"] for job in jobs}) == 2
    assert jobs[0]["result_dir"].endswith(
        f"profiles/{jobs[0]['profile']}/panda_robotiq_wristcam"
    )


def test_eval_only_config_accepts_explicit_checkpoint_and_output(tmp_path):
    document = {
        "schema_version": 1,
        "task": {"env_id": "Dummy-v0"},
        "evaluation": {
            "checkpoint": str(tmp_path / "model.ckpt"),
            "output_dir": str(tmp_path / "results"),
            "run_id": "baseline_comparison",
            "profiles": {"jm2d": "configs/policy/jm2d.yaml"},
            "robots": ["panda_robotiq_wristcam"],
            "sim_backend": "physx_cpu",
            "num_env": 1,
            "num_eval_episodes": 1,
            "obs_mode": "rgb",
            "render_mode": "all",
            "steps_per_inference": 0,
            "max_episode_steps": 16,
            "obstacle": True,
        },
    }
    path = tmp_path / "eval.yaml"
    path.write_text(yaml.safe_dump(document), encoding="utf-8")

    config = load_workflow_config(path, stage="eval")
    paths = workflow_paths(config)

    assert paths["checkpoint"] == tmp_path / "model.ckpt"
    assert paths["evaluation_output"] == tmp_path / "results/baseline_comparison"
    assert config["evaluation"]["profiles"]["jm2d"]["policy_settings"]["baseline_method"] == "jm2d"


def test_profile_runtime_and_robot_overrides_are_applied():
    config = load_workflow_config(CONFIG_PATH, stage="eval")
    config["evaluation"]["profiles"]["ee"]["robots"] = ["ur5_robotiq_wristcam"]
    config["evaluation"]["profiles"]["ee"]["render_mode"] = "all"
    jobs = build_evaluation_jobs(config)
    ee_job = next(job for job in jobs if job["profile"] == "ee")

    assert ee_job["robot"] == "ur5_robotiq_wristcam"
    assert ee_job["command"][ee_job["command"].index("--render_mode") + 1] == "all"


def test_cli_robot_override_replaces_profile_robot_lists():
    config = load_workflow_config(CONFIG_PATH, stage="eval")
    config["evaluation"]["profiles"]["ee"]["robots"] = ["ur5_robotiq_wristcam"]

    changed = apply_evaluation_overrides(
        config, robots=["xarm6_robotiq_wristcam"], run_id="rerun"
    )

    assert changed == ("run_id", "robots")
    assert all(job["robot"] == "xarm6_robotiq_wristcam" for job in build_evaluation_jobs(config))


def _eval_only_config(tmp_path, robots=None):
    checkpoint = tmp_path / "model.ckpt"
    checkpoint.write_bytes(b"checkpoint")
    document = {
        "schema_version": 1,
        "task": {"env_id": "Dummy-v0"},
        "evaluation": {
            "checkpoint": str(checkpoint),
            "output_dir": str(tmp_path / "results"),
            "run_id": "test_run",
            "profiles": {"ee": "configs/policy/ee.yaml"},
            "robots": robots or ["panda_robotiq_wristcam"],
            "sim_backend": "physx_cpu",
            "num_env": 1,
            "num_eval_episodes": 1,
            "obs_mode": "rgb",
            "render_mode": "all",
            "steps_per_inference": 0,
            "max_episode_steps": 16,
            "obstacle": True,
            "continue_on_error": True,
        },
    }
    path = tmp_path / "eval.yaml"
    path.write_text(yaml.safe_dump(document), encoding="utf-8")
    return load_workflow_config(path, stage="eval")


def _fake_completed_result(command):
    output_dir = Path(command[command.index("--output-dir") + 1])
    profile = command[command.index("--profile-name") + 1]
    robot = command[command.index("--robot_uids") + 1]
    result_dir = output_dir / "profiles" / profile / robot
    result_dir.mkdir(parents=True, exist_ok=True)
    (result_dir / "metrics.json").write_text(
        json.dumps({"schema_version": 1, "metrics": {"success_once_rate": 1.0}}),
        encoding="utf-8",
    )
    np.savez_compressed(result_dir / "episode_metrics.npz", success_once=np.ones(1))


def test_eval_stage_writes_manifest_reports_and_resumes(tmp_path, monkeypatch):
    config = _eval_only_config(tmp_path)
    calls = []

    def fake_run(command, **kwargs):
        calls.append(command)
        _fake_completed_result(command)

    monkeypatch.setattr("run_sim_workflow.run_command", fake_run)
    run_evaluation_stage(config)
    output = workflow_paths(config)["evaluation_output"]

    assert len(calls) == 1
    assert (output / "run_manifest.yaml").is_file()
    assert (output / "results.json").is_file()
    manifest = yaml.safe_load((output / "run_manifest.yaml").read_text())
    assert manifest["status"] == "completed"
    assert manifest["checkpoint"] == {
        "path": str(workflow_paths(config)["checkpoint"]),
    }
    assert "evaluation_fingerprint" not in manifest

    calls.clear()
    run_evaluation_stage(config, resume=True)
    assert calls == []


def test_eval_stage_resume_rejects_changed_unhashed_run_definition(
    tmp_path, monkeypatch
):
    config = _eval_only_config(tmp_path)

    def fake_run(command, **kwargs):
        _fake_completed_result(command)

    monkeypatch.setattr("run_sim_workflow.run_command", fake_run)
    run_evaluation_stage(config)
    config["evaluation"]["render_mode"] = "rgb_array"

    with pytest.raises(WorkflowConfigError, match="evaluation settings changed"):
        run_evaluation_stage(config, resume=True)


def test_eval_stage_records_failure_and_continues(tmp_path, monkeypatch):
    config = _eval_only_config(
        tmp_path, ["panda_robotiq_wristcam", "ur5_robotiq_wristcam"]
    )
    calls = []

    def fake_run(command, **kwargs):
        robot = command[command.index("--robot_uids") + 1]
        calls.append(robot)
        if robot.startswith("panda"):
            raise subprocess.CalledProcessError(1, command)
        _fake_completed_result(command)

    monkeypatch.setattr("run_sim_workflow.run_command", fake_run)
    with pytest.raises(WorkflowConfigError, match="1 evaluation job"):
        run_evaluation_stage(config)

    output = workflow_paths(config)["evaluation_output"]
    manifest = yaml.safe_load((output / "run_manifest.yaml").read_text())
    assert calls == ["panda_robotiq_wristcam", "ur5_robotiq_wristcam"]
    assert manifest["status"] == "partial"


def test_workflow_rejects_collection_count_not_divisible_by_robot_count():
    document = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    document = deepcopy(document)
    document["collection"]["robot_uids"] = [
        FLOATING_ROBOT_UID,
        "panda_robotiq_wristcam",
    ]
    document["collection"]["total_trajectories"] = 5
    with tempfile.TemporaryDirectory() as temp_dir:
        path = Path(temp_dir) / "invalid.yaml"
        path.write_text(yaml.safe_dump(document), encoding="utf-8")
        try:
            load_workflow_config(path)
        except WorkflowConfigError as exc:
            assert "number of collection.robot_uids" in str(exc)
        else:
            raise AssertionError("invalid collection count was accepted")


def test_workflow_rejects_duplicate_collection_robots():
    document = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    document["collection"]["robot_uids"] = [FLOATING_ROBOT_UID] * 2
    with tempfile.TemporaryDirectory() as temp_dir:
        path = Path(temp_dir) / "invalid.yaml"
        path.write_text(yaml.safe_dump(document), encoding="utf-8")
        try:
            load_workflow_config(path)
        except WorkflowConfigError as exc:
            assert "must not contain duplicates" in str(exc)
        else:
            raise AssertionError("duplicate collection robots were accepted")


def test_coffee_workflow_collects_only_with_floating_robotiq():
    config = load_workflow_config(COFFEE_CONFIG_PATH)
    collect, _, _ = build_collection_command(config)

    assert config["collection"]["robot_uids"] == [FLOATING_ROBOT_UID]
    assert config["collection"]["total_trajectories"] == 200
    assert collect[collect.index("--traj-num") + 1] == "200"
    assert collect[collect.index("--robot-uids") + 1] == FLOATING_ROBOT_UID
    assert len(config["evaluation"]["robots"]) == 9


def test_coffee_all_methods_profiles_share_task_specific_safety_settings():
    config = load_workflow_config(ALL_METHODS_CONFIG_PATH, stage="eval")
    profiles = config["evaluation"]["profiles"]

    assert set(profiles) == {
        "ee",
        "joint_no_guidance",
        "joint_gd",
        "embodisteer",
        "post_hoc_cbf",
        "batch_sampling",
    }
    for name in {"joint_gd", "embodisteer", "post_hoc_cbf"}:
        assert profiles[name]["policy_settings"]["guidance_safety_margin"] == 0.07
    for name in {"embodisteer", "post_hoc_cbf"}:
        policy = profiles[name]["policy_settings"]
        assert policy["guidance_task_rot_weight"] == 1.0
    assert profiles["joint_gd"]["policy_settings"]["guidance_scale"] == 0.05
    assert profiles["joint_gd"]["policy_settings"]["guidance_loss_power"] == 2.0
    assert profiles["embodisteer"]["policy_settings"]["guidance_scale"] == 1.0
    assert profiles["post_hoc_cbf"]["policy_settings"]["guidance_scale"] == 1.0
    assert profiles["batch_sampling"]["policy_settings"]["batch_sampling_num"] == 16
    assert profiles["batch_sampling"]["policy_config"].endswith(
        "configs/policy/make_iced_coffee/batch_sampling.yaml"
    )


@pytest.mark.parametrize(
    ("path", "env_id", "collection_robots", "trajectories_per_robot"),
    (
        (
            PLACE_TOAST_CONFIG_PATH,
            "PickPlaceToasterToCounter-v1",
            TOAST_COLLECTION_ROBOTS,
            40,
        ),
        (TURN_FAUCET_CONFIG_PATH, "TurnOnSinkFaucet-v1", [FLOATING_ROBOT_UID], 200),
    ),
)
def test_additional_full_task_workflows(
    path, env_id, collection_robots, trajectories_per_robot
):
    config = load_workflow_config(path)
    collect, _, _ = build_collection_command(config)

    assert config["task"]["env_id"] == env_id
    assert config["collection"]["robot_uids"] == collection_robots
    assert config["collection"]["total_trajectories"] == 200
    assert collect[collect.index("--traj-num") + 1] == str(trajectories_per_robot)
    assert len(config["evaluation"]["robots"]) == 9


def test_place_toast_all_methods_use_reference_policy_settings():
    config = load_workflow_config(PLACE_TOAST_METHODS_CONFIG_PATH, stage="eval")
    profiles = config["evaluation"]["profiles"]

    assert set(profiles) == {
        "ee",
        "joint_no_guidance",
        "joint_gd",
        "embodisteer",
        "post_hoc_cbf",
        "batch_sampling",
    }
    for name in {"joint_gd", "embodisteer", "post_hoc_cbf"}:
        assert profiles[name]["policy_settings"]["guidance_safety_margin"] == 0.05
    for name in {"embodisteer", "post_hoc_cbf"}:
        policy = profiles[name]["policy_settings"]
        assert policy["guidance_task_rot_weight"] == 0.1
    assert profiles["joint_gd"]["policy_settings"]["guidance_scale"] == 1.0
    assert profiles["joint_gd"]["policy_settings"]["guidance_loss_power"] == 2.0
    assert profiles["embodisteer"]["policy_settings"]["guidance_scale"] == 1.5
    assert profiles["post_hoc_cbf"]["policy_settings"]["guidance_scale"] == 1.0
    assert profiles["batch_sampling"]["policy_settings"]["batch_sampling_num"] == 16
    for name in {"joint_gd", "embodisteer", "post_hoc_cbf", "batch_sampling"}:
        assert "/place_toast/" in profiles[name]["policy_config"]


def test_turn_faucet_all_methods_use_task_specific_safety_settings():
    config = load_workflow_config(TURN_FAUCET_METHODS_CONFIG_PATH, stage="eval")
    profiles = config["evaluation"]["profiles"]

    assert set(profiles) == {
        "ee",
        "joint_no_guidance",
        "joint_gd",
        "embodisteer",
        "post_hoc_cbf",
        "batch_sampling",
    }
    for name in {"joint_gd", "embodisteer", "post_hoc_cbf"}:
        assert profiles[name]["policy_settings"]["guidance_safety_margin"] == 0.07
    for name in {"embodisteer", "post_hoc_cbf"}:
        assert profiles[name]["policy_settings"]["guidance_task_rot_weight"] == 1.0
    assert profiles["joint_gd"]["policy_settings"]["guidance_scale"] == 0.025
    assert profiles["joint_gd"]["policy_settings"]["guidance_loss_power"] == 2.0
    assert profiles["embodisteer"]["policy_settings"]["guidance_scale"] == 1.0
    assert profiles["post_hoc_cbf"]["policy_settings"]["guidance_scale"] == 1.0
    assert profiles["batch_sampling"]["policy_settings"]["batch_sampling_num"] == 16
    assert profiles["batch_sampling"]["policy_config"].endswith(
        "configs/policy/turn_faucet/batch_sampling.yaml"
    )


def test_workflow_rejects_unknown_fields():
    document = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    document["training"]["num_epohcs"] = 120
    with tempfile.TemporaryDirectory() as temp_dir:
        path = Path(temp_dir) / "invalid.yaml"
        path.write_text(yaml.safe_dump(document), encoding="utf-8")
        try:
            load_workflow_config(path)
        except WorkflowConfigError as exc:
            assert "num_epohcs" in str(exc)
        else:
            raise AssertionError("unknown workflow field was accepted")


def test_workflow_rejects_gpu_selection_in_yaml():
    document = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    document["training"]["gpu_index"] = 0
    with tempfile.TemporaryDirectory() as temp_dir:
        path = Path(temp_dir) / "invalid.yaml"
        path.write_text(yaml.safe_dump(document), encoding="utf-8")
        with pytest.raises(WorkflowConfigError, match="gpu_index"):
            load_workflow_config(path)


def _write_dataset(path: Path, action_dim: int = 7) -> None:
    num_steps = 3
    with zarr.ZipStore(str(path), mode="w") as store:
        root = zarr.group(store=store)
        data = root.create_group("data")
        meta = root.create_group("meta")
        arrays = {
            "action": np.zeros((num_steps, action_dim), dtype=np.float32),
            "camera0_rgb": np.zeros((num_steps, 224, 224, 3), dtype=np.uint8),
            "robot0_demo_end_pose": np.zeros((num_steps, 6), dtype=np.float32),
            "robot0_demo_start_pose": np.zeros((num_steps, 6), dtype=np.float32),
            "robot0_eef_pos": np.zeros((num_steps, 3), dtype=np.float32),
            "robot0_eef_rot_axis_angle": np.zeros((num_steps, 3), dtype=np.float32),
            "robot0_gripper_width": np.zeros((num_steps, 1), dtype=np.float32),
        }
        for name, array in arrays.items():
            data.array(name, array, chunks=array.shape, compressor=None)
        meta.array("episode_ends", np.asarray([num_steps], dtype=np.int64), compressor=None)


def test_dataset_validator_accepts_converter_storage_schema():
    with tempfile.TemporaryDirectory() as temp_dir:
        path = Path(temp_dir) / "dataset.zarr.zip"
        _write_dataset(path)
        report = validate_dataset(path)

    assert report["num_episodes"] == 1
    assert report["num_steps"] == 3
    assert report["raw_action_dim"] == 7
    assert report["training_action_dim"] == 10


def test_dataset_validator_rejects_training_facing_action_mismatch():
    with tempfile.TemporaryDirectory() as temp_dir:
        path = Path(temp_dir) / "dataset.zarr.zip"
        _write_dataset(path, action_dim=10)
        try:
            validate_dataset(path)
        except DatasetValidationError as exc:
            assert "raw action must have 7" in str(exc)
        else:
            raise AssertionError("invalid raw action dimension was accepted")


def test_hdf5_converter_produces_validator_compatible_umi_dataset():
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_dir = Path(temp_dir)
        h5_path = temp_dir / "demos.h5"
        zarr_path = temp_dir / "dataset.zarr.zip"
        steps = 3
        with h5py.File(h5_path, "w") as handle:
            traj = handle.create_group("traj_0")
            obs = traj.create_group("obs")
            extra = obs.create_group("extra")
            agent = obs.create_group("agent")
            sensor_data = obs.create_group("sensor_data")
            camera = sensor_data.create_group("hand_camera")
            extra.create_dataset(
                "tcp_pose",
                data=np.tile(np.asarray([0.1, 0.2, 0.3, 1.0, 0.0, 0.0, 0.0], np.float32), (steps, 1)),
            )
            agent.create_dataset("qpos", data=np.zeros((steps, 8), np.float32))
            camera.create_dataset("rgb", data=np.zeros((steps, 8, 8, 3), np.uint8))
            actions = np.tile(
                np.asarray([0.1, 0.2, 0.3, 1.0, 0.0, 0.0, 0.0, 0.02], np.float32),
                (steps, 1),
            )
            traj.create_dataset("actions", data=actions)

        convert_dataset(h5_path, zarr_path, camera_name="hand_camera", image_size=224)
        report = validate_dataset(zarr_path)

    assert report["num_episodes"] == 1
    assert report["num_steps"] == steps
    assert report["raw_action_dim"] == 7
