from copy import deepcopy
from pathlib import Path
import tempfile

import numpy as np
import h5py
import yaml
import zarr
from hydra.utils import get_class

from scripts_maniskill.run_sim_workflow import (
    ROOT_DIR,
    WorkflowConfigError,
    build_collection_command,
    build_evaluation_commands,
    build_training_command,
    load_workflow_config,
    workflow_paths,
)
from scripts_maniskill.validate_umi_dataset import (
    DatasetValidationError,
    validate_dataset,
)
from convert_hdf5_to_umi_zarr import convert_dataset


CONFIG_PATH = ROOT_DIR / "configs" / "workflows" / "simulation.yaml"


def test_checked_in_workflow_resolves_pinned_fork_and_artifacts():
    config = load_workflow_config(CONFIG_PATH)
    paths = workflow_paths(config)

    assert paths["maniskill_root"] == ROOT_DIR / "third_party" / "src" / "maniskill"
    assert paths["dataset"].name.endswith(".zarr.zip")
    assert paths["checkpoint"] == paths["train_output"] / "checkpoints" / "latest.ckpt"
    assert get_class("embodisteer.training.NoOpImageRunner").__name__ == "NoOpImageRunner"


def test_workflow_commands_connect_collection_training_and_evaluation():
    config = load_workflow_config(CONFIG_PATH)
    paths = workflow_paths(config)

    collect, collect_cwd, collect_env = build_collection_command(config)
    assert collect_cwd == paths["maniskill_root"]
    assert collect[collect.index("--traj-num") + 1] == "20"
    assert collect_env["CUDA_VISIBLE_DEVICES"] == "0"

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


def test_workflow_rejects_collection_count_not_divisible_by_five():
    document = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    document = deepcopy(document)
    document["collection"]["total_trajectories"] = 6
    with tempfile.TemporaryDirectory() as temp_dir:
        path = Path(temp_dir) / "invalid.yaml"
        path.write_text(yaml.safe_dump(document), encoding="utf-8")
        try:
            load_workflow_config(path)
        except WorkflowConfigError as exc:
            assert "multiple of five" in str(exc)
        else:
            raise AssertionError("invalid collection count was accepted")


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
