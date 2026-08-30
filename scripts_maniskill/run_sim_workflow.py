#!/usr/bin/env python3
"""Run the reproducible simulation data/train/evaluation workflow.

The workflow is deliberately a thin, config-driven orchestrator.  It invokes
the pinned ManiSkill collection utility, the repository's HDF5 converter, the
existing Hydra training entry point, and the public simulation evaluator in
sequence.  No data or checkpoint is copied into the source release tree.

Examples
--------
Preview every command without requiring data or starting a process::

    python scripts_maniskill/run_sim_workflow.py \
      --config configs/workflows/simulation.yaml --stage all --dry-run

Run only conversion, training, or evaluation after the preceding artifact is
available::

    python scripts_maniskill/run_sim_workflow.py --stage convert
    python scripts_maniskill/run_sim_workflow.py --stage train
    python scripts_maniskill/run_sim_workflow.py --stage eval
"""

from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any, Iterable, Mapping

import yaml


ROOT_DIR = Path(__file__).resolve().parents[1]


class WorkflowConfigError(ValueError):
    """Raised when the workflow YAML is incomplete or inconsistent."""


def _path(value: str | os.PathLike[str]) -> Path:
    candidate = Path(value).expanduser()
    if not candidate.is_absolute():
        candidate = ROOT_DIR / candidate
    return candidate.resolve()


def _mapping(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise WorkflowConfigError(f"{name} must be a mapping")
    return dict(value)


def _reject_unknown(mapping: Mapping[str, Any], allowed: set[str], name: str) -> None:
    unknown = sorted(set(mapping).difference(allowed))
    if unknown:
        raise WorkflowConfigError(f"{name} contains unknown fields: {', '.join(unknown)}")


def load_workflow_config(path: str | os.PathLike[str]) -> dict[str, Any]:
    """Load and validate a schema-versioned workflow configuration."""
    config_path = _path(path)
    try:
        with config_path.open("r", encoding="utf-8") as handle:
            document = yaml.safe_load(handle)
    except OSError as exc:
        raise WorkflowConfigError(f"Cannot read workflow config {config_path}: {exc}") from exc
    except yaml.YAMLError as exc:
        raise WorkflowConfigError(f"Invalid YAML in workflow config {config_path}: {exc}") from exc

    if not isinstance(document, Mapping):
        raise WorkflowConfigError("workflow config must be a YAML mapping")
    if document.get("schema_version") != 1:
        raise WorkflowConfigError("workflow config schema_version must be 1")
    _reject_unknown(
        document,
        {"schema_version", "task", "paths", "collection", "conversion", "training", "evaluation"},
        "workflow config",
    )

    task = _mapping(document.get("task"), "task")
    paths = _mapping(document.get("paths"), "paths")
    collection = _mapping(document.get("collection"), "collection")
    conversion = _mapping(document.get("conversion"), "conversion")
    training = _mapping(document.get("training"), "training")
    evaluation = _mapping(document.get("evaluation"), "evaluation")

    required = {
        "task": ("name", "env_id", "env_name"),
        "paths": ("maniskill_root", "data_root", "h5_filename", "dataset_filename", "train_output_dir"),
        "collection": ("total_trajectories", "obs_mode", "control_mode", "sim_backend_gen", "sim_backend_replay", "num_procs", "gpu_index"),
        "conversion": ("camera_name", "image_size"),
        "training": ("config_dir", "config_name", "device", "gpu_index", "logging_mode", "overrides"),
        "evaluation": ("profiles", "gpu_index", "robots", "sim_backend", "num_env", "num_eval_episodes", "env_seed", "obs_mode", "render_mode", "steps_per_inference", "max_episode_steps", "obstacle"),
    }
    allowed = {
        "task": set(required["task"]),
        "paths": set(required["paths"]),
        "collection": set(required["collection"]) | {"save_video"},
        "conversion": set(required["conversion"]),
        "training": set(required["training"]),
        "evaluation": set(required["evaluation"]) | {"obstacle_observation_noise"},
    }
    sections = {
        "task": task,
        "paths": paths,
        "collection": collection,
        "conversion": conversion,
        "training": training,
        "evaluation": evaluation,
    }
    for section, keys in required.items():
        missing = [key for key in keys if key not in sections[section]]
        if missing:
            raise WorkflowConfigError(f"{section} is missing fields: {', '.join(missing)}")
        _reject_unknown(sections[section], allowed[section], section)

    total = collection["total_trajectories"]
    if isinstance(total, bool) or not isinstance(total, int) or total < 5 or total % 5:
        raise WorkflowConfigError("collection.total_trajectories must be a positive multiple of five")
    for section, key in (
        (collection, "num_procs"),
        (evaluation, "num_env"),
        (evaluation, "num_eval_episodes"),
        (evaluation, "max_episode_steps"),
    ):
        value = section[key]
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise WorkflowConfigError(f"{key} must be a positive integer")
    if evaluation["num_eval_episodes"] % evaluation["num_env"]:
        raise WorkflowConfigError("evaluation.num_eval_episodes must be divisible by evaluation.num_env")
    steps_per_inference = evaluation["steps_per_inference"]
    if isinstance(steps_per_inference, bool) or not isinstance(steps_per_inference, int) or steps_per_inference < 0:
        raise WorkflowConfigError("evaluation.steps_per_inference must be a non-negative integer")
    for section_name, section in (
        ("collection", collection),
        ("training", training),
        ("evaluation", evaluation),
    ):
        gpu_index = section["gpu_index"]
        if isinstance(gpu_index, bool) or not isinstance(gpu_index, int) or gpu_index < 0:
            raise WorkflowConfigError(f"{section_name}.gpu_index must be a non-negative integer")
    if isinstance(conversion["image_size"], bool) or not isinstance(conversion["image_size"], int) or conversion["image_size"] < 1:
        raise WorkflowConfigError("conversion.image_size must be a positive integer")
    for section_name, section, key in (
        ("collection", collection, "save_video"),
        ("evaluation", evaluation, "obstacle"),
    ):
        if key in section and not isinstance(section[key], bool):
            raise WorkflowConfigError(f"{section_name}.{key} must be a YAML boolean")
    for filename_key in ("h5_filename", "dataset_filename"):
        filename = str(paths[filename_key])
        if Path(filename).name != filename:
            raise WorkflowConfigError(f"paths.{filename_key} must be a filename, not a path")
    robots = evaluation["robots"]
    if not isinstance(robots, list) or not robots or not all(isinstance(robot, str) and robot for robot in robots):
        raise WorkflowConfigError("evaluation.robots must be a non-empty list of robot UID strings")
    profiles = evaluation["profiles"]
    if not isinstance(profiles, Mapping) or not profiles:
        raise WorkflowConfigError("evaluation.profiles must be a non-empty name-to-YAML mapping")
    for name, profile_path in profiles.items():
        if not isinstance(name, str) or not name or not isinstance(profile_path, str) or not profile_path:
            raise WorkflowConfigError("evaluation.profiles must map non-empty names to YAML paths")
        if not _path(profile_path).is_file():
            raise WorkflowConfigError(f"evaluation profile does not exist: {_path(profile_path)}")
    overrides = training["overrides"]
    if not isinstance(overrides, Mapping):
        raise WorkflowConfigError("training.overrides must be a mapping of Hydra keys to values")

    result = {
        "config_path": str(config_path),
        "task": task,
        "paths": paths,
        "collection": collection,
        "conversion": conversion,
        "training": training,
        "evaluation": evaluation,
    }
    return result


def workflow_paths(config: Mapping[str, Any]) -> dict[str, Path]:
    """Resolve all artifact paths without creating directories."""
    task = config["task"]
    paths = config["paths"]
    data_root = _path(paths["data_root"])
    maniskill_root = _path(paths["maniskill_root"])
    env_id = str(task["env_id"])
    motionplanning = maniskill_root / "demos" / env_id / "motionplanning"
    return {
        "maniskill_root": maniskill_root,
        "h5": motionplanning / str(paths["h5_filename"]),
        "dataset": data_root / str(paths["dataset_filename"]),
        "train_output": _path(paths["train_output_dir"]),
        "checkpoint": _path(paths["train_output_dir"]) / "checkpoints" / "latest.ckpt",
    }


def build_collection_command(config: Mapping[str, Any]) -> tuple[list[str], Path, dict[str, str]]:
    task = config["task"]
    collection = config["collection"]
    paths = workflow_paths(config)
    per_robot = int(collection["total_trajectories"]) // 5
    script = paths["maniskill_root"] / "multi_robot_data_collection.py"
    command = [
        sys.executable,
        str(script),
        "--env", str(task["env_id"]),
        "--output-filename", str(config["paths"]["h5_filename"]),
        "--traj-num", str(per_robot),
        "--obs-mode", str(collection["obs_mode"]),
        "--control-mode", str(collection["control_mode"]),
        "--sim-backend-gen", str(collection["sim_backend_gen"]),
        "--sim-backend-replay", str(collection["sim_backend_replay"]),
        "--num-procs", str(collection["num_procs"]),
    ]
    if collection.get("save_video", False):
        command.append("--save-video")
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(collection["gpu_index"])
    return command, paths["maniskill_root"], env


def build_conversion_command(config: Mapping[str, Any]) -> list[str]:
    paths = workflow_paths(config)
    conversion = config["conversion"]
    return [
        sys.executable,
        str(ROOT_DIR / "convert_hdf5_to_umi_zarr.py"),
        "--input-h5-path", str(paths["h5"]),
        "--output-path", str(paths["dataset"]),
        "--camera-name", str(conversion["camera_name"]),
        "--image-size", str(conversion["image_size"]),
    ]


def build_validation_command(config: Mapping[str, Any]) -> list[str]:
    paths = workflow_paths(config)
    return [
        sys.executable,
        str(ROOT_DIR / "scripts_maniskill" / "validate_umi_dataset.py"),
        "--dataset", str(paths["dataset"]),
        "--image-size", str(config["conversion"]["image_size"]),
    ]


def build_training_command(config: Mapping[str, Any]) -> list[str]:
    paths = workflow_paths(config)
    training = config["training"]
    command = [
        sys.executable,
        str(ROOT_DIR / "train.py"),
        f"--config-dir={_path(training['config_dir'])}",
        f"--config-name={training['config_name']}",
        f"task.dataset_path={paths['dataset']}",
        f"env_name={config['task']['env_name']}",
        f"training.device={training['device']}",
        f"logging.mode={training['logging_mode']}",
        f"hydra.run.dir={paths['train_output']}",
    ]
    for key, value in training["overrides"].items():
        command.append(f"{key}={_hydra_value(value)}")
    return command


def build_evaluation_commands(config: Mapping[str, Any]) -> list[list[str]]:
    paths = workflow_paths(config)
    evaluation = config["evaluation"]
    commands: list[list[str]] = []
    for _, profile_path in evaluation["profiles"].items():
        for robot in evaluation["robots"]:
            command = [
                sys.executable,
                str(ROOT_DIR / "eval_sim_single_robot.py"),
                # Training workspaces write ``checkpoints/latest.ckpt`` while
                # some existing evaluation artifacts use ``ckpt/latest.ckpt``.
                # Passing the exact file avoids relying on either directory
                # convention and keeps resumed workflows unambiguous.
                "--input", str(paths["checkpoint"]),
                "--ckpt_filename", "latest",
                "--env_id", str(config["task"]["env_id"]),
                "--robot_uids", str(robot),
                "--sim_backend", str(evaluation["sim_backend"]),
                "--num_env", str(evaluation["num_env"]),
                "--num_eval_episodes", str(evaluation["num_eval_episodes"]),
                "--env_seed", str(evaluation["env_seed"]),
                "--obs_mode", str(evaluation["obs_mode"]),
                "--render_mode", str(evaluation["render_mode"]),
                "--steps_per_inference", str(evaluation["steps_per_inference"]),
                "--max_episode_steps", str(evaluation["max_episode_steps"]),
                "--policy-config", str(_path(profile_path)),
            ]
            if evaluation.get("obstacle", False):
                command.append("--obstacle")
            noise = evaluation.get("obstacle_observation_noise")
            if noise is not None:
                if not isinstance(noise, list) or len(noise) != 3:
                    raise WorkflowConfigError("evaluation.obstacle_observation_noise must have three values")
                command.extend(["--obstacle-observation-noise", *(str(value) for value in noise)])
            commands.append(command)
    return commands


def _check_required_artifact(path: Path, label: str) -> None:
    if not path.exists():
        raise WorkflowConfigError(f"{label} does not exist: {path}")


def _check_new_artifact(path: Path, label: str) -> None:
    if path.exists():
        raise WorkflowConfigError(
            f"refusing to overwrite existing {label}: {path}; choose a new path in the workflow YAML"
        )


def _hydra_value(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def run_command(command: Iterable[str], *, cwd: Path | None = None, env: Mapping[str, str] | None = None, dry_run: bool = False) -> None:
    command = [str(part) for part in command]
    printable = shlex.join(command)
    print(f"$ {printable}")
    if dry_run:
        return
    subprocess.run(command, cwd=str(cwd) if cwd else None, env=dict(env) if env else None, check=True)


def run_stage(config: Mapping[str, Any], stage: str, *, dry_run: bool = False) -> None:
    paths = workflow_paths(config)
    if stage == "collect":
        command, cwd, env = build_collection_command(config)
        if not dry_run:
            _check_required_artifact(cwd / "multi_robot_data_collection.py", "ManiSkill collection script")
            _check_new_artifact(paths["h5"], "HDF5 demos")
        run_command(command, cwd=cwd, env=env, dry_run=dry_run)
    elif stage == "convert":
        if not dry_run:
            _check_required_artifact(paths["h5"], "HDF5 demos")
            _check_new_artifact(paths["dataset"], "UMI dataset")
            paths["dataset"].parent.mkdir(parents=True, exist_ok=True)
        run_command(build_conversion_command(config), cwd=ROOT_DIR, dry_run=dry_run)
    elif stage == "validate":
        if not dry_run:
            _check_required_artifact(paths["dataset"], "UMI dataset")
        run_command(build_validation_command(config), cwd=ROOT_DIR, dry_run=dry_run)
    elif stage == "train":
        if not dry_run:
            _check_required_artifact(paths["dataset"], "UMI dataset")
            resume = bool(config["training"]["overrides"].get("training.resume", False))
            if paths["train_output"].exists() and any(paths["train_output"].iterdir()) and not resume:
                raise WorkflowConfigError(
                    "training output is not empty while training.resume is false: "
                    f"{paths['train_output']}"
                )
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(config["training"]["gpu_index"])
        run_command(build_training_command(config), cwd=ROOT_DIR, env=env, dry_run=dry_run)
    elif stage == "eval":
        if not dry_run:
            _check_required_artifact(paths["checkpoint"], "trained checkpoint")
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(config["evaluation"]["gpu_index"])
        for command in build_evaluation_commands(config):
            run_command(command, cwd=ROOT_DIR, env=env, dry_run=dry_run)
    else:  # pragma: no cover - parser prevents this
        raise WorkflowConfigError(f"unsupported stage {stage}")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(ROOT_DIR / "configs/workflows/simulation.yaml"))
    parser.add_argument("--stage", choices=("collect", "convert", "validate", "train", "eval", "all"), default="all")
    parser.add_argument("--dry-run", action="store_true", help="Print validated commands without starting external processes")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    try:
        config = load_workflow_config(args.config)
        stages = ("collect", "convert", "validate", "train", "eval") if args.stage == "all" else (args.stage,)
        for stage in stages:
            print(f"\n=== {stage} ===")
            run_stage(config, stage, dry_run=args.dry_run)
    except WorkflowConfigError as exc:
        raise SystemExit(f"workflow configuration error: {exc}") from exc


if __name__ == "__main__":
    main()
