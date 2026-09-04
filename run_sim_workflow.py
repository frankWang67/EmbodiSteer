#!/usr/bin/env python3
"""Run the reproducible simulation data/train/evaluation workflow.

The workflow is deliberately a thin, config-driven orchestrator.  It invokes
the pinned ManiSkill collection utility, the repository's HDF5 converter, the
existing Hydra training entry point, and the public simulation evaluator in
sequence.  No data or checkpoint is copied into the source release tree.

Examples
--------
Preview every command without requiring data or starting a process::

    ./run_sim_pipeline.sh \
      --config configs/workflows/simulation.yaml --stage all --dry-run

Run only conversion, training, or evaluation after the preceding artifact is
available::

    ./run_sim_pipeline.sh --stage convert
    ./run_sim_pipeline.sh --stage train
    ./run_sim_pipeline.sh --stage eval
"""

from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

import yaml

from embodisteer.evaluation import (
    validate_output_name,
    workflow_result_dir,
    write_workflow_results,
)
from embodisteer.runtime_config import PolicyConfigError, load_policy_config


ROOT_DIR = Path(__file__).resolve().parent


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


EVALUATION_RUNTIME_FIELDS = (
    "sim_backend",
    "num_env",
    "num_eval_episodes",
    "env_seed",
    "obs_mode",
    "render_mode",
    "steps_per_inference",
    "max_episode_steps",
    "obstacle",
    "obstacle_observation_noise",
    "control_mode",
)

PROFILE_FIELDS = {"policy_config", "robots", *EVALUATION_RUNTIME_FIELDS}


def _optional_mapping(value: Any, name: str) -> dict[str, Any]:
    if value is None:
        return {}
    return _mapping(value, name)


def _validate_robot_list(value: Any, name: str) -> list[str]:
    if (
        not isinstance(value, list)
        or not value
        or not all(isinstance(robot, str) and robot for robot in value)
    ):
        raise WorkflowConfigError(f"{name} must be a non-empty list of robot UID strings")
    if len(value) != len(set(value)):
        raise WorkflowConfigError(f"{name} must not contain duplicates")
    for robot in value:
        try:
            validate_output_name(robot, "robot UID")
        except ValueError as exc:
            raise WorkflowConfigError(str(exc)) from exc
    return list(value)


def _validate_positive_integer(value: Any, name: str, *, allow_zero: bool = False) -> None:
    minimum = 0 if allow_zero else 1
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        qualifier = "non-negative" if allow_zero else "positive"
        raise WorkflowConfigError(f"{name} must be a {qualifier} integer")


def _validate_noise(value: Any, name: str) -> list[float] | None:
    if value is None:
        return None
    if not isinstance(value, list) or len(value) != 3:
        raise WorkflowConfigError(f"{name} must contain three numeric values")
    if any(isinstance(item, bool) or not isinstance(item, (int, float)) for item in value):
        raise WorkflowConfigError(f"{name} must contain three numeric values")
    result = [float(item) for item in value]
    if any(item < 0 for item in result) or not any(item > 0 for item in result):
        raise WorkflowConfigError(
            f"{name} must be non-negative and contain at least one positive value"
        )
    return result


def _effective_profile(evaluation: Mapping[str, Any], profile: Mapping[str, Any]) -> dict[str, Any]:
    result = {key: evaluation.get(key) for key in EVALUATION_RUNTIME_FIELDS}
    result.update({key: value for key, value in profile.items() if key in EVALUATION_RUNTIME_FIELDS})
    result["robots"] = list(profile.get("robots", evaluation["robots"]))
    return result


def _validate_effective_profile(
    name: str,
    runtime: Mapping[str, Any],
    policy_settings: Mapping[str, Any],
) -> None:
    _validate_robot_list(runtime["robots"], f"evaluation.profiles.{name}.robots")
    for field in ("num_env", "num_eval_episodes", "max_episode_steps"):
        _validate_positive_integer(runtime[field], f"evaluation.profiles.{name}.{field}")
    if isinstance(runtime["env_seed"], bool) or not isinstance(runtime["env_seed"], int):
        raise WorkflowConfigError(f"evaluation.profiles.{name}.env_seed must be an integer")
    for field in ("sim_backend", "obs_mode", "render_mode"):
        if not isinstance(runtime[field], str) or not runtime[field]:
            raise WorkflowConfigError(
                f"evaluation.profiles.{name}.{field} must be a non-empty string"
            )
    _validate_positive_integer(
        runtime["steps_per_inference"],
        f"evaluation.profiles.{name}.steps_per_inference",
        allow_zero=True,
    )
    if runtime["num_eval_episodes"] % runtime["num_env"]:
        raise WorkflowConfigError(
            f"evaluation.profiles.{name}.num_eval_episodes must be divisible by num_env"
        )
    if not isinstance(runtime["obstacle"], bool):
        raise WorkflowConfigError(f"evaluation.profiles.{name}.obstacle must be a YAML boolean")
    noise = _validate_noise(
        runtime.get("obstacle_observation_noise"),
        f"evaluation.profiles.{name}.obstacle_observation_noise",
    )
    needs_obstacles = bool(policy_settings["guidance"] or policy_settings["baseline_method"])
    if needs_obstacles and not runtime["obstacle"]:
        raise WorkflowConfigError(f"evaluation profile {name} requires obstacle: true")
    if noise is not None and (not runtime["obstacle"] or not needs_obstacles):
        raise WorkflowConfigError(
            f"evaluation profile {name} can use obstacle noise only with an obstacle-aware policy"
        )
    control_mode = runtime.get("control_mode")
    is_joint = policy_settings["inference_space"] == "joint" or bool(policy_settings["baseline_method"])
    if control_mode is not None:
        if not isinstance(control_mode, str) or not control_mode:
            raise WorkflowConfigError(f"evaluation.profiles.{name}.control_mode must be null or a string")
        if is_joint and not control_mode.startswith("pd_joint"):
            raise WorkflowConfigError(f"evaluation profile {name} requires a pd_joint* control mode")


def load_workflow_config(
    path: str | os.PathLike[str], *, stage: str = "all"
) -> dict[str, Any]:
    """Load a schema-versioned workflow, with a compact eval-only form."""
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

    if stage not in {"collect", "convert", "validate", "train", "eval", "all"}:
        raise WorkflowConfigError(f"unsupported stage {stage}")
    task = _mapping(document.get("task"), "task")
    paths = _optional_mapping(document.get("paths"), "paths")
    collection = _optional_mapping(document.get("collection"), "collection")
    conversion = _optional_mapping(document.get("conversion"), "conversion")
    training = _optional_mapping(document.get("training"), "training")
    evaluation = _mapping(document.get("evaluation"), "evaluation")

    required = {
        "task": ("name", "env_id", "env_name"),
        "paths": ("maniskill_root", "data_root", "h5_filename", "dataset_filename", "train_output_dir"),
        "collection": ("robot_uids", "total_trajectories", "obs_mode", "control_mode", "sim_backend_gen", "sim_backend_replay", "num_procs"),
        "conversion": ("camera_name", "image_size"),
        "training": ("config_dir", "config_name", "device", "logging_mode", "overrides"),
        "evaluation": ("profiles", "robots", "sim_backend", "num_env", "num_eval_episodes", "env_seed", "obs_mode", "render_mode", "steps_per_inference", "max_episode_steps", "obstacle"),
    }
    if stage == "eval":
        required["task"] = ("env_id",)
    allowed = {
        "task": {"name", "env_id", "env_name"},
        "paths": set(required["paths"]),
        "collection": set(required["collection"]) | {"save_video"},
        "conversion": set(required["conversion"]),
        "training": set(required["training"]),
        "evaluation": set(required["evaluation"]) | {
            "obstacle_observation_noise",
            "control_mode",
            "checkpoint",
            "output_dir",
            "run_id",
            "continue_on_error",
        },
    }
    sections = {
        "task": task,
        "paths": paths,
        "collection": collection,
        "conversion": conversion,
        "training": training,
        "evaluation": evaluation,
    }
    required_sections = {"task", "paths", "collection", "conversion", "training", "evaluation"}
    if stage == "eval":
        required_sections = {"task", "evaluation"}
    for section, keys in required.items():
        if section not in required_sections:
            if sections[section]:
                if section in {"collection", "conversion", "training"}:
                    missing = [key for key in keys if key not in sections[section]]
                    if missing:
                        raise WorkflowConfigError(
                            f"{section} is missing fields: {', '.join(missing)}"
                        )
                _reject_unknown(sections[section], allowed[section], section)
            continue
        missing = [key for key in keys if key not in sections[section]]
        if missing:
            raise WorkflowConfigError(f"{section} is missing fields: {', '.join(missing)}")
        _reject_unknown(sections[section], allowed[section], section)

    if collection:
        collection_robots = _validate_robot_list(collection["robot_uids"], "collection.robot_uids")
        total = collection["total_trajectories"]
        robot_count = len(collection_robots)
        if (
            isinstance(total, bool)
            or not isinstance(total, int)
            or total < robot_count
            or total % robot_count
        ):
            raise WorkflowConfigError(
                "collection.total_trajectories must be a positive multiple of "
                "the number of collection.robot_uids"
            )
        _validate_positive_integer(collection["num_procs"], "collection.num_procs")
    if conversion:
        _validate_positive_integer(conversion["image_size"], "conversion.image_size")
    for section_name, section, key in (
        ("collection", collection, "save_video"),
        ("evaluation", evaluation, "obstacle"),
    ):
        if section and key in section and not isinstance(section[key], bool):
            raise WorkflowConfigError(f"{section_name}.{key} must be a YAML boolean")
    for filename_key in ("h5_filename", "dataset_filename"):
        if filename_key not in paths:
            continue
        filename = str(paths[filename_key])
        if Path(filename).name != filename:
            raise WorkflowConfigError(f"paths.{filename_key} must be a filename, not a path")
    evaluation["robots"] = _validate_robot_list(evaluation["robots"], "evaluation.robots")
    evaluation.setdefault("obstacle_observation_noise", None)
    evaluation.setdefault("control_mode", None)
    evaluation.setdefault("run_id", "default")
    evaluation.setdefault("continue_on_error", True)
    try:
        evaluation["run_id"] = validate_output_name(evaluation["run_id"], "evaluation.run_id")
    except ValueError as exc:
        raise WorkflowConfigError(str(exc)) from exc
    if not isinstance(evaluation["continue_on_error"], bool):
        raise WorkflowConfigError("evaluation.continue_on_error must be a YAML boolean")
    profiles = evaluation["profiles"]
    if not isinstance(profiles, Mapping) or not profiles:
        raise WorkflowConfigError("evaluation.profiles must be a non-empty name-to-YAML mapping")
    normalized_profiles: dict[str, dict[str, Any]] = {}
    for name, profile_value in profiles.items():
        try:
            name = validate_output_name(name, "evaluation profile name")
        except (TypeError, ValueError) as exc:
            raise WorkflowConfigError(str(exc)) from exc
        if isinstance(profile_value, str):
            profile = {"policy_config": profile_value}
        elif isinstance(profile_value, Mapping):
            profile = dict(profile_value)
        else:
            raise WorkflowConfigError(
                "evaluation.profiles values must be a policy YAML path or a mapping"
            )
        _reject_unknown(profile, PROFILE_FIELDS, f"evaluation.profiles.{name}")
        policy_path = profile.get("policy_config")
        if not isinstance(policy_path, str) or not policy_path:
            raise WorkflowConfigError(f"evaluation.profiles.{name}.policy_config is required")
        try:
            policy_settings = load_policy_config(_path(policy_path))
        except PolicyConfigError as exc:
            raise WorkflowConfigError(f"invalid evaluation profile {name}: {exc}") from exc
        profile["policy_config"] = policy_settings["config_path"]
        if "robots" in profile:
            profile["robots"] = _validate_robot_list(
                profile["robots"], f"evaluation.profiles.{name}.robots"
            )
        runtime = _effective_profile(evaluation, profile)
        _validate_effective_profile(name, runtime, policy_settings)
        profile["policy_settings"] = policy_settings
        normalized_profiles[name] = profile
    evaluation["profiles"] = normalized_profiles
    if training and not isinstance(training["overrides"], Mapping):
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
    evaluation = config["evaluation"]
    result: dict[str, Path] = {}
    if "maniskill_root" in paths:
        maniskill_root = _path(paths["maniskill_root"])
        result["maniskill_root"] = maniskill_root
        if "h5_filename" in paths:
            result["h5"] = (
                maniskill_root
                / "demos"
                / str(task["env_id"])
                / "motionplanning"
                / str(paths["h5_filename"])
            )
    if "data_root" in paths and "dataset_filename" in paths:
        result["dataset"] = _path(paths["data_root"]) / str(paths["dataset_filename"])
    train_output = _path(paths["train_output_dir"]) if "train_output_dir" in paths else None
    if train_output is not None:
        result["train_output"] = train_output
    checkpoint_value = evaluation.get("checkpoint")
    if checkpoint_value is not None:
        result["checkpoint"] = _path(checkpoint_value)
    elif train_output is not None:
        result["checkpoint"] = train_output / "checkpoints" / "latest.ckpt"
    output_value = evaluation.get("output_dir")
    if output_value is not None:
        output_base = _path(output_value)
    elif train_output is not None:
        output_base = train_output / "evaluations"
    else:  # guarded by load_workflow_config
        raise WorkflowConfigError("evaluation output directory cannot be resolved")
    result["evaluation_output_base"] = output_base
    result["evaluation_output"] = output_base / str(evaluation["run_id"])
    return result


def build_collection_command(config: Mapping[str, Any]) -> tuple[list[str], Path, dict[str, str]]:
    task = config["task"]
    collection = config["collection"]
    paths = workflow_paths(config)
    robot_uids = [str(robot) for robot in collection["robot_uids"]]
    per_robot = int(collection["total_trajectories"]) // len(robot_uids)
    script = paths["maniskill_root"] / "multi_robot_data_collection.py"
    command = [
        sys.executable,
        str(script),
        "--env", str(task["env_id"]),
        "--output-filename", str(config["paths"]["h5_filename"]),
        "--traj-num", str(per_robot),
        "--robot-uids", *robot_uids,
        "--obs-mode", str(collection["obs_mode"]),
        "--control-mode", str(collection["control_mode"]),
        "--sim-backend-gen", str(collection["sim_backend_gen"]),
        "--sim-backend-replay", str(collection["sim_backend_replay"]),
        "--num-procs", str(collection["num_procs"]),
    ]
    if collection.get("save_video", False):
        command.append("--save-video")
    return command, paths["maniskill_root"], os.environ.copy()


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


def build_evaluation_jobs(config: Mapping[str, Any]) -> list[dict[str, Any]]:
    paths = workflow_paths(config)
    evaluation = config["evaluation"]
    jobs: list[dict[str, Any]] = []
    for profile_name, profile in evaluation["profiles"].items():
        runtime = _effective_profile(evaluation, profile)
        for robot in runtime["robots"]:
            result_dir = workflow_result_dir(
                paths["evaluation_output"], profile_name, robot
            )
            command = [
                sys.executable,
                str(ROOT_DIR / "eval_sim_single_robot.py"),
                # Training workspaces write ``checkpoints/latest.ckpt`` while
                # some existing evaluation artifacts use ``ckpt/latest.ckpt``.
                # Passing the exact file avoids relying on either directory
                # convention and keeps resumed workflows unambiguous.
                "--input", str(paths["checkpoint"]),
                "--ckpt_filename", "latest",
                "--output-dir", str(paths["evaluation_output"]),
                "--profile-name", str(profile_name),
                "--env_id", str(config["task"]["env_id"]),
                "--robot_uids", str(robot),
                "--sim_backend", str(runtime["sim_backend"]),
                "--num_env", str(runtime["num_env"]),
                "--num_eval_episodes", str(runtime["num_eval_episodes"]),
                "--env_seed", str(runtime["env_seed"]),
                "--obs_mode", str(runtime["obs_mode"]),
                "--render_mode", str(runtime["render_mode"]),
                "--steps_per_inference", str(runtime["steps_per_inference"]),
                "--max_episode_steps", str(runtime["max_episode_steps"]),
                "--policy-config", str(profile["policy_config"]),
            ]
            if runtime.get("control_mode"):
                command.extend(["--control_mode", str(runtime["control_mode"])])
            if runtime.get("obstacle", False):
                command.append("--obstacle")
            noise = runtime.get("obstacle_observation_noise")
            if noise is not None:
                command.extend(["--obstacle-observation-noise", *(str(value) for value in noise)])
            jobs.append(
                {
                    "profile": profile_name,
                    "robot": robot,
                    "policy_config": str(profile["policy_config"]),
                    "runtime": runtime,
                    "command": command,
                    "result_dir": str(result_dir),
                    "status": "pending",
                }
            )
    return jobs


def build_evaluation_commands(config: Mapping[str, Any]) -> list[list[str]]:
    """Compatibility helper used by tests and command previews."""
    return [job["command"] for job in build_evaluation_jobs(config)]


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


def apply_evaluation_overrides(
    config: dict[str, Any],
    *,
    checkpoint: str | None = None,
    output_dir: str | None = None,
    run_id: str | None = None,
    robots: list[str] | None = None,
) -> tuple[str, ...]:
    """Apply operational eval overrides without changing algorithm profiles."""
    changed: list[str] = []
    evaluation = config["evaluation"]
    if checkpoint is not None:
        evaluation["checkpoint"] = checkpoint
        changed.append("checkpoint")
    if output_dir is not None:
        evaluation["output_dir"] = output_dir
        changed.append("output_dir")
    if run_id is not None:
        try:
            evaluation["run_id"] = validate_output_name(run_id, "--run-id")
        except ValueError as exc:
            raise WorkflowConfigError(str(exc)) from exc
        changed.append("run_id")
    if robots is not None:
        robots = _validate_robot_list(robots, "--robots")
        evaluation["robots"] = robots
        for profile in evaluation["profiles"].values():
            profile.pop("robots", None)
        changed.append("robots")
    return tuple(changed)


def run_command(command: Iterable[str], *, cwd: Path | None = None, env: Mapping[str, str] | None = None, dry_run: bool = False) -> None:
    command = [str(part) for part in command]
    printable = shlex.join(command)
    cuda_prefix = ""
    if env is not None and "CUDA_VISIBLE_DEVICES" in env:
        cuda_prefix = (
            "CUDA_VISIBLE_DEVICES="
            f"{shlex.quote(str(env['CUDA_VISIBLE_DEVICES']))} "
        )
    print(f"$ {cuda_prefix}{printable}")
    if dry_run:
        return
    subprocess.run(command, cwd=str(cwd) if cwd else None, env=dict(env) if env else None, check=True)


def _git_revision(path: Path) -> str | None:
    try:
        result = subprocess.run(
            ["git", "-C", str(path), "rev-parse", "HEAD"],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip() or None


def _git_dirty(path: Path) -> bool | None:
    try:
        result = subprocess.run(
            ["git", "-C", str(path), "status", "--porcelain"],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return bool(result.stdout.strip())


def _declared_dependency_revisions() -> dict[str, Any]:
    path = ROOT_DIR / "third_party" / "manifest.yaml"
    try:
        document = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError):
        return {}
    repositories = document.get("repositories", {}) if isinstance(document, Mapping) else {}
    return {
        str(name): values.get("commit")
        for name, values in repositories.items()
        if isinstance(values, Mapping) and values.get("commit")
    }


def _public_profile(profile: Mapping[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in profile.items() if key != "policy_settings"}


def _write_manifest(path: Path, manifest: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_name(f".{path.name}.tmp")
    temp_path.write_text(
        yaml.safe_dump(dict(manifest), sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    temp_path.replace(path)


def _build_manifest(
    config: Mapping[str, Any],
    jobs: list[dict[str, Any]],
) -> dict[str, Any]:
    paths = workflow_paths(config)
    maniskill_root = paths.get("maniskill_root", ROOT_DIR / "third_party/src/maniskill")
    profile_manifest = {
        name: {
            **_public_profile(profile),
            "resolved_policy": dict(profile["policy_settings"]),
        }
        for name, profile in config["evaluation"]["profiles"].items()
    }
    return {
        "schema_version": 1,
        "status": "running",
        "started_at": datetime.now(timezone.utc).isoformat(),
        "finished_at": None,
        "config_path": config["config_path"],
        "task": dict(config["task"]),
        "checkpoint": {
            "path": str(paths["checkpoint"]),
        },
        "output_dir": str(paths["evaluation_output"]),
        "reset_protocol": "unseeded_env_reset",
        "revisions": {
            "repository": {
                "commit": _git_revision(ROOT_DIR),
                "dirty": _git_dirty(ROOT_DIR),
            },
            "maniskill": {
                "commit": _git_revision(maniskill_root),
                "dirty": _git_dirty(maniskill_root),
            },
            "declared_dependencies": _declared_dependency_revisions(),
        },
        "profiles": profile_manifest,
        "jobs": jobs,
    }


def _resume_spec(manifest: Mapping[str, Any]) -> dict[str, Any]:
    """Return the unhashed run definition that must match when resuming."""
    checkpoint = manifest.get("checkpoint")
    checkpoint_path = (
        checkpoint.get("path") if isinstance(checkpoint, Mapping) else checkpoint
    )
    jobs = manifest.get("jobs")
    job_specs = []
    if isinstance(jobs, list):
        for job in jobs:
            if isinstance(job, Mapping):
                job_specs.append(
                    {
                        key: job.get(key)
                        for key in ("profile", "robot", "runtime")
                    }
                )
    return {
        "task": manifest.get("task"),
        "checkpoint_path": checkpoint_path,
        "output_dir": manifest.get("output_dir"),
        "profiles": manifest.get("profiles"),
        "jobs": job_specs,
    }


def _refresh_eval_reports(output_dir: Path, jobs: list[dict[str, Any]]) -> None:
    write_workflow_results(output_dir, jobs)


def run_evaluation_stage(
    config: Mapping[str, Any],
    *,
    dry_run: bool = False,
    resume: bool = False,
    force: bool = False,
) -> None:
    paths = workflow_paths(config)
    checkpoint = paths.get("checkpoint")
    if checkpoint is None:
        raise WorkflowConfigError(
            "evaluation checkpoint cannot be resolved; set evaluation.checkpoint "
            "or paths.train_output_dir"
        )
    output_dir = paths["evaluation_output"]
    if dry_run:
        for job in build_evaluation_jobs(config):
            run_command(
                job["command"], cwd=ROOT_DIR, env=os.environ.copy(), dry_run=True
            )
        print(f"# evaluation output: {output_dir}")
        return

    _check_required_artifact(checkpoint, "evaluation checkpoint")
    if output_dir.exists() and any(output_dir.iterdir()) and not (resume or force):
        raise WorkflowConfigError(
            f"evaluation output is not empty: {output_dir}; use --resume, "
            "--force, or choose another --run-id"
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    jobs = build_evaluation_jobs(config)
    manifest_path = output_dir / "run_manifest.yaml"
    manifest = _build_manifest(config, jobs)
    existing_manifest = None
    if resume and manifest_path.exists():
        try:
            existing_manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
        except (OSError, yaml.YAMLError) as exc:
            raise WorkflowConfigError(f"cannot resume invalid manifest {manifest_path}: {exc}") from exc
        if not isinstance(existing_manifest, Mapping):
            raise WorkflowConfigError(f"cannot resume invalid manifest {manifest_path}")
        if _resume_spec(existing_manifest) != _resume_spec(manifest):
            raise WorkflowConfigError(
                "cannot resume because the checkpoint path, task, profiles, robots "
                "or evaluation settings changed"
            )
        manifest["started_at"] = existing_manifest.get("started_at", manifest["started_at"])

    env = os.environ.copy()
    _write_manifest(manifest_path, manifest)
    _refresh_eval_reports(output_dir, jobs)
    failures = 0
    for job in jobs:
        result_dir = Path(job["result_dir"])
        if resume and (result_dir / "metrics.json").is_file():
            job["status"] = "skipped"
            print(f"Skipping completed evaluation: {job['profile']} / {job['robot']}")
        else:
            job["status"] = "running"
            if force:
                # Remove only exact, regenerable metric artifacts. Videos are
                # left to ManiSkill's recorder so --force never recursively
                # deletes a user-controlled directory.
                for filename in (
                    "metrics.json",
                    "episode_metrics.npz",
                    "eval_results.txt",
                ):
                    artifact = result_dir / filename
                    if artifact.is_file():
                        artifact.unlink()
            _write_manifest(manifest_path, manifest)
            try:
                run_command(job["command"], cwd=ROOT_DIR, env=env)
            except subprocess.CalledProcessError as exc:
                failures += 1
                job["status"] = "failed"
                job["error"] = f"command exited with status {exc.returncode}"
                print(
                    f"Evaluation failed: {job['profile']} / {job['robot']}: "
                    f"{job['error']}",
                    file=sys.stderr,
                )
            else:
                job["status"] = "completed"
        _refresh_eval_reports(output_dir, jobs)
        _write_manifest(manifest_path, manifest)
        if failures and not config["evaluation"]["continue_on_error"]:
            break

    manifest["finished_at"] = datetime.now(timezone.utc).isoformat()
    completed_jobs = sum(job["status"] in {"completed", "skipped"} for job in jobs)
    manifest["status"] = (
        "partial" if failures and completed_jobs else "failed" if failures else "completed"
    )
    _refresh_eval_reports(output_dir, jobs)
    _write_manifest(manifest_path, manifest)
    print(f"Evaluation summaries: {output_dir / 'results.md'} and {output_dir / 'results.json'}")
    if failures:
        raise WorkflowConfigError(
            f"{failures} evaluation job(s) failed; completed results were preserved in {output_dir}"
        )


def run_stage(
    config: Mapping[str, Any],
    stage: str,
    *,
    dry_run: bool = False,
    resume: bool = False,
    force: bool = False,
) -> None:
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
        run_command(
            build_training_command(config),
            cwd=ROOT_DIR,
            env=os.environ.copy(),
            dry_run=dry_run,
        )
    elif stage == "eval":
        run_evaluation_stage(
            config, dry_run=dry_run, resume=resume, force=force
        )
    else:  # pragma: no cover - parser prevents this
        raise WorkflowConfigError(f"unsupported stage {stage}")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(ROOT_DIR / "configs/workflows/simulation.yaml"))
    parser.add_argument("--stage", choices=("collect", "convert", "validate", "train", "eval", "all"), default="all")
    parser.add_argument("--checkpoint", help="Checkpoint path for eval; overrides the workflow YAML")
    parser.add_argument("--output-dir", help="Evaluation output base directory; overrides the workflow YAML")
    parser.add_argument("--run-id", help="Evaluation run directory name; overrides the workflow YAML")
    parser.add_argument(
        "--robots",
        nargs="+",
        help="Evaluate only these robot UIDs, overriding all profile robot lists",
    )
    eval_mode = parser.add_mutually_exclusive_group()
    eval_mode.add_argument(
        "--resume",
        action="store_true",
        help="Skip jobs whose structured metrics already exist",
    )
    eval_mode.add_argument(
        "--force",
        action="store_true",
        help="Re-run jobs and overwrite result files in the selected run directory",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print validated commands without starting external processes")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    try:
        config = load_workflow_config(args.config, stage=args.stage)
        eval_overrides = apply_evaluation_overrides(
            config,
            checkpoint=args.checkpoint,
            output_dir=args.output_dir,
            run_id=args.run_id,
            robots=args.robots,
        )
        if eval_overrides and args.stage not in {"eval", "all"}:
            raise WorkflowConfigError(
                "--checkpoint, --output-dir, --run-id and --robots apply only to eval or all"
            )
        if (args.resume or args.force) and args.stage not in {"eval", "all"}:
            raise WorkflowConfigError("--resume and --force apply only to eval or all")
        if eval_overrides:
            print(f"Evaluation overrides: {', '.join(eval_overrides)}")
        stages = ("collect", "convert", "validate", "train", "eval") if args.stage == "all" else (args.stage,)
        for stage in stages:
            print(f"\n=== {stage} ===")
            run_stage(
                config,
                stage,
                dry_run=args.dry_run,
                resume=args.resume,
                force=args.force,
            )
    except (WorkflowConfigError, ValueError) as exc:
        raise SystemExit(f"workflow configuration error: {exc}") from exc


if __name__ == "__main__":
    main()
