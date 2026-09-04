"""Shared evaluation paths, serialization and multi-robot aggregation."""

from __future__ import annotations

import json
import math
import os
import re
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np

SAFE_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


def evaluation_subdir(
    *,
    inference_space: str,
    guidance: str,
    baseline_method: str,
    obstacle: bool,
    obstacle_observation_noise=None,
) -> str:
    """Return a collision-resistant result directory for one policy method.

    The method name is part of the path so runs with the same checkpoint and
    environment cannot silently overwrite one another (for example joint CBF
    and joint GD).
    """
    inference_space = str(inference_space).lower().strip()
    guidance = str(guidance).lower().strip()
    baseline_method = str(baseline_method).lower().strip()
    if baseline_method:
        subdir = f"obstacle_baseline_{baseline_method}"
    elif inference_space == "joint":
        if guidance:
            subdir = f"obstacle_joint_space_guidance_{guidance}"
        else:
            subdir = (
                "obstacle_joint_space" if obstacle
                else "no_obstacle_joint_space"
            )
    elif guidance:
        subdir = f"obstacle_ee_space_guidance_{guidance}"
    else:
        subdir = "obstacle_ee_space" if obstacle else "no_obstacle_ee_space"

    if obstacle_observation_noise is not None:
        pos_std, size_std, rot_std = obstacle_observation_noise
        subdir = os.path.join(
            subdir,
            f"obs_noise_pos{float(pos_std):.6g}_"
            f"size{float(size_std):.6g}_rot{float(rot_std):.6g}",
        )
    return subdir


def validate_output_name(value: str, label: str) -> str:
    """Validate a user-controlled path component."""
    value = str(value).strip()
    if not SAFE_NAME_RE.fullmatch(value):
        raise ValueError(
            f"{label} must start with an alphanumeric character and contain "
            "only letters, numbers, '.', '_' or '-'"
        )
    return value


def workflow_result_dir(
    output_dir: str | os.PathLike[str], profile_name: str, robot_uid: str
) -> Path:
    """Return the isolated result directory for one workflow evaluation job."""
    profile_name = validate_output_name(profile_name, "profile name")
    robot_uid = validate_output_name(robot_uid, "robot UID")
    return Path(output_dir) / "profiles" / profile_name / robot_uid


def parse_legacy_results(path: str | os.PathLike[str]) -> dict[str, Any]:
    """Parse the line-oriented ``eval_results.txt`` compatibility format."""
    result: dict[str, Any] = {}
    try:
        lines = Path(path).read_text(encoding="utf-8").splitlines()
    except OSError:
        return result
    for line in lines:
        if ":" not in line:
            continue
        key, value = (part.strip() for part in line.split(":", 1))
        if not key:
            continue
        if value == "N/A":
            result[key] = None
            continue
        try:
            result[key] = float(value)
        except ValueError:
            result[key] = value
    return result


def load_job_metrics(result_dir: str | os.PathLike[str]) -> dict[str, Any] | None:
    """Load structured metrics, falling back to the compatibility text file."""
    result_dir = Path(result_dir)
    json_path = result_dir / "metrics.json"
    try:
        payload = json.loads(json_path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        legacy = parse_legacy_results(result_dir / "eval_results.txt")
        return {"metrics": legacy} if legacy else None
    except (OSError, json.JSONDecodeError) as exc:
        return {"error": f"cannot read {json_path}: {exc}", "metrics": {}}
    if not isinstance(payload, dict):
        return {"error": f"{json_path} is not a JSON object", "metrics": {}}
    return payload


def _finite_numbers(values: Iterable[Any]) -> list[float]:
    result = []
    for value in values:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            continue
        value = float(value)
        if math.isfinite(value):
            result.append(value)
    return result


def macro_average(rows: Iterable[Mapping[str, Any]]) -> dict[str, float]:
    """Compute explicitly labelled robot-level macro averages.

    Standard-deviation fields are deliberately excluded: averaging per-robot
    standard deviations is not a pooled standard deviation.
    """
    metrics = [dict(row.get("metrics", {})) for row in rows]
    keys = sorted({key for row in metrics for key in row})
    result: dict[str, float] = {}
    for key in keys:
        if key.startswith("std_") or key.endswith("_count"):
            continue
        values = _finite_numbers(row.get(key) for row in metrics)
        if values:
            result[key] = sum(values) / len(values)
    return result


def pooled_episode_metrics(rows: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    """Pool saved episode arrays across robots for episode-weighted metrics."""
    arrays: dict[str, list[np.ndarray]] = {}
    for row in rows:
        path = Path(str(row["result_dir"])) / "episode_metrics.npz"
        try:
            with np.load(path, allow_pickle=False) as payload:
                for key in (
                    "success_once",
                    "success_at_end",
                    "collision_count",
                    "max_reward",
                    "first_success_step",
                    "fail_once",
                ):
                    if key in payload:
                        arrays.setdefault(key, []).append(np.asarray(payload[key]).reshape(-1))
        except (OSError, ValueError):
            continue
    pooled = {
        key: np.concatenate(values) for key, values in arrays.items() if values
    }
    if not pooled:
        return {}
    result: dict[str, Any] = {}
    if "success_once" in pooled:
        result["success_once_rate"] = float(np.mean(pooled["success_once"]))
        result["episode_count"] = int(pooled["success_once"].size)
    if "success_at_end" in pooled:
        result["success_at_end_rate"] = float(np.mean(pooled["success_at_end"]))
    if "collision_count" in pooled:
        result["avg_collision_per_episode"] = float(np.mean(pooled["collision_count"]))
        result["std_collision_per_episode"] = float(np.std(pooled["collision_count"]))
        result["fail_rate_all"] = float(np.mean(pooled["collision_count"] > 0))
    if "max_reward" in pooled:
        result["avg_max_reward_per_episode"] = float(np.mean(pooled["max_reward"]))
        result["std_max_reward_per_episode"] = float(np.std(pooled["max_reward"]))
    if "first_success_step" in pooled:
        succeeded = pooled["first_success_step"] >= 0
        if np.any(succeeded):
            result["avg_first_success_step"] = float(
                np.mean(pooled["first_success_step"][succeeded])
            )
            result["std_first_success_step"] = float(
                np.std(pooled["first_success_step"][succeeded])
            )
        else:
            result["avg_first_success_step"] = None
            result["std_first_success_step"] = None
    if "fail_once" in pooled and "success_once" in pooled:
        non_success = pooled["success_once"] == 0
        result["fail_rate_among_non_success"] = (
            float(np.mean(pooled["fail_once"][non_success]))
            if np.any(non_success)
            else None
        )
    return result


def aggregate_workflow_results(
    output_dir: str | os.PathLike[str],
    jobs: Iterable[Mapping[str, Any]],
) -> dict[str, Any]:
    """Collect completed/failed jobs into one machine-readable result object."""
    profiles: dict[str, dict[str, Any]] = {}
    for job in jobs:
        profile = str(job["profile"])
        robot = str(job["robot"])
        profile_result = profiles.setdefault(profile, {"robots": {}})
        result_dir = Path(str(job["result_dir"]))
        payload = load_job_metrics(result_dir)
        status = str(job.get("status", "pending"))
        row: dict[str, Any] = {
            "status": status,
            "result_dir": str(result_dir),
            "metrics": {},
        }
        if payload is not None:
            row.update(payload)
            row["status"] = "completed" if status in {"pending", "running"} else status
        if job.get("error"):
            row["error"] = str(job["error"])
        profile_result["robots"][robot] = row

    for profile_result in profiles.values():
        completed = [
            row
            for row in profile_result["robots"].values()
            if row.get("status") in {"completed", "skipped"} and row.get("metrics")
        ]
        profile_result["macro_average"] = macro_average(completed)
        profile_result["pooled_episode_metrics"] = pooled_episode_metrics(completed)

    return {
        "schema_version": 1,
        "aggregation": (
            "macro_average is the arithmetic mean of available robot-level "
            "metrics; per-robot standard deviations are not averaged"
        ),
        "profiles": profiles,
    }


def _pct(value: Any) -> str:
    return "N/A" if value is None else f"{float(value) * 100:.1f}%"


def _mean_std(metrics: Mapping[str, Any], mean_key: str, std_key: str) -> str:
    mean = metrics.get(mean_key)
    if mean is None:
        return "N/A"
    std = metrics.get(std_key)
    return f"{float(mean):.3f}" if std is None else f"{float(mean):.3f}±{float(std):.3f}"


def render_results_markdown(results: Mapping[str, Any]) -> str:
    """Render the common paper-facing metrics without dropping custom fields."""
    lines = [
        "# Simulation evaluation results",
        "",
        "> Average rows are robot-level macro averages. Per-robot standard "
        "deviations are not averaged or presented as pooled deviations.",
        "",
        "## Profile comparison",
        "",
        "| Profile | Completed robots | Macro success once | Pooled success once | Pooled collisions/ep | Pooled max reward/ep |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for profile, profile_result in results.get("profiles", {}).items():
        robots = profile_result.get("robots", {})
        completed = sum(
            row.get("status") in {"completed", "skipped"}
            for row in robots.values()
        )
        macro = profile_result.get("macro_average", {})
        pooled = profile_result.get("pooled_episode_metrics", {})
        lines.append(
            "| "
            + " | ".join(
                [
                    profile,
                    f"{completed}/{len(robots)}",
                    _pct(macro.get("success_once_rate")),
                    _pct(pooled.get("success_once_rate")),
                    _mean_std(pooled, "avg_collision_per_episode", "std_collision_per_episode"),
                    _mean_std(pooled, "avg_max_reward_per_episode", "std_max_reward_per_episode"),
                ]
            )
            + " |"
        )
    common_keys = {
        "success_once_rate",
        "success_at_end_rate",
        "avg_collision_per_episode",
        "std_collision_per_episode",
        "avg_max_reward_per_episode",
        "std_max_reward_per_episode",
        "fail_rate_all",
        "fail_rate_among_non_success",
        "avg_first_success_step",
        "std_first_success_step",
    }
    for profile, profile_result in results.get("profiles", {}).items():
        lines.extend(
            [
                "",
                f"## {profile}",
                "",
                "| Robot | Status | Success once | Success at end | Collisions/ep | Max reward/ep | Fail@All | Fail@NeverSuccess | First success step |",
                "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        custom_keys: set[str] = set()
        for robot, row in profile_result.get("robots", {}).items():
            metrics = row.get("metrics", {})
            custom_keys.update(set(metrics) - common_keys)
            lines.append(
                "| "
                + " | ".join(
                    [
                        robot,
                        str(row.get("status", "pending")),
                        _pct(metrics.get("success_once_rate")),
                        _pct(metrics.get("success_at_end_rate")),
                        _mean_std(metrics, "avg_collision_per_episode", "std_collision_per_episode"),
                        _mean_std(metrics, "avg_max_reward_per_episode", "std_max_reward_per_episode"),
                        _pct(metrics.get("fail_rate_all")),
                        _pct(metrics.get("fail_rate_among_non_success")),
                        _mean_std(metrics, "avg_first_success_step", "std_first_success_step"),
                    ]
                )
                + " |"
            )
        average = profile_result.get("macro_average", {})
        lines.append(
            "| **Macro average** | — | "
            + " | ".join(
                [
                    _pct(average.get("success_once_rate")),
                    _pct(average.get("success_at_end_rate")),
                    _mean_std(average, "avg_collision_per_episode", "__none__"),
                    _mean_std(average, "avg_max_reward_per_episode", "__none__"),
                    _pct(average.get("fail_rate_all")),
                    _pct(average.get("fail_rate_among_non_success")),
                    _mean_std(average, "avg_first_success_step", "__none__"),
                ]
            )
            + " |"
        )
        pooled = profile_result.get("pooled_episode_metrics", {})
        if pooled:
            lines.append(
                "| **All episodes (pooled)** | — | "
                + " | ".join(
                    [
                        _pct(pooled.get("success_once_rate")),
                        _pct(pooled.get("success_at_end_rate")),
                        _mean_std(pooled, "avg_collision_per_episode", "std_collision_per_episode"),
                        _mean_std(pooled, "avg_max_reward_per_episode", "std_max_reward_per_episode"),
                        _pct(pooled.get("fail_rate_all")),
                        _pct(pooled.get("fail_rate_among_non_success")),
                        _mean_std(pooled, "avg_first_success_step", "std_first_success_step"),
                    ]
                )
                + " |"
            )
        custom_keys = {
            key
            for key in custom_keys
            if key.startswith("jm2d_") or key.startswith("inference_")
        }
        if custom_keys:
            lines.extend(["", "Additional method-specific metrics:", ""])
            lines.append("| Robot | " + " | ".join(sorted(custom_keys)) + " |")
            lines.append("| --- | " + " | ".join("---:" for _ in custom_keys) + " |")
            for robot, row in profile_result.get("robots", {}).items():
                metrics = row.get("metrics", {})
                values = [
                    "N/A" if metrics.get(key) is None else f"{float(metrics[key]):.6g}"
                    for key in sorted(custom_keys)
                ]
                lines.append("| " + " | ".join([robot, *values]) + " |")
    return "\n".join(lines) + "\n"


def write_workflow_results(
    output_dir: str | os.PathLike[str], jobs: Iterable[Mapping[str, Any]]
) -> dict[str, Any]:
    """Atomically refresh JSON and Markdown summaries."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    results = aggregate_workflow_results(output_dir, jobs)
    json_tmp = output_dir / ".results.json.tmp"
    md_tmp = output_dir / ".results.md.tmp"
    json_tmp.write_text(json.dumps(results, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    md_tmp.write_text(render_results_markdown(results), encoding="utf-8")
    json_tmp.replace(output_dir / "results.json")
    md_tmp.replace(output_dir / "results.md")
    return results


__all__ = [
    "aggregate_workflow_results",
    "evaluation_subdir",
    "load_job_metrics",
    "macro_average",
    "parse_legacy_results",
    "pooled_episode_metrics",
    "render_results_markdown",
    "validate_output_name",
    "workflow_result_dir",
    "write_workflow_results",
]
