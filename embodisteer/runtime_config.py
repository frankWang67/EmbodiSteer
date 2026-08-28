"""Shared YAML configuration for simulation and real-world inference.

The launchers intentionally keep operational arguments (checkpoint paths,
robot addresses, environment IDs and output directories) on the command line.
Algorithm choices and EmbodiSteer hyperparameters live in one versioned YAML
file and are normalized here so the three launchers cannot silently drift.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import yaml

from .runtime import repository_root


DEFAULT_POLICY_CONFIG: dict[str, Any] = {
    "inference_space": "ee",
    "guidance": "",
    "baseline_method": "",
    "batch_sampling_num": 32,
    "jm2d_num_samples": 16,
    "jm2d_temperature": 0.01,
    "jm2d_eta": 1.0,
    "guidance_scale": 1.0,
    "guidance_use_schedule": True,
    "guidance_safety_margin": 0.05,
    "guidance_activation_distance": 1.0,
    "guidance_grad_clip": 0.1,
    "guidance_loss_power": 2.0,
    "guidance_steps_per_denoise": 1,
    "guidance_cbf_lambda": 0.01,
    "guidance_cbf_reverse_task_threshold": None,
    "guidance_sdf_agg": "topk",
    "guidance_sdf_softmax_temp": 20.0,
    "guidance_sdf_topk": 4,
    "guidance_task_pos_weight": 1.0,
    "guidance_task_rot_weight": 0.1,
    "guidance_use_clean_sample": False,
    "guidance_apply_last_step_only": False,
    "guidance_reuse_jacobian": True,
    "jacobian_damping": 0.001,
    "ik_refine_each_step": False,
    "ik_refine_last_step": False,
    "ik_position_threshold": 5e-4,
    "ik_rotation_threshold": 5e-3,
    "ik_num_seeds": 1,
    "init_noise_scale": 0.2,
    "noise_init_mode": "jacobian_projected",
    "jac_noise_alpha": 0.1,
    "max_dq_per_step": 0.5,
    "cartesian_delta_mode": "geometric",
}

JOINT_POLICY_KEYS = (
    "jacobian_damping",
    "ik_refine_each_step",
    "ik_refine_last_step",
    "ik_position_threshold",
    "ik_rotation_threshold",
    "ik_num_seeds",
    "init_noise_scale",
    "noise_init_mode",
    "jac_noise_alpha",
    "max_dq_per_step",
    "cartesian_delta_mode",
    "guidance_scale",
    "guidance_use_schedule",
    "guidance_safety_margin",
    "guidance_activation_distance",
    "guidance_grad_clip",
    "guidance_loss_power",
    "guidance_steps_per_denoise",
    "guidance_use_clean_sample",
    "guidance_apply_last_step_only",
    "guidance_cbf_lambda",
    "guidance_cbf_reverse_task_threshold",
    "guidance_sdf_agg",
    "guidance_sdf_softmax_temp",
    "guidance_sdf_topk",
    "guidance_task_pos_weight",
    "guidance_task_rot_weight",
    "guidance_reuse_jacobian",
)

IK_POLICY_KEYS = {
    "jacobian_damping",
    "ik_refine_each_step",
    "ik_refine_last_step",
    "ik_position_threshold",
    "ik_rotation_threshold",
    "ik_num_seeds",
    "init_noise_scale",
    "noise_init_mode",
    "jac_noise_alpha",
    "max_dq_per_step",
    "cartesian_delta_mode",
}


class PolicyConfigError(ValueError):
    """Raised when an inference config is malformed or inconsistent."""


def _path_from_user(path: str | Path) -> Path:
    candidate = Path(path).expanduser()
    if not candidate.is_absolute():
        candidate = repository_root() / candidate
    return candidate.resolve()


def _nested_mapping(value: Any, name: str) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise PolicyConfigError(f"policy.{name} must be a mapping")
    return dict(value)


def _copy_defaults() -> dict[str, Any]:
    return dict(DEFAULT_POLICY_CONFIG)


def _reject_unknown(mapping: Mapping[str, Any], allowed: set[str], location: str) -> None:
    unknown = sorted(set(mapping) - allowed)
    if unknown:
        raise PolicyConfigError(
            f"{location} contains unknown fields: {', '.join(unknown)}"
        )


def load_policy_config(path: str | Path) -> dict[str, Any]:
    """Load and normalize the canonical nested policy YAML.

    The returned mapping uses the flat names expected by the existing policy
    construction code. Config files use the nested ``guidance``, ``baseline``, ``jm2d`` and
    ``ik`` sections documented in ``configs/policy/embodisteer.yaml``.
    """

    config_path = _path_from_user(path)
    try:
        with config_path.open("r", encoding="utf-8") as handle:
            document = yaml.safe_load(handle)
    except OSError as exc:
        raise PolicyConfigError(f"Cannot read policy config {config_path}: {exc}") from exc
    except yaml.YAMLError as exc:
        raise PolicyConfigError(f"Invalid YAML in policy config {config_path}: {exc}") from exc

    if not isinstance(document, Mapping):
        raise PolicyConfigError("policy config must be a YAML mapping")
    if document.get("schema_version") != 1:
        raise PolicyConfigError("policy config schema_version must be 1")
    _reject_unknown(document, {"schema_version", "policy"}, "policy config")
    policy = document.get("policy")
    if not isinstance(policy, Mapping):
        raise PolicyConfigError("policy config's 'policy' field must be a mapping")
    policy = dict(policy)

    values = _copy_defaults()
    guidance_value = policy.get("guidance")
    guidance = (
        {"method": guidance_value}
        if isinstance(guidance_value, str)
        else _nested_mapping(guidance_value, "guidance")
    )
    baseline_value = policy.get("baseline")
    baseline = (
        {"method": baseline_value}
        if isinstance(baseline_value, str)
        else _nested_mapping(baseline_value, "baseline")
    )
    jm2d = _nested_mapping(baseline.get("jm2d", policy.get("jm2d")), "baseline.jm2d")
    ik = _nested_mapping(policy.get("ik"), "ik")
    _reject_unknown(
        policy,
        set(DEFAULT_POLICY_CONFIG) | {"guidance", "baseline", "jm2d", "ik"},
        "policy",
    )
    _reject_unknown(
        guidance,
        {"method"}
        | {key.removeprefix("guidance_") for key in DEFAULT_POLICY_CONFIG if key.startswith("guidance_")}
        | {key for key in DEFAULT_POLICY_CONFIG if key.startswith("guidance_")},
        "policy.guidance",
    )
    _reject_unknown(
        baseline,
        {"method", "jm2d", "batch_sampling_num"},
        "policy.baseline",
    )
    _reject_unknown(
        jm2d,
        {key.removeprefix("jm2d_") for key in DEFAULT_POLICY_CONFIG if key.startswith("jm2d_")}
        | {key for key in DEFAULT_POLICY_CONFIG if key.startswith("jm2d_")},
        "policy.baseline.jm2d",
    )
    ik_short_names = {
        "refine_each_step",
        "refine_last_step",
        "position_threshold",
        "rotation_threshold",
        "num_seeds",
    }
    _reject_unknown(
        ik,
        IK_POLICY_KEYS | ik_short_names,
        "policy.ik",
    )

    if "inference_space" in policy:
        values["inference_space"] = policy["inference_space"]
    if "method" in guidance:
        values["guidance"] = guidance["method"]
    if "method" in baseline:
        values["baseline_method"] = baseline["method"]
    for key, value in guidance.items():
        if key == "method":
            continue
        flat_key = key if key.startswith("guidance_") else f"guidance_{key}"
        if flat_key in values:
            values[flat_key] = value
    for key, value in baseline.items():
        if key in ("method", "jm2d"):
            continue
        if key in values:
            values[key] = value
    for key, value in jm2d.items():
        flat_key = f"jm2d_{key}"
        if flat_key in values:
            values[flat_key] = value
    for key, value in ik.items():
        flat_key = key if key in values else f"ik_{key}"
        if flat_key in values:
            values[flat_key] = value

    # Direct constructor-style names inside ``policy`` are accepted last for
    # generated experiment configs; the checked-in profiles use nested fields.
    nested_keys = {"guidance", "baseline", "jm2d", "ik"}
    for key, value in policy.items():
        if key in nested_keys:
            continue
        if key in values:
            values[key] = value

    values["inference_space"] = str(values["inference_space"]).lower().strip()
    values["guidance"] = str(values["guidance"]).lower().strip()
    values["baseline_method"] = str(values["baseline_method"]).lower().strip()
    values["guidance_sdf_agg"] = str(values["guidance_sdf_agg"]).lower().strip()
    validate_policy_config(values)
    values["config_path"] = str(config_path)
    return values


def validate_policy_config(values: Mapping[str, Any]) -> None:
    """Validate method combinations and safety-critical numeric settings."""

    inference_space = str(values.get("inference_space", "")).lower().strip()
    guidance = str(values.get("guidance", "")).lower().strip()
    baseline = str(values.get("baseline_method", "")).lower().strip()
    if inference_space not in ("ee", "joint"):
        raise PolicyConfigError("policy.inference_space must be 'ee' or 'joint'")
    if guidance not in ("", "cbf", "gd"):
        raise PolicyConfigError("policy.guidance.method must be '', 'cbf' or 'gd'")
    if baseline not in ("", "post_hoc_cbf", "batch_sampling", "jm2d"):
        raise PolicyConfigError(
            "policy.baseline.method must be '', 'post_hoc_cbf', "
            "'batch_sampling' or 'jm2d'"
        )
    if baseline and guidance:
        raise PolicyConfigError(
            "policy.guidance.method must be empty when a baseline is selected"
        )
    if inference_space == "ee" and guidance == "cbf":
        raise PolicyConfigError("EE-space inference does not support CBF guidance")
    reverse = values.get("guidance_cbf_reverse_task_threshold")
    if reverse is not None:
        if float(reverse) <= 0:
            raise PolicyConfigError("guidance.cbf_reverse_task_threshold must be positive")
        if baseline or inference_space != "joint" or guidance != "cbf":
            raise PolicyConfigError(
                "reverse CBF requires joint inference with CBF and no baseline"
            )
    if int(values["batch_sampling_num"]) < 2:
        raise PolicyConfigError("baseline.batch_sampling_num must be at least 2")
    if int(values["jm2d_num_samples"]) < 1:
        raise PolicyConfigError("baseline.jm2d.num_samples must be at least 1")
    if float(values["jm2d_temperature"]) <= 0:
        raise PolicyConfigError("baseline.jm2d.temperature must be positive")
    if float(values["jm2d_eta"]) < 0:
        raise PolicyConfigError("baseline.jm2d.eta must be non-negative")
    if int(values["guidance_steps_per_denoise"]) < 1:
        raise PolicyConfigError("guidance.steps_per_denoise must be at least 1")
    if float(values["guidance_loss_power"]) <= 0:
        raise PolicyConfigError("guidance.loss_power must be positive")
    if str(values["guidance_sdf_agg"]).lower() not in ("max", "topk", "softmax"):
        raise PolicyConfigError(
            "guidance.sdf_agg must be 'max', 'topk' or 'softmax'"
        )
    if int(values["guidance_sdf_topk"]) < 1:
        raise PolicyConfigError("guidance.sdf_topk must be at least 1")
    if float(values["guidance_sdf_softmax_temp"]) <= 0:
        raise PolicyConfigError("guidance.sdf_softmax_temp must be positive")
    if float(values["guidance_cbf_lambda"]) <= 0:
        raise PolicyConfigError("guidance.cbf_lambda must be positive")
    if float(values["guidance_activation_distance"]) <= 0:
        raise PolicyConfigError("guidance.activation_distance must be positive")
    if float(values["guidance_safety_margin"]) < 0:
        raise PolicyConfigError("guidance.safety_margin must be non-negative")
    if float(values["guidance_grad_clip"]) < 0:
        raise PolicyConfigError("guidance.grad_clip must be non-negative")
    if float(values["guidance_scale"]) < 0:
        raise PolicyConfigError("guidance.scale must be non-negative")
    if float(values["guidance_task_pos_weight"]) <= 0:
        raise PolicyConfigError("guidance.task_pos_weight must be positive")
    if float(values["guidance_task_rot_weight"]) < 0:
        raise PolicyConfigError("guidance.task_rot_weight must be non-negative")
    if float(values["jacobian_damping"]) <= 0:
        raise PolicyConfigError("ik.jacobian_damping must be positive")
    if float(values["max_dq_per_step"]) < 0:
        raise PolicyConfigError("ik.max_dq_per_step must be non-negative")
    if int(values["ik_num_seeds"]) < 1:
        raise PolicyConfigError("ik.num_seeds must be at least 1")
    if float(values["ik_position_threshold"]) <= 0:
        raise PolicyConfigError("ik.position_threshold must be positive")
    if float(values["ik_rotation_threshold"]) <= 0:
        raise PolicyConfigError("ik.rotation_threshold must be positive")
    if float(values["init_noise_scale"]) < 0:
        raise PolicyConfigError("ik.init_noise_scale must be non-negative")
    if float(values["jac_noise_alpha"]) < 0:
        raise PolicyConfigError("ik.jac_noise_alpha must be non-negative")
    for key in (
        "guidance_use_schedule",
        "guidance_use_clean_sample",
        "guidance_apply_last_step_only",
        "guidance_reuse_jacobian",
        "ik_refine_each_step",
        "ik_refine_last_step",
    ):
        if not isinstance(values[key], bool):
            raise PolicyConfigError(f"{key} must be a YAML boolean")
    if str(values["noise_init_mode"]) not in (
        "isotropic",
        "jacobian_projected",
        "jacobian_diagonal",
    ):
        raise PolicyConfigError(
            "ik.noise_init_mode must be isotropic, jacobian_projected or jacobian_diagonal"
        )
    if str(values["cartesian_delta_mode"]) not in ("geometric", "se3_delta"):
        raise PolicyConfigError("ik.cartesian_delta_mode must be geometric or se3_delta")


def joint_policy_overrides(values: Mapping[str, Any]) -> dict[str, Any]:
    """Return constructor overrides shared by joint and baseline policies."""

    overrides = {key: values[key] for key in JOINT_POLICY_KEYS}
    overrides["guidance_method"] = values["guidance"]
    return overrides


__all__ = [
    "DEFAULT_POLICY_CONFIG",
    "PolicyConfigError",
    "joint_policy_overrides",
    "load_policy_config",
    "validate_policy_config",
]
