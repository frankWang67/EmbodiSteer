from copy import deepcopy
from pathlib import Path
import tempfile

from embodisteer.runtime_config import (
    DEFAULT_POLICY_CONFIG,
    PolicyConfigError,
    joint_policy_overrides,
    load_policy_config,
    validate_policy_config,
)


ROOT = Path(__file__).resolve().parents[1]


def test_checked_in_policy_profiles_load():
    embodisteer = load_policy_config(ROOT / "configs/policy/embodisteer.yaml")
    ee = load_policy_config(ROOT / "configs/policy/ee.yaml")
    assert (embodisteer["inference_space"], embodisteer["guidance"]) == (
        "joint",
        "cbf",
    )
    assert (ee["inference_space"], ee["guidance"]) == ("ee", "")


def test_joint_overrides_include_configured_guidance_and_ik_values():
    config = load_policy_config(ROOT / "configs/policy/embodisteer.yaml")
    overrides = joint_policy_overrides(config)
    assert overrides["guidance_method"] == "cbf"
    assert overrides["guidance_use_schedule"] is True
    assert overrides["guidance_steps_per_denoise"] == 1
    assert overrides["jacobian_damping"] == 0.001
    assert overrides["noise_init_mode"] == "jacobian_projected"


def test_invalid_method_combination_is_rejected():
    values = deepcopy(DEFAULT_POLICY_CONFIG)
    values["inference_space"] = "ee"
    values["guidance"] = "cbf"
    try:
        validate_policy_config(values)
    except PolicyConfigError as exc:
        assert "does not support CBF" in str(exc)
    else:
        raise AssertionError("invalid EE/CBF combination was accepted")


def test_unknown_yaml_field_is_rejected():
    text = """\
schema_version: 1
policy:
  inference_space: joint
  guidance:
    method: cbf
    saftey_margin: 0.05
"""
    with tempfile.TemporaryDirectory() as tmp_dir:
        path = Path(tmp_dir) / "invalid.yaml"
        path.write_text(text, encoding="utf-8")
        try:
            load_policy_config(path)
        except PolicyConfigError as exc:
            assert "saftey_margin" in str(exc)
        else:
            raise AssertionError("unknown policy config field was accepted")
