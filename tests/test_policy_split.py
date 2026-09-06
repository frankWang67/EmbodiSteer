"""Contracts separating the paper policy from joint-space comparisons."""

from copy import deepcopy
from pathlib import Path

import pytest
import torch
from hydra.utils import get_class

import embodisteer.policies as public_policies
from embodisteer.policies import (
    DiffusionUnetTimmPolicyEmbodiSteer,
    DiffusionUnetTimmPolicyJointSpace,
    EmbodiSteerJointPolicy,
)
from embodisteer.policies.baselines import DiffusionUnetTimmPolicyBaseline
from embodisteer.policies.ee2joint import EmbodiSteerJointPolicy as OldModuleAlias
from embodisteer.policies.jm2d import DiffusionUnetTimmPolicyJM2D
from embodisteer.runtime_config import joint_policy_overrides, load_policy_config, policy_target
from scripts_maniskill.eval_trajectory_continuity import ContinuityPolicyProxy
from test_policy_config_integration import (
    _base_constructor_args,
    _make_cpu_joint_policy,
    _set_identity_normalizer,
)


ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("cls", [
    DiffusionUnetTimmPolicyBaseline,
    DiffusionUnetTimmPolicyJM2D,
])
def test_public_baseline_exports_preserve_class_identity(cls):
    assert getattr(public_policies, cls.__name__) is cls
    assert cls.__name__ in public_policies.__all__


def test_paper_policy_is_a_sibling_not_a_gd_subclass():
    assert EmbodiSteerJointPolicy is DiffusionUnetTimmPolicyEmbodiSteer
    assert OldModuleAlias is EmbodiSteerJointPolicy
    assert not issubclass(EmbodiSteerJointPolicy, DiffusionUnetTimmPolicyJointSpace)
    assert not hasattr(EmbodiSteerJointPolicy, "_compute_collision_grad")
    assert hasattr(DiffusionUnetTimmPolicyJointSpace, "_compute_collision_grad")
    assert not hasattr(DiffusionUnetTimmPolicyJointSpace, "_apply_cbf_guidance")


@pytest.mark.parametrize("cls,guidance", [
    (DiffusionUnetTimmPolicyEmbodiSteer, "gd"),
    (DiffusionUnetTimmPolicyEmbodiSteer, ""),
    (DiffusionUnetTimmPolicyJointSpace, "cbf"),
])
def test_wrong_method_class_is_rejected_before_allocating_robot_resources(cls, guidance):
    with pytest.raises(ValueError, match="DiffusionUnetTimmPolicy"):
        cls(guidance_method=guidance)


@pytest.mark.parametrize("profile,cls,guidance", [
    ("embodisteer", DiffusionUnetTimmPolicyEmbodiSteer, "cbf"),
    ("joint_gd", DiffusionUnetTimmPolicyJointSpace, "gd"),
    ("joint_no_guidance", DiffusionUnetTimmPolicyJointSpace, ""),
    ("post_hoc_cbf", DiffusionUnetTimmPolicyBaseline, ""),
    ("batch_sampling", DiffusionUnetTimmPolicyBaseline, ""),
    ("jm2d", DiffusionUnetTimmPolicyJM2D, ""),
])
def test_profile_selects_and_constructs_the_expected_class(profile, cls, guidance):
    values = load_policy_config(ROOT / f"configs/policy/{profile}.yaml")
    target = get_class(policy_target(values))
    assert target is cls
    kwargs = joint_policy_overrides(values)
    if values["baseline_method"]:
        kwargs["baseline_method"] = values["baseline_method"]
    policy = target(
        **_base_constructor_args(), robot_cfg_name="unused.yml", arm_dof=6, **kwargs,
    )
    assert policy.guidance_method == guidance
    assert policy._kin_model is None


def test_checkpoint_state_dicts_load_strictly_across_sibling_policies():
    reference = DiffusionUnetTimmPolicyJointSpace(
        **_base_constructor_args(), robot_cfg_name="unused.yml", arm_dof=6,
    )
    _set_identity_normalizer(reference)
    state = deepcopy(reference.state_dict())
    assert any(key.startswith("model.") for key in state)
    assert any(key.startswith("normalizer.") for key in state)
    for cls in (
        DiffusionUnetTimmPolicyEmbodiSteer,
        DiffusionUnetTimmPolicyJointSpace,
        DiffusionUnetTimmPolicyBaseline,
        DiffusionUnetTimmPolicyJM2D,
    ):
        policy = cls(**_base_constructor_args(), robot_cfg_name="unused.yml", arm_dof=6)
        policy.load_state_dict(state, strict=True)
        actual = policy.state_dict()
        assert actual.keys() == state.keys()
        for name in state:
            torch.testing.assert_close(actual[name], state[name], rtol=0, atol=0)


@pytest.mark.parametrize("state_change,recompute", [(0.499, False), (0.5, True)])
def test_cbf_correction_preserves_jacobian_threshold_mask_and_clip(state_change, recompute):
    policy = _make_cpu_joint_policy("cbf")
    policy._world_collision = object()
    policy.guidance_grad_clip = 0.05
    q_ref = torch.zeros(1, 4, 6)
    q_new = q_ref + state_change
    jac_ref = torch.ones(4, 6, 6)
    calls = []
    policy._jacobian = lambda q: calls.append(q.clone()) or jac_ref * 2
    policy._guidance_scale_at = lambda *args: torch.tensor(0.7)

    def qp(jac, grad_h, h, gamma):
        torch.testing.assert_close(jac, (jac_ref * (2 if recompute else 1)).reshape(1, 4, 6, 6))
        assert gamma == torch.tensor(0.7)
        return torch.full_like(grad_h, 0.2), None, None, None

    policy._cbf_qp_fn = qp
    mask = torch.tensor([[True, False, False, False]])
    corrected = policy._apply_cbf_guidance(q_new, jac_ref, q_ref, mask, 1, 2, 0)
    assert bool(calls) is recompute
    torch.testing.assert_close(corrected[:, 0], q_new[:, 0], rtol=0, atol=0)
    # CBF gamma goes into the QP, not a second multiplication after clipping.
    torch.testing.assert_close(corrected[:, 1:], q_new[:, 1:] + 0.05, rtol=0, atol=0)


@pytest.mark.parametrize("guidance", ["", "gd", "cbf"])
@pytest.mark.parametrize("conditioned", [False, True])
def test_sampler_preserves_inpainting_and_explicit_generator(guidance, conditioned):
    policy = _make_cpu_joint_policy(guidance)
    policy._ik_from_absolute = lambda pos, rot, seed: torch.full_like(seed, 0.25)
    data = torch.zeros(2, 4, 10)
    data[..., 3:9] = torch.tensor([1., 0., 0., 0., 1., 0.])
    mask = torch.zeros_like(data, dtype=torch.bool)
    if conditioned:
        mask[:, 0] = True
    outputs = []
    for _ in range(2):
        generator = torch.Generator().manual_seed(17)
        result = policy.conditional_sample(
            data, mask, generator=generator,
            chunk_start_pose=torch.zeros(2, 6),
            current_joint_angles=torch.zeros(2, 6),
        )
        outputs.append((result.clone(), policy._last_joint_traj.clone(), generator.get_state()))
        torch.testing.assert_close(result[mask], data[mask], rtol=0, atol=0)
        if conditioned:
            torch.testing.assert_close(policy._last_joint_traj[:, 0, :6], torch.full((2, 6), 0.25))
    for first, second in zip(*outputs):
        torch.testing.assert_close(first, second, rtol=0, atol=0)


def test_continuity_proxy_still_compares_exact_noise_and_restores_cbf_state():
    policy = _make_cpu_joint_policy("cbf")
    kwargs = dict(
        chunk_start_pose=torch.zeros(1, 6),
        current_joint_angles=torch.zeros(1, 7),
    )
    observations = {"state": torch.zeros(1, 1, 2)}
    torch.manual_seed(19)
    expected = policy.predict_action(observations, **kwargs)
    expected_rng = torch.get_rng_state().clone()
    proxy = ContinuityPolicyProxy(policy, torch)
    torch.manual_seed(19)
    actual = proxy.predict_action(observations, **kwargs)
    assert policy.guidance_method == "cbf"
    assert len(proxy.records) == 1
    for name in expected:
        torch.testing.assert_close(actual[name], expected[name], rtol=0, atol=0)
    torch.testing.assert_close(policy._last_joint_traj, expected["joint_action_pred"], rtol=0, atol=0)
    assert torch.equal(torch.get_rng_state(), expected_rng)
    record = proxy.records[0]
    assert (record["q_cbf"] != record["q_no_cbf"]).any()


@pytest.mark.parametrize("method,expected_suffix,guidance", [
    ("vanilla", "ee_space.EmbodiSteerEESpacePolicy", None),
    ("joint_space_no_guidance", "ee2joint.DiffusionUnetTimmPolicyJointSpace", ""),
    ("embodisteer", "embodisteer.DiffusionUnetTimmPolicyEmbodiSteer", "cbf"),
    ("batch_sampling_4", "baselines.DiffusionUnetTimmPolicyBaseline", ""),
    ("jm2d_n4_i16", "jm2d.DiffusionUnetTimmPolicyJM2D", ""),
])
def test_benchmark_method_labels_use_shared_routing(monkeypatch, method, expected_suffix, guidance):
    from omegaconf import OmegaConf
    from scripts_maniskill import benchmark_jm2d_speed as benchmark

    monkeypatch.setattr(benchmark, "infer_robot_kinematic_args", lambda name: (None, "eef", 6))
    values = load_policy_config(ROOT / "configs/policy/joint_gd.yaml")
    before = deepcopy(values)
    base_cfg = OmegaConf.create({"policy": {"_target_": "old.checkpoint.Policy"}})
    cfg = benchmark.configure_method(base_cfg, method, values)
    assert cfg.policy._target_.endswith(expected_suffix)
    if guidance is not None:
        assert cfg.policy.guidance_method == guidance
    else:
        assert cfg.policy.use_ee_guidance is False
    assert cfg.policy.num_inference_steps == values["num_inference_steps"]
    if method != "vanilla":
        assert cfg.policy.guidance_scale == values["guidance_scale"]
    assert base_cfg.policy._target_ == "old.checkpoint.Policy"
    assert values == before
