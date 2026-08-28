from copy import deepcopy

import torch
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler

from diffusion_policy.model.common.normalizer import SingleFieldLinearNormalizer
from embodisteer.policies.ee2joint import EmbodiSteerJointPolicy
from embodisteer.policies.ee_space import EmbodiSteerEESpacePolicy
from embodisteer.runtime_config import (
    DEFAULT_POLICY_CONFIG,
    ee_policy_overrides,
    joint_policy_overrides,
)


class DummyObservationEncoder(torch.nn.Module):
    def output_shape(self):
        return (4,)

    def forward(self, observations):
        batch_size = next(iter(observations.values())).shape[0]
        return torch.zeros(batch_size, 4, dtype=torch.float32)


class ZeroNoiseModel(torch.nn.Module):
    def forward(self, sample, timestep, local_cond=None, global_cond=None):
        return torch.zeros_like(sample)


def _base_constructor_args():
    return {
        "shape_meta": {
            "action": {"shape": [10], "horizon": 4},
            "obs": {"state": {"shape": [2], "horizon": 1}},
        },
        "noise_scheduler": DDPMScheduler(
            num_train_timesteps=4,
            beta_start=0.0001,
            beta_end=0.02,
            beta_schedule="linear",
            prediction_type="epsilon",
        ),
        "obs_encoder": DummyObservationEncoder(),
        "diffusion_step_embed_dim": 16,
        "down_dims": (8, 16),
        "kernel_size": 3,
        "n_groups": 4,
    }


def _set_identity_normalizer(policy):
    policy.normalizer["state"] = SingleFieldLinearNormalizer.create_identity()
    policy.normalizer["action"] = SingleFieldLinearNormalizer.create_identity()


def test_ee_config_overrides_reach_policy_constructor():
    values = deepcopy(DEFAULT_POLICY_CONFIG)
    values.update(
        {
            "guidance": "gd",
            "num_inference_steps": 3,
            "guidance_scale": 1.4,
            "guidance_safety_margin": 0.025,
            "guidance_grad_clip": 0.06,
            "guidance_schedule_midpoint": 0.45,
            "guidance_schedule_steepness": 18.0,
        }
    )
    policy = EmbodiSteerEESpacePolicy(
        **_base_constructor_args(),
        **ee_policy_overrides(values),
    )

    assert policy.use_ee_guidance is True
    assert policy.num_inference_steps == 3
    assert policy.guidance_scale == 1.4
    assert policy.guidance_safety_margin == 0.025
    assert policy.guidance_grad_clip == 0.06
    assert policy.guidance_schedule_midpoint == 0.45
    assert policy.guidance_schedule_steepness == 18.0
    assert tuple(policy.eef_corner_pts.shape) == (8, 3)


def test_joint_config_overrides_reach_policy_constructor_without_initializing_kinematics():
    values = deepcopy(DEFAULT_POLICY_CONFIG)
    values.update(
        {
            "inference_space": "joint",
            "guidance": "cbf",
            "num_inference_steps": 5,
            "guidance_scale": 1.2,
            "guidance_schedule_midpoint": 0.35,
            "guidance_schedule_steepness": 12.0,
            "jacobian_damping": 0.002,
        }
    )
    policy = EmbodiSteerJointPolicy(
        **_base_constructor_args(),
        robot_cfg_name="unused-in-lazy-constructor-test.yml",
        arm_dof=6,
        **joint_policy_overrides(values),
    )

    assert policy.guidance_method == "cbf"
    assert policy.num_inference_steps == 5
    assert policy.guidance_scale == 1.2
    assert policy.guidance_schedule_midpoint == 0.35
    assert policy.guidance_schedule_steepness == 12.0
    assert policy.jacobian_damping == 0.002
    assert policy._kin_model is None


def test_cpu_synthetic_ee_predict_action_is_finite_reproducible_and_conditioned():
    values = deepcopy(DEFAULT_POLICY_CONFIG)
    values.update({"guidance": "", "num_inference_steps": 2})
    args = _base_constructor_args()
    policy = EmbodiSteerEESpacePolicy(
        **args,
        **ee_policy_overrides(values),
        inpaint_fixed_action_prefix=True,
    )
    policy.model = ZeroNoiseModel()
    _set_identity_normalizer(policy)
    observations = {"state": torch.zeros(1, 1, 2)}
    prefix = torch.full((1, 2, 10), 0.125)

    torch.manual_seed(123)
    first = policy.predict_action(observations, fixed_action_prefix=prefix)["action"]
    torch.manual_seed(123)
    second = policy.predict_action(observations, fixed_action_prefix=prefix)["action"]

    assert tuple(first.shape) == (1, 4, 10)
    assert first.device.type == "cpu"
    assert torch.isfinite(first).all()
    torch.testing.assert_close(first, second)
    torch.testing.assert_close(first[:, :2], prefix)


def _make_cpu_joint_policy(guidance):
    values = deepcopy(DEFAULT_POLICY_CONFIG)
    values.update({"inference_space": "joint", "guidance": guidance, "num_inference_steps": 2})
    constructor = joint_policy_overrides(values)
    constructor.update(init_noise_scale=0.0, noise_init_mode="isotropic")
    policy = EmbodiSteerJointPolicy(
        **_base_constructor_args(),
        robot_cfg_name="unused-in-mock-test.yml",
        arm_dof=6,
        **constructor,
    )
    policy.model = ZeroNoiseModel()
    _set_identity_normalizer(policy)
    policy._robot_dof = 6

    def fake_fk(q_arm):
        batch, horizon, _ = q_arm.shape
        pos = q_arm[..., :3]
        eye = torch.eye(3, dtype=q_arm.dtype, device=q_arm.device)
        rot = eye.view(1, 1, 3, 3).expand(batch, horizon, 3, 3)
        return pos, rot

    def fake_jacobian(q_arm):
        flat = q_arm.reshape(-1, q_arm.shape[-1])
        eye = torch.eye(6, dtype=q_arm.dtype, device=q_arm.device)
        return eye.unsqueeze(0).expand(flat.shape[0], 6, 6)

    policy._ensure_kinematics = lambda device: None
    policy._fk_to_absolute = fake_fk
    policy._jacobian = fake_jacobian
    policy._twist_fn = lambda pos, rot, target_pos, target_rot: torch.cat(
        [target_pos - pos, torch.zeros_like(target_pos)], dim=-1
    )
    policy._joint_traj_to_env_action = lambda q: q
    if guidance:
        policy._build_world_collision = lambda obstacle_info, device, dtype: setattr(
            policy, "_world_collision", object()
        )
        if guidance == "cbf":
            def fake_cbf_linearization(q):
                h = torch.full(q.shape[:2], -0.1, dtype=q.dtype, device=q.device)
                grad = torch.zeros_like(q)
                grad[..., 0] = 1.0
                return h, grad, torch.zeros_like(h)

            policy._compute_cbf_linearization = fake_cbf_linearization
            policy._cbf_qp_fn = lambda jac, grad, h, scale: (
                torch.full_like(grad, 0.01),
                torch.zeros_like(h),
                torch.zeros_like(h),
                torch.ones_like(h, dtype=torch.bool),
            )
        else:
            policy._compute_collision_grad = lambda q: (
                torch.full_like(q, 0.01),
                torch.ones((), dtype=q.dtype, device=q.device),
            )
    return policy


def test_cpu_synthetic_joint_predict_action_covers_no_guidance_cbf_and_gd():
    observations = {"state": torch.zeros(1, 1, 2)}
    chunk_start_pose = torch.zeros(1, 6)
    current_joint_angles = torch.zeros(1, 7)

    for guidance in ("", "cbf", "gd"):
        policy = _make_cpu_joint_policy(guidance)
        torch.manual_seed(321)
        result = policy.predict_action(
            observations,
            chunk_start_pose=chunk_start_pose,
            current_joint_angles=current_joint_angles,
            obstacle_info=[{"synthetic": True}] if guidance else None,
            return_debug=True,
        )

        assert tuple(result["joint_action_pred"].shape) == (1, 4, 7)
        assert tuple(result["action_pred"].shape) == (1, 4, 10)
        assert torch.isfinite(result["joint_action_pred"]).all()
        assert "debug" in result
        assert len(result["debug"]["step_cart_l2"]) == 2
        if guidance:
            assert len(result["debug"]["guidance_loss"]) == 2
