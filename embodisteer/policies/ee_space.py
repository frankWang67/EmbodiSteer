from typing import Dict
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from einops import rearrange, reduce
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler

from diffusion_policy.model.common.normalizer import LinearNormalizer
from diffusion_policy.policy.base_image_policy import BaseImagePolicy
from diffusion_policy.model.diffusion.conditional_unet1d import ConditionalUnet1D
from diffusion_policy.model.diffusion.mask_generator import LowdimMaskGenerator
from diffusion_policy.model.vision.timm_obs_encoder import TimmObsEncoder
from diffusion_policy.common.pytorch_util import dict_apply
from embodisteer.guidance.diffusion import (
    get_pred_x0,
    rel_action_obstacle_loss,
    get_guidance_strength,
)


class DiffusionUnetTimmPolicyEESpace(BaseImagePolicy):
    """
    End-effector space diffusion policy with optional Cartesian guidance.

    When use_ee_guidance=False, behaves identically to the base DiffusionUnetTimmPolicy.
    When use_ee_guidance=True, applies gradient-based guidance using EEF corner points
    and obstacle information during denoising (same as DiffusionUnetTimmPolicyWithGuidance).
    """

    eef_corner_pts = torch.tensor([
        [ 0.01,  0.043, 0.01],
        [ 0.01, -0.043, 0.01],
        [-0.01,  0.043, 0.01],
        [-0.01, -0.043, 0.01],
        [ 0.01,  0.043, -0.03],
        [ 0.01, -0.043, -0.03],
        [-0.01,  0.043, -0.03],
        [-0.01, -0.043, -0.03],
    ])

    def __init__(self,
            shape_meta: dict,
            noise_scheduler: DDPMScheduler,
            obs_encoder: TimmObsEncoder,
            num_inference_steps=None,
            obs_as_global_cond=True,
            diffusion_step_embed_dim=256,
            down_dims=(256,512,1024),
            kernel_size=5,
            n_groups=8,
            cond_predict_scale=True,
            input_pertub=0.1,
            inpaint_fixed_action_prefix=False,
            train_diffusion_n_samples=1,
            use_ee_guidance=False,
            # parameters passed to step
            **kwargs
        ):
        super().__init__()

        # parse shapes
        action_shape = shape_meta['action']['shape']
        assert len(action_shape) == 1
        action_dim = action_shape[0]
        action_horizon = shape_meta['action']['horizon']
        obs_feature_dim = np.prod(obs_encoder.output_shape())

        # create diffusion model
        assert obs_as_global_cond
        input_dim = action_dim
        global_cond_dim = obs_feature_dim

        model = ConditionalUnet1D(
            input_dim=input_dim,
            local_cond_dim=None,
            global_cond_dim=global_cond_dim,
            diffusion_step_embed_dim=diffusion_step_embed_dim,
            down_dims=down_dims,
            kernel_size=kernel_size,
            n_groups=n_groups,
            cond_predict_scale=cond_predict_scale
        )

        self.obs_encoder = obs_encoder
        self.model = model
        self.noise_scheduler = noise_scheduler
        self.normalizer = LinearNormalizer()
        self.obs_feature_dim = obs_feature_dim
        self.action_dim = action_dim
        self.action_horizon = action_horizon
        self.obs_as_global_cond = obs_as_global_cond
        self.input_pertub = input_pertub
        self.inpaint_fixed_action_prefix = inpaint_fixed_action_prefix
        self.train_diffusion_n_samples = int(train_diffusion_n_samples)
        self.use_ee_guidance = bool(use_ee_guidance)
        self.kwargs = kwargs

        if num_inference_steps is None:
            num_inference_steps = noise_scheduler.config.num_train_timesteps
        self.num_inference_steps = num_inference_steps

    def to(self, device):
        super().to(device)
        self.eef_corner_pts = self.eef_corner_pts.to(device)

    # ========= inference  ============
    def conditional_sample(self,
            condition_data,
            condition_mask,
            local_cond=None,
            global_cond=None,
            generator=None,
            chunk_start_pose=None,
            obstacle_info=[],
            current_joint_angles=None,
            # keyword arguments to scheduler.step
            **kwargs
        ):
        if self.use_ee_guidance:
            assert chunk_start_pose is not None, "chunk_start_pose is required for guided diffusion"

        model = self.model
        scheduler = self.noise_scheduler

        trajectory = torch.randn(
            size=condition_data.shape,
            dtype=condition_data.dtype,
            device=condition_data.device,
            generator=generator)

        scheduler.set_timesteps(self.num_inference_steps)

        for t in scheduler.timesteps:
            # 1. apply conditioning
            trajectory[condition_mask] = condition_data[condition_mask]

            # 2. predict model output
            model_output = model(trajectory, t,
                local_cond=local_cond, global_cond=global_cond)

            # 3. optional EE guidance
            if self.use_ee_guidance and chunk_start_pose is not None and len(obstacle_info) > 0:
                alpha_prod_t = self.noise_scheduler.alphas_cumprod[t]

                with torch.enable_grad():
                    x_in = trajectory.detach().clone().requires_grad_(True)
                    pred_action_normalized = get_pred_x0(model_output, x_in, alpha_prod_t)
                    x_physical = self.normalizer['action'].unnormalize(pred_action_normalized)
                    x_physical = x_physical[:, :, :9]

                    loss = rel_action_obstacle_loss(
                        action_pred=x_physical,
                        current_state=chunk_start_pose,
                        robot_corners=self.eef_corner_pts,
                        obstacles=obstacle_info,
                    )

                    grad = torch.autograd.grad(loss, x_in, allow_unused=True)[0]
                    if grad is None:
                        grad = torch.zeros_like(x_in)
                    grad = torch.clamp(grad, -0.1, 0.1)

                gamma = get_guidance_strength(t, self.num_inference_steps)
            else:
                grad = None
                gamma = None

            # 4. compute previous image: x_t -> x_t-1
            trajectory = scheduler.step(
                model_output, t, trajectory,
                generator=generator,
                **kwargs
                ).prev_sample

            # 5. apply guidance
            if grad is not None:
                trajectory = trajectory - gamma * grad

        # finally make sure conditioning is enforced
        trajectory[condition_mask] = condition_data[condition_mask]

        return trajectory

    def predict_action(self,
        obs_dict: Dict[str, torch.Tensor],
        fixed_action_prefix: torch.Tensor=None,
        env_batched=False,
        chunk_start_pose: torch.Tensor=None,
        obstacle_info=[],
        current_joint_angles=None,
    ) -> Dict[str, torch.Tensor]:
        assert 'past_action' not in obs_dict
        nobs = self.normalizer.normalize(obs_dict)
        B = next(iter(nobs.values())).shape[0]

        if env_batched:
            env_batch_size = next(iter(nobs.values())).shape[1]
            nobs = dict_apply(nobs,
                lambda x: x.reshape(B * env_batch_size, *x.shape[2:]))
        global_cond = self.obs_encoder(nobs)

        if env_batched:
            cond_data = torch.zeros(size=(B*env_batch_size, self.action_horizon, self.action_dim), device=self.device, dtype=self.dtype)
        else:
            cond_data = torch.zeros(size=(B, self.action_horizon, self.action_dim), device=self.device, dtype=self.dtype)
        cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)

        if fixed_action_prefix is not None and self.inpaint_fixed_action_prefix:
            n_fixed_steps = fixed_action_prefix.shape[1]
            cond_data[:, :n_fixed_steps] = fixed_action_prefix
            cond_mask[:, :n_fixed_steps] = True
            cond_data = self.normalizer['action'].normalize(cond_data)

        nsample = self.conditional_sample(
            condition_data=cond_data,
            condition_mask=cond_mask,
            local_cond=None,
            global_cond=global_cond,
            chunk_start_pose=chunk_start_pose,
            obstacle_info=obstacle_info,
            **self.kwargs
        )

        if env_batched:
            assert nsample.shape == (B * env_batch_size, self.action_horizon, self.action_dim)
        else:
            assert nsample.shape == (B, self.action_horizon, self.action_dim)
        action_pred = self.normalizer['action'].unnormalize(nsample)
        if env_batched:
            action_pred = action_pred.reshape(B, env_batch_size, self.action_horizon, self.action_dim)

        result = {
            'action': action_pred,
            'action_pred': action_pred
        }
        return result

    # ========= training  ============
    def set_normalizer(self, normalizer: LinearNormalizer):
        self.normalizer.load_state_dict(normalizer.state_dict())

    def compute_loss(self, batch):
        assert 'valid_mask' not in batch
        nobs = self.normalizer.normalize(batch['obs'])
        nactions = self.normalizer['action'].normalize(batch['action'])

        assert self.obs_as_global_cond
        global_cond = self.obs_encoder(nobs)

        if self.train_diffusion_n_samples != 1:
            global_cond = torch.repeat_interleave(global_cond,
                repeats=self.train_diffusion_n_samples, dim=0)
            nactions = torch.repeat_interleave(nactions,
                repeats=self.train_diffusion_n_samples, dim=0)

        trajectory = nactions
        noise = torch.randn(trajectory.shape, device=trajectory.device)
        noise_new = noise + self.input_pertub * torch.randn(trajectory.shape, device=trajectory.device)

        timesteps = torch.randint(
            0, self.noise_scheduler.config.num_train_timesteps,
            (nactions.shape[0],), device=trajectory.device
        ).long()

        noisy_trajectory = self.noise_scheduler.add_noise(
            trajectory, noise_new, timesteps)

        pred = self.model(
            noisy_trajectory,
            timesteps,
            local_cond=None,
            global_cond=global_cond
        )

        pred_type = self.noise_scheduler.config.prediction_type
        if pred_type == 'epsilon':
            target = noise
        elif pred_type == 'sample':
            target = trajectory
        else:
            raise ValueError(f"Unsupported prediction type {pred_type}")

        loss = F.mse_loss(pred, target, reduction='none')
        loss = loss.type(loss.dtype)
        loss = reduce(loss, 'b ... -> b (...)', 'mean')
        loss = loss.mean()

        return loss

    def forward(self, batch):
        return self.compute_loss(batch)


EmbodiSteerEESpacePolicy = DiffusionUnetTimmPolicyEESpace

__all__ = ["EmbodiSteerEESpacePolicy", "DiffusionUnetTimmPolicyEESpace"]
