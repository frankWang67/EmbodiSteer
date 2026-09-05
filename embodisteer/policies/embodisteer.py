"""The paper's EmbodiSteer joint-space policy.

This module is intentionally small and algorithm-facing. Robot-specific FK,
IK, Jacobian, collision-world and scheduler plumbing is supplied by
``ee2joint._JointSpacePolicyRuntime``; the denoising loop below is the
paper method described in ``docs/method.md``.
"""

from __future__ import annotations

from typing import Optional

import torch

from embodisteer.kinematics.rotation import (
    axis_angle_to_matrix,
    rotation_6d_to_matrix as rot6d_to_matrix,
)
from .ee2joint import _JointSpacePolicyRuntime


class DiffusionUnetTimmPolicyEmbodiSteer(_JointSpacePolicyRuntime):
    """EmbodiSteer: Cartesian diffusion projected into joint space with CBF."""

    def __init__(self, *args, guidance_method: str = "cbf", **kwargs):
        guidance_method = str(guidance_method).lower().strip()
        if guidance_method != "cbf":
            raise ValueError(
                "DiffusionUnetTimmPolicyEmbodiSteer requires guidance_method='cbf'. "
                "Use DiffusionUnetTimmPolicyJointSpace for no-guidance or GD comparison."
            )
        super().__init__(*args, guidance_method=guidance_method, **kwargs)

    def _apply_cbf_guidance(
        self,
        q_arm_new: torch.Tensor,
        jac_ref: torch.Tensor,
        q_ref_for_jac: torch.Tensor,
        cond_step_mask: torch.Tensor,
        step_index: int,
        num_steps: int,
        timestep,
    ) -> torch.Tensor:
        """Apply one paper CBF-QP correction to each denoising state."""
        if self._world_collision is None:
            return q_arm_new

        batch, horizon, robot_dof = q_arm_new.shape
        h_value, grad_h, _ = self._compute_cbf_linearization(q_arm_new)
        state_change = (q_arm_new - q_ref_for_jac).abs().max()
        if state_change < 0.5:
            jac_lin = jac_ref.reshape(batch, horizon, 6, robot_dof)
        else:
            jac_lin = self._jacobian(q_arm_new).reshape(
                batch, horizon, 6, robot_dof
            )
        gamma = self._guidance_scale_at(
            step_index,
            num_steps,
            timestep,
            q_arm_new.dtype,
            q_arm_new.device,
        )
        dq_cbf, _, _, _ = self._cbf_qp_fn(
            jac_lin,
            grad_h,
            h_value,
            gamma,
        )
        if torch.any(cond_step_mask):
            dq_cbf = dq_cbf.clone()
            dq_cbf[cond_step_mask] = 0.0
        if self.guidance_grad_clip > 0:
            dq_cbf = torch.clamp(
                dq_cbf,
                -self.guidance_grad_clip,
                self.guidance_grad_clip,
            )
        return q_arm_new + dq_cbf

    def conditional_sample(
        self,
        condition_data,
        condition_mask,
        local_cond=None,
        global_cond=None,
        generator=None,
        chunk_start_pose: Optional[torch.Tensor] = None,
        obstacle_info=None,
        current_joint_angles: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        """Run the EmbodiSteer reverse-diffusion algorithm.

        Each reverse-diffusion step denoises in Cartesian space, realizes the
        target through the robot Jacobian, then applies one CBF correction.
        """
        if chunk_start_pose is None:
            raise ValueError("chunk_start_pose is required for joint-space inference.")

        # The continuity diagnostic temporarily disables CBF on this warmed
        # instance for an exact-noise ablation; GD is never supported here.
        if self.guidance_method not in ("", "cbf"):
            raise ValueError("EmbodiSteer inference supports CBF, not GD.")

        self._ensure_kinematics(condition_data.device)
        use_guidance = self.guidance_method != ""
        if use_guidance:
            self._build_world_collision(
                obstacle_info=obstacle_info,
                device=condition_data.device,
                dtype=condition_data.dtype,
            )

        base_pos = chunk_start_pose[:, :3]
        base_rot = axis_angle_to_matrix(chunk_start_pose[:, 3:6])
        base_rot_t = base_rot.transpose(-2, -1)
        bsz, horizon = condition_data.shape[:2]

        # Jacobian-projected task noise and joint-space inpainting targets.
        q_traj, q_cond, cond_step_mask = self._initialize_joint_sample(
            condition_data, condition_mask, chunk_start_pose,
            current_joint_angles, base_pos, base_rot, generator,
        )

        # Cartesian denoising -> Jacobian realization -> CBF correction.
        scheduler = self.noise_scheduler
        scheduler.set_timesteps(self.num_inference_steps)
        timesteps = list(scheduler.timesteps)
        for step_index, timestep in enumerate(timesteps):
            if q_cond is not None:
                q_traj[cond_step_mask] = q_cond[cond_step_mask]

            q_arm_curr = q_traj[..., : self._robot_dof]
            # 1. FK and relative Cartesian encoding for the frozen U-Net.
            cart_n_curr, abs_pos_curr, abs_rot_curr = self._joint_to_cartesian(
                q_traj, base_pos, base_rot_t
            )
            cart_n_curr[condition_mask] = condition_data[condition_mask]

            # 2. One Cartesian reverse-diffusion step.
            eps_cart_n = self.model(
                cart_n_curr,
                timestep,
                local_cond=local_cond,
                global_cond=global_cond,
            )
            cart_n_prev_tgt = scheduler.step(
                eps_cart_n,
                timestep,
                cart_n_curr,
                generator=generator,
                **kwargs,
            ).prev_sample
            cart_n_prev_tgt[condition_mask] = condition_data[condition_mask]
            cart_phys_prev_tgt = self.normalizer["action"].unnormalize(
                cart_n_prev_tgt
            )

            rel9_tgt = cart_phys_prev_tgt[..., :9]
            rel_pos_tgt = rel9_tgt[..., :3]
            rel_rot_tgt = rot6d_to_matrix(rel9_tgt[..., 3:])
            abs_pos_tgt = (
                base_rot.unsqueeze(1) @ rel_pos_tgt.unsqueeze(-1)
            ).squeeze(-1) + base_pos.unsqueeze(1)
            abs_rot_tgt = base_rot.unsqueeze(1) @ rel_rot_tgt

            # 3. Realize the world-frame pose residual through the Jacobian.
            twist6 = self._twist_fn(
                abs_pos_curr,
                abs_rot_curr,
                abs_pos_tgt,
                abs_rot_tgt,
            ).clone()
            jac = self._jacobian(q_arm_curr)
            dq = self._dls_pinv_map(
                twist6.reshape(-1, 6), jac
            ).reshape(bsz, horizon, self._robot_dof)
            if self.max_dq_per_step > 0:
                dq = torch.clamp(
                    dq,
                    -self.max_dq_per_step,
                    self.max_dq_per_step,
                )
            q_arm_new = q_arm_curr + dq
            # 4. Whole-body CBF-QP correction at the projected joint state.
            if use_guidance:
                q_arm_new = self._apply_cbf_guidance(
                    q_arm_new,
                    jac,
                    q_arm_curr,
                    cond_step_mask,
                    step_index,
                    len(timesteps),
                    timestep,
                )
            q_traj = torch.cat(
                [q_arm_new, cart_phys_prev_tgt[..., 9:10]], dim=-1
            )

        return self._finalize_joint_sample(
            q_traj, q_cond, cond_step_mask, condition_data, condition_mask,
            base_pos, base_rot_t,
        )


EmbodiSteerJointPolicy = DiffusionUnetTimmPolicyEmbodiSteer

__all__ = ["DiffusionUnetTimmPolicyEmbodiSteer", "EmbodiSteerJointPolicy"]
