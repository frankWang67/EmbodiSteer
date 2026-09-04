from typing import Dict, Optional, Tuple

import torch

from diffusion_policy.common.pytorch_util import dict_apply
from .ee2joint import (
    DiffusionUnetTimmPolicyJointSpace,
)

from embodisteer.kinematics.rotation import (
    axis_angle_to_matrix,
    matrix_to_rotation_6d as matrix_to_rot6d,
    rotation_6d_to_matrix as rot6d_to_matrix,
)


class DiffusionUnetTimmPolicyBaseline(DiffusionUnetTimmPolicyJointSpace):
    """
    Baseline policies for comparison:
      - "post_hoc_cbf": full EE-space denoising, then IK, then one-shot CBF-QP correction
      - "batch_sampling": full EE-space denoising with B samples, then IK, select safest

    Both baselines perform denoising entirely in end-effector space (no per-step guidance),
    then convert to joint space post-hoc for collision handling.
    """

    def __init__(
        self,
        *args,
        baseline_method: str = "post_hoc_cbf",
        batch_sampling_num: int = 32,
        **kwargs,
    ):
        # Force guidance_method="" in parent so denoising has no per-step guidance
        kwargs["guidance_method"] = ""
        super().__init__(*args, **kwargs)

        baseline_method = str(baseline_method).lower().strip()
        if baseline_method not in ("post_hoc_cbf", "batch_sampling"):
            raise ValueError(
                f"Unsupported baseline_method={baseline_method}. "
                "Use one of: ['post_hoc_cbf', 'batch_sampling']"
            )
        self.baseline_method = baseline_method
        self.batch_sampling_num = int(max(batch_sampling_num, 2))

    def _ee_space_conditional_sample(
        self,
        condition_data,
        condition_mask,
        local_cond=None,
        global_cond=None,
        generator=None,
        **kwargs,
    ):
        """
        Run vanilla EE-space denoising (no guidance, no joint-space conversion).
        This is the base DiffusionUnetTimmPolicyEESpace.conditional_sample with
        use_ee_guidance=False behavior.
        """
        model = self.model
        scheduler = self.noise_scheduler

        trajectory = torch.randn(
            size=condition_data.shape,
            dtype=condition_data.dtype,
            device=condition_data.device,
            generator=generator,
        )

        scheduler.set_timesteps(self.num_inference_steps)

        for t in scheduler.timesteps:
            trajectory[condition_mask] = condition_data[condition_mask]
            model_output = model(trajectory, t, local_cond=local_cond, global_cond=global_cond)
            trajectory = scheduler.step(
                model_output, t, trajectory, generator=generator, **kwargs,
            ).prev_sample

        trajectory[condition_mask] = condition_data[condition_mask]
        return trajectory

    def _post_hoc_cbf(
        self,
        cart_n_sample: torch.Tensor,
        condition_mask: torch.Tensor,
        chunk_start_pose: torch.Tensor,
        obstacle_info,
        current_joint_angles: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Post-hoc CBF-QP baseline:
        1. Unnormalize EE trajectory
        2. IK to joint space
        3. Build collision world
        4. Apply CBF-QP correction per timestep
        5. FK back to EE space
        """
        self._ensure_kinematics(cart_n_sample.device)
        self._build_world_collision(obstacle_info, cart_n_sample.device, cart_n_sample.dtype)

        bsz, horizon, _ = cart_n_sample.shape
        cart_phys = self.normalizer["action"].unnormalize(cart_n_sample)

        base_pos = chunk_start_pose[:, :3]
        base_rot = axis_angle_to_matrix(chunk_start_pose[:, 3:6])
        base_rot_t = base_rot.transpose(-2, -1)

        # Convert relative to absolute pose
        rel9 = cart_phys[..., :9]
        rel_pos = rel9[..., :3]
        rel_rot = rot6d_to_matrix(rel9[..., 3:])
        abs_pos = (base_rot.unsqueeze(1) @ rel_pos.unsqueeze(-1)).squeeze(-1) + base_pos.unsqueeze(1)
        abs_rot = base_rot.unsqueeze(1) @ rel_rot

        # IK to get joint trajectory
        if current_joint_angles is not None:
            q_start = current_joint_angles[..., :self._robot_dof].to(device=abs_pos.device, dtype=abs_pos.dtype)
        else:
            q_start = self._solve_start_joint_from_pose(chunk_start_pose)
        q_seed = q_start.unsqueeze(1).expand(bsz, horizon, self._robot_dof)
        q_arm = self._ik_from_absolute(abs_pos, abs_rot, q_seed)

        # Apply CBF-QP correction if collision world exists
        if self._world_collision is not None:
            cond_step_mask = condition_mask.any(dim=-1)
            h_value, grad_h, _ = self._compute_cbf_linearization(q_arm)
            jac = self._jacobian(q_arm).reshape(bsz, horizon, 6, self._robot_dof)
            gamma = torch.tensor(self.guidance_scale, device=q_arm.device, dtype=q_arm.dtype)
            dq_cbf, _, _, _ = self._solve_batched_cbf_qp(jac, grad_h, h_value, gamma)

            if torch.any(cond_step_mask):
                dq_cbf = dq_cbf.clone()
                dq_cbf[cond_step_mask] = 0.0
            if self.guidance_grad_clip > 0:
                dq_cbf = torch.clamp(dq_cbf, -self.guidance_grad_clip, self.guidance_grad_clip)
            q_arm = q_arm + dq_cbf

        # Build full q_traj with gripper
        grip = cart_phys[..., 9:10]
        q_traj = torch.cat([q_arm, grip], dim=-1)
        self._last_joint_traj = q_traj.detach()

        # FK back to EE space for normalized output
        abs_p_f, abs_r_f = self._fk_to_absolute(q_arm)

        rel_pos_f = (base_rot_t.unsqueeze(1) @ (abs_p_f - base_pos.unsqueeze(1)).unsqueeze(-1)).squeeze(-1)
        rel_rot_f = base_rot_t.unsqueeze(1) @ abs_r_f
        rel_rot6d_f = matrix_to_rot6d(rel_rot_f.reshape(-1, 3, 3)).reshape(bsz, horizon, 6)
        cart_final = torch.cat([rel_pos_f, rel_rot6d_f, grip], dim=-1)
        cart_final_n = self.normalizer["action"].normalize(cart_final)
        cart_final_n[condition_mask] = cart_n_sample[condition_mask]
        return cart_final_n, q_traj

    def _batch_sampling(
        self,
        condition_data: torch.Tensor,
        condition_mask: torch.Tensor,
        global_cond: torch.Tensor,
        chunk_start_pose: torch.Tensor,
        obstacle_info,
        generator=None,
        current_joint_angles: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Batch sampling baseline:
        1. Repeat global_cond B times, sample B trajectories in EE space
        2. IK each trajectory to joint space
        3. Query collision distance for each
        4. Select the trajectory with maximum minimum distance to obstacles
        """
        self._ensure_kinematics(condition_data.device)
        self._build_world_collision(obstacle_info, condition_data.device, condition_data.dtype)

        bsz, horizon, action_dim = condition_data.shape
        B_sample = self.batch_sampling_num

        # Repeat condition data and global_cond for batch sampling
        cond_data_rep = condition_data.repeat(B_sample, 1, 1)
        cond_mask_rep = condition_mask.repeat(B_sample, 1, 1)
        global_cond_rep = global_cond.repeat(B_sample, 1)

        # Run EE-space denoising for all B_sample trajectories at once
        cart_n_samples = self._ee_space_conditional_sample(
            condition_data=cond_data_rep,
            condition_mask=cond_mask_rep,
            local_cond=None,
            global_cond=global_cond_rep,
            generator=generator,
            **kwargs,
        )
        # Reshape: (B_sample * bsz, horizon, action_dim) -> (B_sample, bsz, horizon, action_dim)
        cart_n_samples = cart_n_samples.reshape(B_sample, bsz, horizon, action_dim)

        base_pos = chunk_start_pose[:, :3]
        base_rot = axis_angle_to_matrix(chunk_start_pose[:, 3:6])
        base_rot_t = base_rot.transpose(-2, -1)

        if current_joint_angles is not None:
            q_start = current_joint_angles[..., :self._robot_dof].to(device=condition_data.device, dtype=condition_data.dtype)
        else:
            q_start = self._solve_start_joint_from_pose(chunk_start_pose)

        best_cart_n = None
        best_q_traj = None
        best_max_dist = None

        for s in range(B_sample):
            cart_n_s = cart_n_samples[s]  # (bsz, horizon, action_dim)
            cart_phys_s = self.normalizer["action"].unnormalize(cart_n_s)

            rel9 = cart_phys_s[..., :9]
            rel_pos = rel9[..., :3]
            rel_rot = rot6d_to_matrix(rel9[..., 3:])
            abs_pos = (base_rot.unsqueeze(1) @ rel_pos.unsqueeze(-1)).squeeze(-1) + base_pos.unsqueeze(1)
            abs_rot = base_rot.unsqueeze(1) @ rel_rot

            q_seed = q_start.unsqueeze(1).expand(bsz, horizon, self._robot_dof)
            q_arm = self._ik_from_absolute(abs_pos, abs_rot, q_seed)

            # Compute distance for this sample
            if self._world_collision is not None:
                dist_worst = self._query_worst_signed_distance(q_arm)
                # max over horizon: the most dangerous timestep for this trajectory
                # cuRobo convention: positive = inside obstacle, negative = safe
                # A trajectory is only as safe as its worst timestep
                max_dist_per_batch = dist_worst.max(dim=-1).values  # (bsz,)
            else:
                max_dist_per_batch = torch.zeros(bsz, device=cart_n_s.device, dtype=cart_n_s.dtype)

            grip = cart_phys_s[..., 9:10]
            q_traj_s = torch.cat([q_arm, grip], dim=-1)

            if best_max_dist is None:
                best_max_dist = max_dist_per_batch
                best_cart_n = cart_n_s
                best_q_traj = q_traj_s
            else:
                # Select samples where this trajectory is safer
                # (lower max signed distance = worst-case timestep is further from obstacle)
                better = max_dist_per_batch < best_max_dist
                best_max_dist = torch.where(better, max_dist_per_batch, best_max_dist)
                better_expanded = better.unsqueeze(-1).unsqueeze(-1)
                best_cart_n = torch.where(better_expanded, cart_n_s, best_cart_n)
                better_q = better.unsqueeze(-1).unsqueeze(-1).expand_as(q_traj_s)
                best_q_traj = torch.where(better_q, q_traj_s, best_q_traj)

        self._last_joint_traj = best_q_traj.detach()

        # FK the best joint trajectory back to EE space for consistent output
        q_arm_best = best_q_traj[..., : self._robot_dof]
        grip_best = best_q_traj[..., self._robot_dof : self._robot_dof + 1]
        abs_p_f, abs_r_f = self._fk_to_absolute(q_arm_best)
        rel_pos_f = (base_rot_t.unsqueeze(1) @ (abs_p_f - base_pos.unsqueeze(1)).unsqueeze(-1)).squeeze(-1)
        rel_rot_f = base_rot_t.unsqueeze(1) @ abs_r_f
        rel_rot6d_f = matrix_to_rot6d(rel_rot_f.reshape(-1, 3, 3)).reshape(bsz, horizon, 6)
        cart_final = torch.cat([rel_pos_f, rel_rot6d_f, grip_best], dim=-1)
        cart_final_n = self.normalizer["action"].normalize(cart_final)
        cart_final_n[condition_mask] = condition_data[condition_mask]
        return cart_final_n, best_q_traj

    def predict_action(
        self,
        obs_dict: Dict[str, torch.Tensor],
        env_batched=False,
        chunk_start_pose: torch.Tensor = None,
        obstacle_info=None,
        current_joint_angles: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        assert "past_action" not in obs_dict

        nobs = self.normalizer.normalize(obs_dict)
        B = next(iter(nobs.values())).shape[0]

        if env_batched:
            env_batch_size = next(iter(nobs.values())).shape[1]
            nobs = dict_apply(nobs, lambda x: x.reshape(B * env_batch_size, *x.shape[2:]))
        global_cond = self.obs_encoder(nobs)

        if env_batched:
            cond_data = torch.zeros(
                size=(B * env_batch_size, self.action_horizon, self.action_dim),
                device=self.device, dtype=self.dtype,
            )
        else:
            cond_data = torch.zeros(
                size=(B, self.action_horizon, self.action_dim),
                device=self.device, dtype=self.dtype,
            )
        cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)

        if chunk_start_pose is None:
            raise ValueError("chunk_start_pose must be provided for baseline policy inference.")

        if self.baseline_method == "post_hoc_cbf":
            cart_n_sample = self._ee_space_conditional_sample(
                condition_data=cond_data,
                condition_mask=cond_mask,
                local_cond=None,
                global_cond=global_cond,
                **self.kwargs,
            )
            nsample, q_traj = self._post_hoc_cbf(
                cart_n_sample=cart_n_sample,
                condition_mask=cond_mask,
                chunk_start_pose=chunk_start_pose,
                obstacle_info=obstacle_info,
                current_joint_angles=current_joint_angles,
            )
        else:  # batch_sampling
            nsample, q_traj = self._batch_sampling(
                condition_data=cond_data,
                condition_mask=cond_mask,
                global_cond=global_cond,
                chunk_start_pose=chunk_start_pose,
                obstacle_info=obstacle_info,
                current_joint_angles=current_joint_angles,
                **self.kwargs,
            )

        if env_batched:
            actual_bsz = B * env_batch_size
        else:
            actual_bsz = B
        assert nsample.shape == (actual_bsz, self.action_horizon, self.action_dim)
        action_pred = self.normalizer["action"].unnormalize(nsample)

        joint_action_pred = None
        if self._last_joint_traj is not None:
            joint_action_pred = self._joint_traj_to_env_action(self._last_joint_traj)

        if env_batched:
            action_pred = action_pred.reshape(B, env_batch_size, self.action_horizon, self.action_dim)
            if joint_action_pred is not None:
                joint_action_pred = joint_action_pred.reshape(
                    B, env_batch_size, self.action_horizon, self.arm_dof + 1
                )

        result = {
            "action": action_pred,
            "action_pred": action_pred,
        }
        if joint_action_pred is not None:
            result["joint_action"] = joint_action_pred
            result["joint_action_pred"] = joint_action_pred
        return result


__all__ = ["DiffusionUnetTimmPolicyBaseline"]
