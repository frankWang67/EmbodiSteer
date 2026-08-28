from typing import Dict, Optional, Tuple
import inspect

import torch
import torch.nn.functional as F

from curobo.types.math import Pose

from diffusion_policy.common.pytorch_util import dict_apply
from embodisteer.kinematics.rotation import (
    axis_angle_to_matrix,
    matrix_to_quaternion,
    rotation_6d_to_matrix as rot6d_to_matrix,
)
from .baselines import (
    DiffusionUnetTimmPolicyBaseline,
)


class DiffusionUnetTimmPolicyJM2D(DiffusionUnetTimmPolicyBaseline):
    """JM2D conditional-generation baseline for whole-body constraints.

    The learned variable remains the Cartesian action chunk used by the frozen
    diffusion policy. At every outer denoising step, this policy:

      1. stochastically denoises ``N`` clean Cartesian candidates;
      2. realizes every clean candidate on the target robot with batched IK;
      3. evaluates the whole-body collision interaction potential;
      4. forms the importance-weighted clean-sample/score estimate; and
      5. applies one reverse-diffusion update to the outer Cartesian sample.

    This is the conditional-generation special case of JM2D (there is no
    separately diffused model-based variable ``k``). The final Cartesian chunk
    is converted to joint space and receives the one-shot CBF correction used
    by the JM2D paper for residual infeasibility.
    """

    def __init__(
        self,
        *args,
        jm2d_num_samples: int = 16,
        jm2d_temperature: float = 0.01,
        jm2d_eta: float = 1.0,
        **kwargs,
    ):
        # The parent class provides EE sampling, IK/SDF utilities, and the
        # faithful final one-shot CBF correction. Its own baseline selector is
        # not used by this class.
        kwargs.pop("baseline_method", None)
        batch_sampling_num = kwargs.pop("batch_sampling_num", jm2d_num_samples)
        super().__init__(
            *args,
            baseline_method="post_hoc_cbf",
            batch_sampling_num=batch_sampling_num,
            **kwargs,
        )

        self.jm2d_num_samples = int(jm2d_num_samples)
        self.jm2d_temperature = float(jm2d_temperature)
        self.jm2d_eta = float(jm2d_eta)
        if self.jm2d_num_samples < 1:
            raise ValueError("jm2d_num_samples must be at least 1.")
        if self.jm2d_temperature <= 0.0:
            raise ValueError("jm2d_temperature must be positive.")
        if self.jm2d_eta < 0.0:
            raise ValueError("jm2d_eta must be non-negative.")

        self._jm2d_scheduler_supports_eta = (
            "eta" in inspect.signature(self.noise_scheduler.step).parameters
        )
        self._last_jm2d_stats = None

    def _scheduler_step(
        self,
        model_output: torch.Tensor,
        timestep,
        sample: torch.Tensor,
        generator=None,
        eta: Optional[float] = None,
        **kwargs,
    ) -> torch.Tensor:
        step_kwargs = dict(kwargs)
        # The checkpoint used in this codebase has a DDIM scheduler. Positive
        # eta is essential for diverse MC proposals from the same noisy state.
        # DDPM schedulers are already stochastic and do not expose eta.
        step_kwargs.pop("eta", None)
        if self._jm2d_scheduler_supports_eta and eta is not None:
            step_kwargs["eta"] = float(eta)
        return self.noise_scheduler.step(
            model_output,
            timestep,
            sample,
            generator=generator,
            **step_kwargs,
        ).prev_sample

    def _sample_clean_candidates(
        self,
        trajectory: torch.Tensor,
        outer_idx: int,
        timesteps,
        condition_data_rep: torch.Tensor,
        condition_mask_rep: torch.Tensor,
        local_cond_rep: Optional[torch.Tensor],
        global_cond_rep: torch.Tensor,
        generator=None,
        **kwargs,
    ) -> torch.Tensor:
        bsz, horizon, action_dim = trajectory.shape
        n_samples = self.jm2d_num_samples
        candidates = (
            trajectory.unsqueeze(1)
            .expand(bsz, n_samples, horizon, action_dim)
            .reshape(bsz * n_samples, horizon, action_dim)
            .clone()
        )

        for inner_t in timesteps[outer_idx:]:
            candidates[condition_mask_rep] = condition_data_rep[condition_mask_rep]
            model_output = self.model(
                candidates,
                inner_t,
                local_cond=local_cond_rep,
                global_cond=global_cond_rep,
            )
            candidates = self._scheduler_step(
                model_output,
                inner_t,
                candidates,
                generator=generator,
                eta=self.jm2d_eta,
                **kwargs,
            )

        candidates[condition_mask_rep] = condition_data_rep[condition_mask_rep]
        return candidates.reshape(bsz, n_samples, horizon, action_dim)

    def _ik_from_absolute_with_success(
        self,
        abs_pos: torch.Tensor,
        abs_rot: torch.Tensor,
        seed_q: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Batched IK that also exposes the feasibility mask for JM2D weights."""
        bsz, horizon, _ = abs_pos.shape
        pos_flat = abs_pos.reshape(-1, 3)
        quat_flat = matrix_to_quaternion(abs_rot.reshape(-1, 3, 3))
        seed_flat = seed_q.reshape(-1, self._robot_dof).contiguous()
        goal_pose = Pose(position=pos_flat, quaternion=quat_flat)

        n_seeds = int(self.ik_num_seeds)
        seed_cfg = seed_flat.unsqueeze(1).expand(
            bsz * horizon, n_seeds, self._robot_dof
        ).contiguous()
        with torch.enable_grad():
            ik_result = self._ik_solver.solve_batch(
                goal_pose,
                retract_config=seed_flat,
                seed_config=seed_cfg,
            )

        q_flat = ik_result.solution.squeeze(1)
        success = torch.ones(
            (bsz * horizon,), device=q_flat.device, dtype=torch.bool
        )
        if hasattr(ik_result, "success"):
            success = ik_result.success
            while success.ndim > 1:
                success = success.squeeze(-1)
            success = success.to(device=q_flat.device, dtype=torch.bool)
            q_flat = torch.where(success.unsqueeze(-1), q_flat, seed_flat)

        return (
            q_flat.reshape(bsz, horizon, self._robot_dof),
            success.reshape(bsz, horizon),
        )

    def _cartesian_candidates_to_joint(
        self,
        cart_n_candidates: torch.Tensor,
        chunk_start_pose: torch.Tensor,
        q_start: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        bsz, n_samples, horizon, action_dim = cart_n_candidates.shape
        cart_n_flat = cart_n_candidates.reshape(
            bsz * n_samples, horizon, action_dim
        )
        cart_phys = self.normalizer["action"].unnormalize(cart_n_flat)

        chunk_pose_rep = chunk_start_pose.repeat_interleave(n_samples, dim=0)
        base_pos = chunk_pose_rep[:, :3]
        base_rot = axis_angle_to_matrix(chunk_pose_rep[:, 3:6])
        rel_pose9 = cart_phys[..., :9]
        rel_pos = rel_pose9[..., :3]
        rel_rot = rot6d_to_matrix(rel_pose9[..., 3:])
        abs_pos = (
            base_rot.unsqueeze(1) @ rel_pos.unsqueeze(-1)
        ).squeeze(-1) + base_pos.unsqueeze(1)
        abs_rot = base_rot.unsqueeze(1) @ rel_rot

        q_start_rep = q_start.repeat_interleave(n_samples, dim=0)
        q_seed = q_start_rep.unsqueeze(1).expand(
            bsz * n_samples, horizon, self._robot_dof
        )
        q_arm, ik_success = self._ik_from_absolute_with_success(
            abs_pos, abs_rot, q_seed
        )
        return (
            q_arm.reshape(bsz, n_samples, horizon, self._robot_dof),
            ik_success.reshape(bsz, n_samples, horizon),
        )

    def _query_candidate_signed_distance(
        self,
        q_candidates: torch.Tensor,
    ) -> torch.Tensor:
        """Collision query for an environment-major ``(B, N, H, D)`` batch."""
        bsz, n_samples, horizon, dof = q_candidates.shape
        q_arm = q_candidates.reshape(bsz * n_samples, horizon, dof)
        q_flat = q_arm.reshape(bsz * n_samples * horizon, dof).contiguous()
        kin_state = self._kin_model.get_state(q_flat)
        spheres = kin_state.link_spheres_tensor.reshape(
            bsz * n_samples, horizon, -1, 4
        )
        self._coll_query_buffer.update_buffer_shape(
            spheres.shape,
            self._tensor_args,
            self._world_collision.collision_types,
        )

        n_worlds = int(self._coll_env_query_idx.numel())
        if n_worlds == 1:
            env_query_idx = torch.zeros(
                bsz * n_samples, device=spheres.device, dtype=torch.int32
            )
        elif n_worlds == bsz:
            env_query_idx = torch.arange(
                bsz, device=spheres.device, dtype=torch.int32
            ).repeat_interleave(n_samples)
        else:
            raise ValueError(
                f"Collision world has {n_worlds} environments, but JM2D "
                f"received batch size {bsz}."
            )

        dist = self._world_collision.get_sphere_distance(
            spheres,
            self._coll_query_buffer,
            self._coll_weight,
            self._coll_activation_distance,
            env_query_idx=env_query_idx,
            return_loss=True,
            compute_esdf=True,
        )
        return self._aggregate_signed_distance(dist).reshape(
            bsz, n_samples, horizon
        )

    def _importance_weights(
        self,
        q_candidates: torch.Tensor,
        ik_success: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        bsz, n_samples = q_candidates.shape[:2]
        if self._world_collision is None:
            energy = torch.zeros(
                (bsz, n_samples),
                device=q_candidates.device,
                dtype=q_candidates.dtype,
            )
            min_clearance = torch.full_like(energy, float("inf"))
        else:
            # cuRobo: positive values are penetration; h(q) = -dist. Thus
            # ReLU(dist + d_safe)^2 is exactly ReLU(d_safe - h(q))^2.
            dist = self._query_candidate_signed_distance(q_candidates)
            violation = F.relu(dist + self.guidance_safety_margin)
            energy = violation.pow(2).sum(dim=-1)
            min_clearance = -dist.max(dim=-1).values

        energy = torch.nan_to_num(
            energy,
            nan=1e6,
            posinf=1e6,
            neginf=0.0,
        )
        valid_trajectory = ik_success.all(dim=-1)
        log_weights_unmasked = -energy / self.jm2d_temperature
        neg_inf = torch.full_like(log_weights_unmasked, -torch.inf)
        log_weights_masked = torch.where(
            valid_trajectory, log_weights_unmasked, neg_inf
        )

        # If every proposal for one environment has invalid IK, retaining the
        # collision-energy ordering is more stable than producing NaNs. The
        # final one-shot CBF/IK stage remains responsible for feasibility.
        has_valid = valid_trajectory.any(dim=-1, keepdim=True)
        log_weights = torch.where(
            has_valid, log_weights_masked, log_weights_unmasked
        )
        weights = torch.softmax(log_weights, dim=-1)
        weights = torch.nan_to_num(weights, nan=1.0 / float(n_samples))
        weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        return weights, energy, min_clearance

    def _model_output_from_clean_estimate(
        self,
        trajectory: torch.Tensor,
        clean_estimate: torch.Tensor,
        timestep,
    ) -> torch.Tensor:
        alpha_prod_t = self.noise_scheduler.alphas_cumprod[timestep].to(
            device=trajectory.device, dtype=trajectory.dtype
        )
        sqrt_alpha = alpha_prod_t.sqrt()
        sqrt_one_minus_alpha = (1.0 - alpha_prod_t).clamp_min(1e-12).sqrt()
        epsilon = (
            trajectory - sqrt_alpha * clean_estimate
        ) / sqrt_one_minus_alpha

        prediction_type = self.noise_scheduler.config.prediction_type
        if prediction_type == "epsilon":
            return epsilon
        if prediction_type == "sample":
            return clean_estimate
        if prediction_type == "v_prediction":
            return sqrt_alpha * epsilon - sqrt_one_minus_alpha * clean_estimate
        raise ValueError(f"Unsupported prediction type: {prediction_type}")

    def conditional_sample(
        self,
        condition_data: torch.Tensor,
        condition_mask: torch.Tensor,
        local_cond=None,
        global_cond=None,
        generator=None,
        chunk_start_pose: Optional[torch.Tensor] = None,
        obstacle_info=None,
        current_joint_angles: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if chunk_start_pose is None:
            raise ValueError("chunk_start_pose is required for JM2D inference.")

        self._ensure_kinematics(condition_data.device)
        self._build_world_collision(
            obstacle_info,
            condition_data.device,
            condition_data.dtype,
        )

        bsz, horizon, action_dim = condition_data.shape
        n_samples = self.jm2d_num_samples
        if current_joint_angles is None:
            q_start = self._solve_start_joint_from_pose(chunk_start_pose)
        else:
            q_start = current_joint_angles[..., : self._robot_dof].to(
                device=condition_data.device,
                dtype=condition_data.dtype,
            )

        trajectory = torch.randn(
            condition_data.shape,
            device=condition_data.device,
            dtype=condition_data.dtype,
            generator=generator,
        )
        trajectory[condition_mask] = condition_data[condition_mask]

        condition_data_rep = condition_data.repeat_interleave(n_samples, dim=0)
        condition_mask_rep = condition_mask.repeat_interleave(n_samples, dim=0)
        global_cond_rep = global_cond.repeat_interleave(n_samples, dim=0)
        local_cond_rep = None
        if local_cond is not None:
            local_cond_rep = local_cond.repeat_interleave(n_samples, dim=0)

        self.noise_scheduler.set_timesteps(self.num_inference_steps)
        timesteps = list(self.noise_scheduler.timesteps)
        ess_per_step = []
        energy_per_step = []
        clearance_per_step = []
        ik_pose_success_per_step = []
        ik_trajectory_success_per_step = []

        for outer_idx, timestep in enumerate(timesteps):
            trajectory[condition_mask] = condition_data[condition_mask]
            clean_candidates = self._sample_clean_candidates(
                trajectory=trajectory,
                outer_idx=outer_idx,
                timesteps=timesteps,
                condition_data_rep=condition_data_rep,
                condition_mask_rep=condition_mask_rep,
                local_cond_rep=local_cond_rep,
                global_cond_rep=global_cond_rep,
                generator=generator,
                **kwargs,
            )
            q_candidates, ik_success = self._cartesian_candidates_to_joint(
                clean_candidates,
                chunk_start_pose,
                q_start,
            )
            weights, energy, min_clearance = self._importance_weights(
                q_candidates,
                ik_success,
            )
            clean_estimate = torch.sum(
                weights[..., None, None] * clean_candidates,
                dim=1,
            )
            model_output = self._model_output_from_clean_estimate(
                trajectory,
                clean_estimate,
                timestep,
            )
            trajectory = self._scheduler_step(
                model_output,
                timestep,
                trajectory,
                generator=generator,
                eta=self.jm2d_eta,
                **kwargs,
            )
            trajectory[condition_mask] = condition_data[condition_mask]

            ess_per_step.append(1.0 / weights.pow(2).sum(dim=-1).clamp_min(1e-12))
            energy_per_step.append(energy.min(dim=-1).values)
            clearance_per_step.append(min_clearance.max(dim=-1).values)
            ik_pose_success_per_step.append(
                ik_success.float().mean(dim=(1, 2))
            )
            ik_trajectory_success_per_step.append(
                ik_success.all(dim=-1).float().mean(dim=-1)
            )

        self._last_jm2d_stats = {
            "effective_sample_size": torch.stack(ess_per_step, dim=1).detach(),
            "best_collision_energy": torch.stack(energy_per_step, dim=1).detach(),
            "best_min_clearance": torch.stack(clearance_per_step, dim=1).detach(),
            "ik_pose_success_rate": torch.stack(
                ik_pose_success_per_step, dim=1
            ).detach(),
            "ik_trajectory_success_rate": torch.stack(
                ik_trajectory_success_per_step, dim=1
            ).detach(),
        }

        # JM2D explicitly recommends one final model-based correction for a
        # residual infeasible sample. The shared helper performs batched IK,
        # applies the same one-shot CBF-QP baseline, and stores _last_joint_traj.
        return self._post_hoc_cbf(
            cart_n_sample=trajectory,
            condition_mask=condition_mask,
            chunk_start_pose=chunk_start_pose,
            obstacle_info=obstacle_info,
            current_joint_angles=current_joint_angles,
        )

    def predict_action(
        self,
        obs_dict: Dict[str, torch.Tensor],
        fixed_action_prefix: torch.Tensor = None,
        env_batched: bool = False,
        chunk_start_pose: torch.Tensor = None,
        obstacle_info=None,
        current_joint_angles: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        assert "past_action" not in obs_dict

        nobs = self.normalizer.normalize(obs_dict)
        batch_size = next(iter(nobs.values())).shape[0]
        if env_batched:
            env_batch_size = next(iter(nobs.values())).shape[1]
            nobs = dict_apply(
                nobs,
                lambda x: x.reshape(batch_size * env_batch_size, *x.shape[2:]),
            )
        global_cond = self.obs_encoder(nobs)

        actual_batch_size = (
            batch_size * env_batch_size if env_batched else batch_size
        )
        cond_data = torch.zeros(
            (actual_batch_size, self.action_horizon, self.action_dim),
            device=self.device,
            dtype=self.dtype,
        )
        cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)
        if fixed_action_prefix is not None and self.inpaint_fixed_action_prefix:
            n_fixed_steps = fixed_action_prefix.shape[1]
            cond_data[:, :n_fixed_steps] = fixed_action_prefix
            cond_mask[:, :n_fixed_steps] = True
            cond_data = self.normalizer["action"].normalize(cond_data)

        nsample, _ = self.conditional_sample(
            condition_data=cond_data,
            condition_mask=cond_mask,
            local_cond=None,
            global_cond=global_cond,
            chunk_start_pose=chunk_start_pose,
            obstacle_info=obstacle_info,
            current_joint_angles=current_joint_angles,
            **self.kwargs,
        )
        action_pred = self.normalizer["action"].unnormalize(nsample)
        joint_action_pred = self._joint_traj_to_env_action(self._last_joint_traj)

        if env_batched:
            action_pred = action_pred.reshape(
                batch_size,
                env_batch_size,
                self.action_horizon,
                self.action_dim,
            )
            joint_action_pred = joint_action_pred.reshape(
                batch_size,
                env_batch_size,
                self.action_horizon,
                self.arm_dof + 1,
            )

        return {
            "action": action_pred,
            "action_pred": action_pred,
            "joint_action": joint_action_pred,
            "joint_action_pred": joint_action_pred,
        }


__all__ = ["DiffusionUnetTimmPolicyJM2D"]
