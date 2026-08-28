from typing import Dict, Optional, Tuple, List, Any
import os
import time

import torch
import torch.nn.functional as F

from curobo.geom.sdf.world import CollisionQueryBuffer, WorldCollisionConfig
from curobo.geom.sdf.utils import create_collision_checker
from curobo.geom.types import WorldConfig, Cuboid
from curobo.util_file import get_robot_configs_path

from diffusion_policy.common.pytorch_util import dict_apply

# Enable Dynamo tracing of data-dependent shape operators (same as joint_space).
torch._dynamo.config.capture_dynamic_output_shape_ops = True

from embodisteer.guidance.diffusion import (
    get_guidance_strength,
    flatten_obstacle_info,
    get_pred_x0,
)
from .legacy_joint_space import DiffusionUnetTimmPolicyJointSpace

from embodisteer.kinematics.rotation import (
    axis_angle_to_matrix,
    matrix_to_axis_angle,
    matrix_to_rotation_6d as matrix_to_rot6d,
    quaternion_to_matrix,
    rotation_6d_to_matrix as rot6d_to_matrix,
)


class DiffusionUnetTimmPolicyJointSpaceWithGuidance(DiffusionUnetTimmPolicyJointSpace):
    """
    Joint-space diffusion with whole-body collision guidance.

    Supported guidance implementations:
      - "cbf" (default): batched CBF-QP in joint space.
      - "gd": gradient-descent guidance (same behavior as main branch).
    """

    def __init__(
        self,
        *args,
        robot_uid: Optional[str] = None,
        robot_cfg_name: Optional[str] = None,
        robot_urdf_path: Optional[str] = None,
        ee_link_name: Optional[str] = None,
        arm_dof: int = -1,
        guidance_scale: float = 1.0,
        guidance_safety_margin: float = 0.01,
        guidance_activation_distance: float = 0.02,
        guidance_grad_clip: float = 1.0,
        guidance_loss_power: float = 2.0,
        guidance_use_schedule: bool = True,
        guidance_apply_last_step_only: bool = False,
        guidance_steps_per_denoise: int = 1,
        guidance_use_clean_sample: bool = False,
        guidance_cbf_lambda: float = 0.01,
        guidance_method: str = "cbf",
        guidance_sdf_agg: str = "topk",
        guidance_sdf_softmax_temp: float = 20.0,
        guidance_sdf_topk: int = 4,
        guidance_task_pos_weight: float = 1.0,
        guidance_task_rot_weight: float = 1.0,
        guidance_reuse_jacobian: bool = True,
        **kwargs,
    ):
        if robot_cfg_name is None:
            robot_cfg_name = self._infer_robot_cfg_name(robot_uid)

        super().__init__(
            *args,
            robot_cfg_name=robot_cfg_name,
            robot_urdf_path=robot_urdf_path,
            ee_link_name=ee_link_name,
            arm_dof=arm_dof,
            **kwargs,
        )

        self.robot_uid = robot_uid
        self.guidance_scale = float(guidance_scale)
        self.guidance_safety_margin = float(guidance_safety_margin)
        self.guidance_activation_distance = float(guidance_activation_distance)
        self.guidance_grad_clip = float(guidance_grad_clip)
        self.guidance_loss_power = float(guidance_loss_power)
        self.guidance_use_schedule = bool(guidance_use_schedule)
        self.guidance_apply_last_step_only = bool(guidance_apply_last_step_only)
        self.guidance_steps_per_denoise = int(max(guidance_steps_per_denoise, 1))
        self.guidance_use_clean_sample = bool(guidance_use_clean_sample)
        self.guidance_cbf_lambda = float(guidance_cbf_lambda)
        guidance_sdf_agg = str(guidance_sdf_agg).lower()
        # Keep "softmax" as a compatibility alias for this branch,
        # but implement it with a top-k normalized smooth max to avoid the
        # large positive bias of unnormalized logsumexp over all spheres.
        if guidance_sdf_agg == "softmax":
            guidance_sdf_agg = "topk"
        if guidance_sdf_agg not in ("max", "topk"):
            raise ValueError(
                f"Unsupported guidance_sdf_agg={guidance_sdf_agg}. Use one of: ['max', 'topk']"
            )
        self.guidance_sdf_agg = guidance_sdf_agg
        self.guidance_sdf_softmax_temp = float(guidance_sdf_softmax_temp)
        self.guidance_sdf_topk = int(max(guidance_sdf_topk, 1))
        self.guidance_task_pos_weight = float(guidance_task_pos_weight)
        self.guidance_task_rot_weight = float(guidance_task_rot_weight)
        guidance_method = str(guidance_method).lower()
        if guidance_method in ("gradient_descent", "gd"):
            guidance_method = "gd"
        elif guidance_method == "cbf":
            guidance_method = "cbf"
        else:
            raise ValueError(
                f"Unsupported guidance_method={guidance_method}. "
                "Use one of: ['cbf', 'gd', 'gradient_descent']"
            )
        self.guidance_method = guidance_method
        self.guidance_reuse_jacobian = bool(guidance_reuse_jacobian)

        self._world_collision = None
        self._coll_query_buffer = None
        self._coll_weight = None
        self._coll_activation_distance = None
        self._cached_world_key = None
        self._coll_env_query_idx = None

    @staticmethod
    def _infer_robot_cfg_name(robot_uid: Optional[str]) -> str:
        if robot_uid is None:
            return "panda_robotiq_wristcam.yml"

        mapping = {
            "panda_robotiq_wristcam": "panda_robotiq_wristcam.yml",
            "ur5_robotiq_wristcam": "ur5_robotiq_wristcam.yml",
            "xarm6_robotiq_wristcam": "xarm6_robotiq_wristcam.yml",
            "xarm7_robotiq_wristcam": "xarm7_robotiq_wristcam.yml",
            "floating_robotiq_2f_85_gripper_wristcam": "floating_robotiq_wristcam.yml",
            "floating_robotiq_wristcam": "floating_robotiq_wristcam.yml",
        }
        if robot_uid in mapping:
            return mapping[robot_uid]

        candidate = f"{robot_uid}.yml"
        cfg_path = os.path.join(get_robot_configs_path(), candidate)
        if os.path.exists(cfg_path):
            return candidate
        raise ValueError(f"Cannot infer cuRobo robot config for robot_uid={robot_uid}")

    def _ensure_kinematics(self, device: torch.device):
        super()._ensure_kinematics(device)
        self._compile_cbf_functions(device)

    def _compile_cbf_functions(self, device: torch.device):
        """Compile CBF QP solver (the only CBF hot-path function safe for compile)."""
        if getattr(self, "_compile_cbf_done", False):
            return
        self._compile_cbf_done = True

        robot_dof = int(self._robot_dof)
        arm_dof = min(max(int(self.arm_dof), 1), robot_dof)
        bsz, horizon = 1, 16

        dummy_jac = torch.randn(bsz, horizon, 6, robot_dof, device=device, dtype=torch.float32)
        dummy_grad_h = torch.randn(bsz, horizon, robot_dof, device=device, dtype=torch.float32)
        dummy_h = torch.randn(bsz, horizon, device=device, dtype=torch.float32)

        def _cbf_qp_wrapper(jac, grad_h, h_value, constraint_scale):
            return self._solve_batched_cbf_qp(
                jac_pos=jac,
                grad_h=grad_h,
                h_value=h_value,
                constraint_scale=constraint_scale,
            )

        compiled_qp = torch.compile(
            _cbf_qp_wrapper, fullgraph=True, mode="reduce-overhead",
        )
        dummy_scale = torch.tensor(1.0, device=device, dtype=torch.float32)
        _ = compiled_qp(dummy_jac, dummy_grad_h, dummy_h, dummy_scale)  # trigger compilation
        self._cbf_qp_fn = compiled_qp

    def _build_world_collision(
        self,
        obstacle_info: Any,
        device: torch.device,
        dtype: torch.dtype,
    ):
        if obstacle_info is None:
            obstacles = []
        else:
            obstacles = flatten_obstacle_info(obstacle_info)
        if len(obstacles) == 0:
            self._world_collision = None
            self._coll_query_buffer = None
            self._cached_world_key = None
            self._coll_env_query_idx = None
            return

        # Build one cuRobo world per parallel environment. A previous version
        # merged all env obstacles into a single world, which made each batch
        # sample avoid obstacles from other randomized envs.
        n_env = 1
        normalized_obstacles = []
        world_key_items = []
        for obs in obstacles:
            center = obs["center"]
            quat = obs["quat"]
            extent = obs["extent"]

            if center.ndim == 1:
                center = center.unsqueeze(0)
            if quat.ndim == 1:
                quat = quat.unsqueeze(0)
            if extent.ndim == 1:
                extent = extent.unsqueeze(0)

            center = center.to(device=device, dtype=dtype)
            quat = quat.to(device=device, dtype=dtype)
            extent = extent.to(device=device, dtype=dtype)

            n_env = max(n_env, int(center.shape[0]), int(quat.shape[0]), int(extent.shape[0]))
            normalized_obstacles.append((center, quat, extent))

        world_cuboids_by_env = [[] for _ in range(n_env)]
        world_key_items = []
        for i, (center, quat, extent) in enumerate(normalized_obstacles):
            for j in range(n_env):
                c = center[j if center.shape[0] > 1 else 0]
                q = quat[j if quat.shape[0] > 1 else 0]
                e = extent[j if extent.shape[0] > 1 else 0]
                world_key_items.append(torch.cat([c, q, e], dim=0))

                world_cuboids_by_env[j].append(
                    Cuboid(
                        name=f"obs_{i}_{j}",
                        pose=[
                            float(c[0].item()),
                            float(c[1].item()),
                            float(c[2].item()),
                            float(q[0].item()),
                            float(q[1].item()),
                            float(q[2].item()),
                            float(q[3].item()),
                        ],
                        dims=[
                            float(2.0 * e[0].item()),
                            float(2.0 * e[1].item()),
                            float(2.0 * e[2].item()),
                        ],
                    )
                )

        world_key = torch.cat(world_key_items, dim=0)
        if (
            self._cached_world_key is not None
            and self._cached_world_key.shape == world_key.shape
            and torch.allclose(self._cached_world_key, world_key, atol=1e-6, rtol=0.0)
        ):
            self._coll_env_query_idx = torch.arange(
                n_env, device=device, dtype=torch.int32
            )
            return

        world_config = [WorldConfig(cuboid=cuboids) for cuboids in world_cuboids_by_env]
        world_coll_config = WorldCollisionConfig(
            tensor_args=self._tensor_args,
            world_model=world_config,
        )
        self._world_collision = create_collision_checker(world_coll_config)
        self._coll_query_buffer = CollisionQueryBuffer()
        self._coll_weight = self._tensor_args.to_device(torch.tensor([1.0], dtype=dtype))
        self._coll_activation_distance = self._tensor_args.to_device(
            torch.tensor([self.guidance_activation_distance], dtype=dtype)
        )
        self._cached_world_key = world_key.detach().clone()
        self._coll_env_query_idx = torch.arange(n_env, device=device, dtype=torch.int32)

    def _env_query_idx_for_batch(self, batch_size: int, device: torch.device) -> Optional[torch.Tensor]:
        if self._coll_env_query_idx is None:
            return None
        if int(self._coll_env_query_idx.numel()) == batch_size:
            return self._coll_env_query_idx.to(device=device)
        if int(self._coll_env_query_idx.numel()) == 1:
            return torch.zeros(batch_size, device=device, dtype=torch.int32)
        raise ValueError(
            f"Collision world has {self._coll_env_query_idx.numel()} envs, "
            f"but query batch has size {batch_size}."
        )

    def _collision_penalty(self, dist: torch.Tensor) -> torch.Tensor:
        penetration = dist + self.guidance_safety_margin
        penalty = F.relu(penetration)
        if self.guidance_loss_power != 1.0:
            penalty = penalty.pow(self.guidance_loss_power)
        return penalty

    def _guidance_scale_at(self, idx: int, n_steps: int, t: torch.Tensor, dtype, device):
        scale = torch.tensor(self.guidance_scale, device=device, dtype=dtype)
        if not self.guidance_use_schedule:
            return scale

        # Guidance should be stronger at late denoising steps, when samples are
        # close to the data manifold and less likely to be washed out.
        # get_guidance_strength() is monotonic in k where small k -> strong,
        # thus we map idx (0->start) to k (large->start, small->end).
        if n_steps <= 1:
            k = torch.tensor(0.0, device=device, dtype=dtype)
            n = torch.tensor(1.0, device=device, dtype=dtype)
        else:
            k = torch.tensor(float((n_steps - 1) - idx), device=device, dtype=dtype)
            n = torch.tensor(float(n_steps - 1), device=device, dtype=dtype)
        return scale * get_guidance_strength(k, n)

    @staticmethod
    def _curobo_signed_distance_to_cbf_h(dist_signed: torch.Tensor) -> torch.Tensor:
        """
        Map cuRobo signed distance to CBF safety function h(q).

        cuRobo ESDF convention:
          - positive: inside obstacle (unsafe)
          - negative: outside obstacle (safe)
        CBF convention used here:
          - h >= 0: safe

        Therefore: h = -dist_signed
        """
        return -dist_signed

    def _query_worst_signed_distance(self, q_arm: torch.Tensor) -> torch.Tensor:
        """
        Returns aggregated cuRobo signed distance over robot collision spheres.
        Output shape: (B, T)
        """
        B, T, D = q_arm.shape
        q_flat = q_arm.reshape(B * T, D).contiguous()
        kin_state = self._kin_model.get_state(q_flat)
        spheres = kin_state.link_spheres_tensor.reshape(B, T, -1, 4)

        self._coll_query_buffer.update_buffer_shape(
            spheres.shape,
            self._tensor_args,
            self._world_collision.collision_types,
        )
        dist = self._world_collision.get_sphere_distance(
            spheres,
            self._coll_query_buffer,
            self._coll_weight,
            self._coll_activation_distance,
            env_query_idx=self._env_query_idx_for_batch(B, spheres.device),
            return_loss=True,
            compute_esdf=True,
        )
        return self._aggregate_signed_distance(dist)

    def _aggregate_signed_distance(self, dist: torch.Tensor) -> torch.Tensor:
        """
        Aggregate per-sphere signed distances into one value per (B, T).

        cuRobo sign convention here is positive inside obstacle, negative outside.
        """
        if self.guidance_sdf_agg == "max":
            return dist.max(dim=-1).values

        k = min(self.guidance_sdf_topk, int(dist.shape[-1]))
        topk = torch.topk(dist, k=k, dim=-1).values
        max_val = topk[..., 0]
        if k == 1:
            return max_val

        temp = torch.tensor(
            max(self.guidance_sdf_softmax_temp, 1e-6),
            device=dist.device,
            dtype=dist.dtype,
        )
        shifted = topk - max_val.unsqueeze(-1)
        # Normalized smooth max over only the top-k spheres. This stays <= max,
        # so it does not create the large positive bias of raw logsumexp.
        smooth_topk = max_val + torch.log(torch.exp(shifted * temp).mean(dim=-1)) / temp
        # If any of the top-k spheres is already penetrating the obstacle
        # (max_val > 0), preserve that positive sign exactly instead of
        # averaging it away with nearby negative spheres.
        # return torch.where(max_val > 0, max_val, smooth_topk)
        return smooth_topk

    def _compute_cbf_linearization(
        self, q_arm: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute linearized CBF terms at q_arm:
          h(q^o), ∇h(q^o), and cuRobo worst signed distance.
        """
        if self._world_collision is None:
            zq = torch.zeros_like(q_arm)
            zh = torch.zeros(q_arm.shape[:2], device=q_arm.device, dtype=q_arm.dtype)
            return zh, zq, zh

        with torch.enable_grad():
            q_req = q_arm.detach().clone().requires_grad_(True)
            dist_worst = self._query_worst_signed_distance(q_req)
            h = self._curobo_signed_distance_to_cbf_h(dist_worst)
            grad_h = torch.autograd.grad(h.sum(), q_req, allow_unused=True)[0]
            if grad_h is None:
                grad_h = torch.zeros_like(q_req)
            return h.detach(), grad_h.detach(), dist_worst.detach()

    def _solve_batched_cbf_qp(
        self,
        jac_pos: torch.Tensor,
        grad_h: torch.Tensor,
        h_value: torch.Tensor,
        constraint_scale: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Solve batched single-inequality QP in closed form.

        QP:
          min 1/2 * dq^T H dq, with H = J_task^T W J_task + λI
          s.t. grad_h^T dq >= d_safe - h

        jac_pos is kept as the argument name for compatibility. It may be a
        3D position Jacobian or a full 6D geometric Jacobian. With a 6D input,
        the rotational rows are weighted by guidance_task_rot_weight so CBF
        cannot satisfy avoidance mainly by rotating the TCP/contact frame.
        """
        B, T, task_dim, D = jac_pos.shape
        N = B * T

        # Solve CBF-QP only in controllable arm subspace.
        # q includes extra gripper/coupler joints from cuRobo kinematics, but
        # the policy action only controls the arm DOFs (plus one scalar gripper
        # command outside this q-space). If unconstrained tail joints are kept
        # in the QP variable, they can absorb constraint satisfaction and dilute
        # effective arm correction.
        ctrl_dof = int(min(max(self.arm_dof, 1), D))

        J = jac_pos[..., :ctrl_dof].reshape(N, task_dim, ctrl_dof)
        a = grad_h[..., :ctrl_dof].reshape(N, ctrl_dof)
        h_flat = h_value.reshape(N)

        if constraint_scale.ndim == 0:
            c_scale = constraint_scale
        else:
            c_scale = constraint_scale.reshape(-1)[0]
        rhs = (self.guidance_safety_margin - h_flat) * c_scale
        rhs = torch.clamp(rhs, min=0.0)

        eye = torch.eye(ctrl_dof, device=J.device, dtype=J.dtype).unsqueeze(0)
        if task_dim >= 6 and self.guidance_task_rot_weight > 0.0:
            weights = torch.ones((task_dim,), device=J.device, dtype=J.dtype)
            weights[:3] = self.guidance_task_pos_weight
            weights[3:6] = self.guidance_task_rot_weight
        else:
            weights = torch.ones((task_dim,), device=J.device, dtype=J.dtype)
            weights[: min(3, task_dim)] = self.guidance_task_pos_weight
        H = (J * weights.view(1, task_dim, 1)).transpose(-2, -1) @ J
        H = H + self.guidance_cbf_lambda * eye

        a_col = a.unsqueeze(-1)
        H_inv_a = torch.linalg.solve(H, a_col).squeeze(-1)
        denom = (a * H_inv_a).sum(dim=-1)

        eps = torch.tensor(1e-9, device=denom.device, dtype=denom.dtype)
        feasible = denom > eps
        alpha = torch.zeros_like(rhs)
        alpha[feasible] = rhs[feasible] / denom[feasible]

        dq_ctrl = H_inv_a * alpha.unsqueeze(-1)
        if ctrl_dof == D:
            dq = dq_ctrl
        else:
            dq = torch.zeros((N, D), device=dq_ctrl.device, dtype=dq_ctrl.dtype)
            dq[:, :ctrl_dof] = dq_ctrl

        return (
            dq.reshape(B, T, D),
            rhs.reshape(B, T),
            denom.reshape(B, T),
            feasible.reshape(B, T),
        )

    def _compute_collision_grad(self, q_arm: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if self._world_collision is None:
            z = torch.zeros_like(q_arm)
            return z, torch.zeros((), device=q_arm.device, dtype=q_arm.dtype)

        B, T, D = q_arm.shape
        with torch.enable_grad():
            q_req = q_arm.detach().clone().requires_grad_(True)
            q_flat = q_req.reshape(B * T, D).contiguous()
            kin_state = self._kin_model.get_state(q_flat)
            spheres = kin_state.link_spheres_tensor.reshape(B, T, -1, 4)

            self._coll_query_buffer.update_buffer_shape(
                spheres.shape,
                self._tensor_args,
                self._world_collision.collision_types,
            )
            dist = self._world_collision.get_sphere_distance(
                spheres,
                self._coll_query_buffer,
                self._coll_weight,
                self._coll_activation_distance,
                env_query_idx=self._env_query_idx_for_batch(B, spheres.device),
                return_loss=False,
                compute_esdf=True,
            )
            # Worst-case aggregation for each parallel environment sample:
            # cuRobo ESDF sign convention here is positive inside obstacle,
            # negative outside; therefore "closest" in physical sense is the
            # maximum signed distance value.
            dist_agg = self._aggregate_signed_distance(dist)
            penalty = self._collision_penalty(dist_agg)

            # Batch dimension is only for parallelism; do not average across B.
            # Summation preserves independent per-sample gradients without
            # introducing 1/B scaling.
            loss = penalty.sum().sum()
            grad = torch.autograd.grad(loss, q_req, allow_unused=True)[0]
            if grad is None:
                grad = torch.zeros_like(q_req)
            return grad.detach(), loss.detach()

    def _estimate_clean_joint_from_cartesian(
        self,
        q_arm_ref: torch.Tensor,
        cart_phys_clean: torch.Tensor,
        chunk_start_pose: torch.Tensor,
    ) -> torch.Tensor:
        """
        Approximate clean joint sample using one Jacobian linearization step.
        This keeps one-step efficiency while avoiding guidance on very noisy q.
        """
        bsz, horizon, _ = q_arm_ref.shape

        # current absolute pose from reference joint trajectory
        abs_pos_ref, abs_rot_ref = self._fk_to_absolute(q_arm_ref)
        abs_rot6d_ref = matrix_to_rot6d(abs_rot_ref.reshape(-1, 3, 3)).reshape(bsz, horizon, 6)
        abs_pose9_ref = torch.cat([abs_pos_ref, abs_rot6d_ref], dim=-1)

        # target absolute pose from predicted clean Cartesian sample
        abs_pos_clean, abs_rot_clean = self._relative_pose9_to_absolute(
            cart_phys_clean[..., :9], chunk_start_pose
        )
        abs_rot6d_clean = matrix_to_rot6d(abs_rot_clean.reshape(-1, 3, 3)).reshape(bsz, horizon, 6)
        abs_pose9_clean = torch.cat([abs_pos_clean, abs_rot6d_clean], dim=-1)

        twist_to_clean = self._absolute_pose_delta_to_twist6(abs_pose9_ref, abs_pose9_clean)
        jac_ref = self._jacobian(q_arm_ref)
        dq_clean = self._dls_pinv_map(
            twist_to_clean.reshape(-1, 6), jac_ref
        ).reshape(bsz, horizon, self._robot_dof)
        if self.max_dq_per_step > 0:
            dq_clean = torch.clamp(dq_clean, -self.max_dq_per_step, self.max_dq_per_step)
        q_arm_clean = q_arm_ref + dq_clean
        return q_arm_clean

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
        return_debug: bool = False,
        **kwargs,
    ):
        if chunk_start_pose is None:
            raise ValueError("chunk_start_pose is required for joint-space inference.")

        self._ensure_kinematics(condition_data.device)
        self._build_world_collision(
            obstacle_info=obstacle_info,
            device=condition_data.device,
            dtype=condition_data.dtype,
        )

        # Pre-compute base coordinate frame (constant across denoising steps).
        base_pos = chunk_start_pose[:, :3]
        base_rot = axis_angle_to_matrix(chunk_start_pose[:, 3:6])
        base_rot_t = base_rot.transpose(-2, -1)

        model = self.model
        scheduler = self.noise_scheduler
        bsz = condition_data.shape[0]
        horizon = condition_data.shape[1]

        timing = {
            "guidance_total": 0.0,
            "guidance_grad": 0.0,
            "guidance_apply": 0.0,
        }

        if current_joint_angles is None:
            q_start = self._solve_start_joint_from_pose(chunk_start_pose)
        else:
            q_start = current_joint_angles[..., : self._robot_dof].to(
                device=condition_data.device, dtype=condition_data.dtype
            )
        q_seed = q_start.unsqueeze(1).expand(bsz, horizon, self._robot_dof)

        trajectory_cart_n = torch.randn(
            size=condition_data.shape,
            dtype=condition_data.dtype,
            device=condition_data.device,
            generator=generator,
        )
        trajectory_cart_n[condition_mask] = condition_data[condition_mask]

        trajectory_cart = self.normalizer["action"].unnormalize(trajectory_cart_n)
        grip = trajectory_cart[..., 9:10]

        if self.noise_init_mode == "jacobian_projected":
            jac_init = self._jacobian(q_seed)
            eps_twist = torch.randn(
                (bsz * horizon, 6),
                device=q_seed.device,
                dtype=q_seed.dtype,
                generator=generator,
            )
            dq_init = self._dls_pinv_map(eps_twist, jac_init)
            dq_init = dq_init.reshape(bsz, horizon, self._robot_dof)
            if self.max_dq_per_step > 0:
                dq_init = torch.clamp(dq_init, -self.max_dq_per_step, self.max_dq_per_step)
            q_arm = q_seed + dq_init * self.jac_noise_alpha
        elif self.noise_init_mode == "jacobian_diagonal":
            jac_init = self._jacobian(q_seed)
            jt = jac_init.transpose(-2, -1)
            jjt = jac_init @ jt
            eye6 = torch.eye(6, device=jac_init.device, dtype=jac_init.dtype).unsqueeze(0)
            Jpinv = torch.linalg.solve(jjt + self.jacobian_damping * eye6, jac_init).transpose(-2, -1)
            Sigma_q = Jpinv @ Jpinv.transpose(-2, -1)
            per_joint_std = torch.sqrt(torch.clamp(torch.diagonal(Sigma_q, dim1=-2, dim2=-1), min=1e-8))
            mean_std = per_joint_std.mean(dim=-1, keepdim=True)
            per_joint_std = per_joint_std / mean_std * self.jac_noise_alpha
            per_joint_std = per_joint_std.reshape(bsz, horizon, self._robot_dof)
            q_arm = q_seed + torch.randn(
                q_seed.shape,
                device=q_seed.device,
                dtype=q_seed.dtype,
                generator=generator,
            ) * per_joint_std
        else:
            q_arm = q_seed + torch.randn(
                q_seed.shape,
                device=q_seed.device,
                dtype=q_seed.dtype,
                generator=generator,
            ) * self.init_noise_scale

        q_traj = torch.cat([q_arm, grip], dim=-1)

        cond_step_mask = condition_mask.any(dim=-1)
        q_cond = None
        if torch.any(cond_step_mask):
            cond_cart = self.normalizer["action"].unnormalize(condition_data)
            # Use cached base_rot for rel→abs conversion.
            _rel9 = cond_cart[..., :9]
            _rel_pos = _rel9[..., :3]
            _rel_rot = rot6d_to_matrix(_rel9[..., 3:])
            cond_abs_pos = (base_rot.unsqueeze(1)
                            @ _rel_pos.unsqueeze(-1)).squeeze(-1) + base_pos.unsqueeze(1)
            cond_abs_rot = base_rot.unsqueeze(1) @ _rel_rot
            q_cond_arm = self._ik_from_absolute(cond_abs_pos, cond_abs_rot, q_seed)
            q_cond = torch.cat([q_cond_arm, cond_cart[..., 9:10]], dim=-1)
            q_traj[cond_step_mask] = q_cond[cond_step_mask]

        scheduler.set_timesteps(self.num_inference_steps)
        debug = {
            "step_cart_l2": [],
            "guidance_loss": [],
            "guidance_grad_norm": [],
            "clean_sample_cart_err_before": [],
            "clean_sample_cart_err_after": [],
        }
        timesteps = list(scheduler.timesteps)
        n_steps = len(timesteps)

        for idx, t in enumerate(timesteps):
            if q_cond is not None:
                q_traj[cond_step_mask] = q_cond[cond_step_mask]

            q_arm_curr = q_traj[..., : self._robot_dof]
            grip_curr = q_traj[..., self._robot_dof : self._robot_dof + 1]
            abs_pos_curr, abs_rot_curr = self._fk_to_absolute(q_arm_curr)

            # ── Relative Cartesian for U-Net (cached base_rot) ──────────
            rel_pos_curr = (base_rot_t.unsqueeze(1)
                            @ (abs_pos_curr - base_pos.unsqueeze(1)).unsqueeze(-1)
                           ).squeeze(-1)
            rel_rot_curr = base_rot_t.unsqueeze(1) @ abs_rot_curr
            rel_rot6d_curr = matrix_to_rot6d(
                rel_rot_curr.reshape(-1, 3, 3)
            ).reshape(bsz, horizon, 6)
            cart_phys_curr = torch.cat(
                [rel_pos_curr, rel_rot6d_curr, grip_curr], dim=-1
            )
            cart_n_curr = self.normalizer["action"].normalize(cart_phys_curr)
            cart_n_curr[condition_mask] = condition_data[condition_mask]

            eps_cart_n = model(cart_n_curr, t, local_cond=local_cond, global_cond=global_cond)

            # Predicted clean Cartesian sample x0 in normalized action space.
            pred_type = self.noise_scheduler.config.prediction_type
            if pred_type == "epsilon":
                alpha_prod_t = self.noise_scheduler.alphas_cumprod[t]
                alpha_prod_t = alpha_prod_t.to(device=cart_n_curr.device, dtype=cart_n_curr.dtype)
                cart_n_clean = get_pred_x0(eps_cart_n, cart_n_curr, alpha_prod_t)
            elif pred_type == "sample":
                cart_n_clean = eps_cart_n
            else:
                cart_n_clean = cart_n_curr
            cart_n_clean[condition_mask] = condition_data[condition_mask]
            cart_phys_clean = self.normalizer["action"].unnormalize(cart_n_clean)

            cart_n_prev_tgt = scheduler.step(
                eps_cart_n, t, cart_n_curr, generator=generator, **kwargs
            ).prev_sample
            cart_n_prev_tgt[condition_mask] = condition_data[condition_mask]
            cart_phys_prev_tgt = self.normalizer["action"].unnormalize(cart_n_prev_tgt)

            # ── Target absolute pose (cached base_rot) ──────────────────
            rel9_tgt = cart_phys_prev_tgt[..., :9]
            rel_pos_tgt = rel9_tgt[..., :3]
            rel_rot_tgt = rot6d_to_matrix(rel9_tgt[..., 3:])
            abs_pos_tgt = (base_rot.unsqueeze(1)
                           @ rel_pos_tgt.unsqueeze(-1)).squeeze(-1) + base_pos.unsqueeze(1)
            abs_rot_tgt = base_rot.unsqueeze(1) @ rel_rot_tgt

            # ── World-frame twist directly from matrices (no rot6d round-trip)
            twist6 = self._twist_fn(
                abs_pos_curr, abs_rot_curr, abs_pos_tgt, abs_rot_tgt
            ).clone()

            jac = self._jacobian(q_arm_curr)
            # Cache Jacobian and reference state for possible CBF reuse.
            # q_arm_curr is used as the reference because jac = J(q_arm_curr).
            jac_ref = jac
            q_ref_for_jac = q_arm_curr
            dq = self._dls_pinv_map(twist6.reshape(-1, 6), jac).reshape(bsz, horizon, self._robot_dof)
            if self.max_dq_per_step > 0:
                dq = torch.clamp(dq, -self.max_dq_per_step, self.max_dq_per_step)
            q_arm_new = q_arm_curr + dq

            # for _ in range(2):
            #     abs_pos_ri, abs_rot_ri = self._fk_to_absolute(q_arm_new)
            #     abs_rot6d_ri = matrix_to_rot6d(abs_rot_ri.reshape(-1, 3, 3)).reshape(bsz, horizon, 6)
            #     abs_pose9_ri = torch.cat([abs_pos_ri, abs_rot6d_ri], dim=-1)
            #     twist_ri = self._absolute_pose_delta_to_twist6(abs_pose9_ri, abs_pose9_tgt)
            #     dq_ri = self._dls_pinv_map(twist_ri.reshape(-1, 6), jac).reshape(bsz, horizon, self._robot_dof)
            #     if self.max_dq_per_step > 0:
            #         dq_ri = torch.clamp(dq_ri, -self.max_dq_per_step, self.max_dq_per_step)
            #     q_arm_new = q_arm_new + dq_ri

            is_last = idx == (n_steps - 1)
            if self.ik_refine_each_step or (is_last and self.ik_refine_last_step):
                q_arm_new = self._ik_from_absolute(abs_pos_tgt, abs_rot_tgt, q_arm_new)

            apply_guidance = (self._world_collision is not None)
            if self.guidance_apply_last_step_only and not is_last:
                apply_guidance = False

            if apply_guidance:
                t0 = time.perf_counter()

                if self.guidance_use_clean_sample:
                    q_guidance_state = self._estimate_clean_joint_from_cartesian(
                        q_arm_ref=q_arm_new,
                        cart_phys_clean=cart_phys_clean,
                        chunk_start_pose=chunk_start_pose,
                    )
                else:
                    q_guidance_state = q_arm_new

                q_before_guidance = q_arm_new.clone()
                for _ in range(self.guidance_steps_per_denoise):
                    if self.guidance_method == "cbf":
                        tg = time.perf_counter()
                        h_value, grad_h, _dist_signed = self._compute_cbf_linearization(q_guidance_state)
                        # Reuse the denoising-step Jacobian when the
                        # guidance query state is close to the reference
                        # state, avoiding a redundant ~9 ms computation.
                        if (
                            self.guidance_reuse_jacobian
                            and not self.guidance_use_clean_sample
                            and q_ref_for_jac is not None
                        ):
                            state_change = (q_guidance_state - q_ref_for_jac).abs().max()
                            if state_change < 0.5:
                                jac_lin = jac_ref.reshape(bsz, horizon, 6, self._robot_dof)
                            else:
                                jac_lin = self._jacobian(q_guidance_state).reshape(
                                    bsz, horizon, 6, self._robot_dof
                                )
                        else:
                            jac_lin = self._jacobian(q_guidance_state).reshape(
                                bsz, horizon, 6, self._robot_dof
                            )
                        gamma = self._guidance_scale_at(
                            idx=idx,
                            n_steps=n_steps,
                            t=t,
                            dtype=q_arm_new.dtype,
                            device=q_arm_new.device,
                        )
                        dq_cbf, _rhs, _denom, _feasible = self._cbf_qp_fn(
                            jac_lin, grad_h, h_value, gamma,
                        )
                        timing["guidance_grad"] += time.perf_counter() - tg

                        # Do not perturb conditioned prefix steps.
                        if torch.any(cond_step_mask):
                            dq_cbf = dq_cbf.clone()
                            dq_cbf[cond_step_mask] = 0.0

                        ta = time.perf_counter()
                        if self.guidance_grad_clip > 0:
                            dq_cbf = torch.clamp(
                                dq_cbf,
                                min=-self.guidance_grad_clip,
                                max=self.guidance_grad_clip,
                            )

                        q_arm_new = q_arm_new + dq_cbf
                        # Keep guidance linearization state synchronized.
                        q_guidance_state = q_guidance_state + dq_cbf
                        timing["guidance_apply"] += time.perf_counter() - ta

                        if return_debug:
                            cbf_violation = torch.relu(self.guidance_safety_margin - h_value)
                            guide_loss = cbf_violation.sum()
                            debug["guidance_loss"].append(guide_loss.detach().cpu())
                            debug["guidance_grad_norm"].append(
                                torch.linalg.norm(grad_h.reshape(bsz, -1), dim=-1).detach().cpu()
                            )
                    else:
                        tg = time.perf_counter()
                        grad, guide_loss = self._compute_collision_grad(q_guidance_state)
                        timing["guidance_grad"] += time.perf_counter() - tg

                        # Do not perturb conditioned prefix steps.
                        if torch.any(cond_step_mask):
                            grad = grad.clone()
                            grad[cond_step_mask] = 0.0

                        ta = time.perf_counter()
                        if self.guidance_grad_clip > 0:
                            grad = torch.clamp(
                                grad,
                                min=-self.guidance_grad_clip,
                                max=self.guidance_grad_clip,
                            )

                        gamma = self._guidance_scale_at(
                            idx=idx,
                            n_steps=n_steps,
                            t=t,
                            dtype=q_arm_new.dtype,
                            device=q_arm_new.device,
                        )
                        q_arm_new = q_arm_new - gamma * grad
                        # Keep guidance linearization state synchronized.
                        q_guidance_state = q_guidance_state - gamma * grad
                        timing["guidance_apply"] += time.perf_counter() - ta

                        if return_debug:
                            debug["guidance_loss"].append(guide_loss.detach().cpu())
                            debug["guidance_grad_norm"].append(
                                torch.linalg.norm(grad.reshape(bsz, -1), dim=-1).detach().cpu()
                            )
                timing["guidance_total"] += time.perf_counter() - t0

                if return_debug and self.guidance_use_clean_sample:
                    abs_p_before, abs_r_before = self._fk_to_absolute(q_before_guidance)
                    rel9_before = self._absolute_pose_to_relative9(abs_p_before, abs_r_before, chunk_start_pose)
                    cart_before = torch.cat([rel9_before, cart_phys_prev_tgt[..., 9:10]], dim=-1)
                    err_before = torch.linalg.norm(
                        (cart_before[..., :9] - cart_phys_clean[..., :9]).reshape(bsz, -1), dim=-1
                    )

                    abs_p_after, abs_r_after = self._fk_to_absolute(q_arm_new)
                    rel9_after = self._absolute_pose_to_relative9(abs_p_after, abs_r_after, chunk_start_pose)
                    cart_after = torch.cat([rel9_after, cart_phys_prev_tgt[..., 9:10]], dim=-1)
                    err_after = torch.linalg.norm(
                        (cart_after[..., :9] - cart_phys_clean[..., :9]).reshape(bsz, -1), dim=-1
                    )
                    debug["clean_sample_cart_err_before"].append(err_before.detach().cpu())
                    debug["clean_sample_cart_err_after"].append(err_after.detach().cpu())

            q_traj = torch.cat([q_arm_new, cart_phys_prev_tgt[..., 9:10]], dim=-1)

            if return_debug:
                q_arm_dbg = q_traj[..., : self._robot_dof]
                abs_p, abs_r = self._fk_to_absolute(q_arm_dbg)
                rel9 = self._absolute_pose_to_relative9(abs_p, abs_r, chunk_start_pose)
                cart_phys_after = torch.cat(
                    [rel9, q_traj[..., self._robot_dof : self._robot_dof + 1]], dim=-1
                )
                debug["step_cart_l2"].append(
                    torch.linalg.norm(
                        (cart_phys_after - cart_phys_prev_tgt).reshape(bsz, -1), dim=-1
                    ).detach().cpu()
                )

        if q_cond is not None:
            q_traj[cond_step_mask] = q_cond[cond_step_mask]

        self._last_joint_traj = q_traj.detach()

        q_arm_final = q_traj[..., : self._robot_dof]
        grip_final = q_traj[..., self._robot_dof : self._robot_dof + 1]
        abs_p_f, abs_r_f = self._fk_to_absolute(q_arm_final)
        # Use cached base_rot for the final relative-pose conversion.
        rel_pos_f = (base_rot_t.unsqueeze(1)
                     @ (abs_p_f - base_pos.unsqueeze(1)).unsqueeze(-1)).squeeze(-1)
        rel_rot_f = base_rot_t.unsqueeze(1) @ abs_r_f
        rel_rot6d_f = matrix_to_rot6d(
            rel_rot_f.reshape(-1, 3, 3)
        ).reshape(bsz, horizon, 6)
        cart_final = torch.cat([rel_pos_f, rel_rot6d_f, grip_final], dim=-1)
        cart_final_n = self.normalizer["action"].normalize(cart_final)
        cart_final_n[condition_mask] = condition_data[condition_mask]

        if return_debug:
            debug["timing_guidance"] = timing
            return cart_final_n, q_traj, debug
        return cart_final_n

    def predict_action(
        self,
        obs_dict: Dict[str, torch.Tensor],
        fixed_action_prefix: torch.Tensor = None,
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
                device=self.device,
                dtype=self.dtype,
            )
        else:
            cond_data = torch.zeros(
                size=(B, self.action_horizon, self.action_dim),
                device=self.device,
                dtype=self.dtype,
            )
        cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)

        if fixed_action_prefix is not None and self.inpaint_fixed_action_prefix:
            n_fixed_steps = fixed_action_prefix.shape[1]
            cond_data[:, :n_fixed_steps] = fixed_action_prefix
            cond_mask[:, :n_fixed_steps] = True
            cond_data = self.normalizer["action"].normalize(cond_data)

        if chunk_start_pose is None:
            raise ValueError("chunk_start_pose must be provided for joint-space policy inference.")

        nsample = self.conditional_sample(
            condition_data=cond_data,
            condition_mask=cond_mask,
            local_cond=None,
            global_cond=global_cond,
            chunk_start_pose=chunk_start_pose,
            obstacle_info=obstacle_info,
            current_joint_angles=current_joint_angles,
            **self.kwargs,
        )

        if env_batched:
            assert nsample.shape == (B * env_batch_size, self.action_horizon, self.action_dim)
        else:
            assert nsample.shape == (B, self.action_horizon, self.action_dim)
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
