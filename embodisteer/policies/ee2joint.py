"""Joint-space comparison policies and shared robot runtime.

The paper method lives in :mod:`embodisteer.policies.embodisteer`.
"""

from typing import Dict, Optional, Tuple, Any

import torch
import pytorch_kinematics as pk

from curobo.types.base import TensorDeviceType
from curobo.types.robot import RobotConfig
from curobo.types.math import Pose
from curobo.wrap.reacher.ik_solver import IKSolver, IKSolverConfig
from curobo.util_file import get_assets_path, get_robot_configs_path, join_path, load_yaml

from curobo.geom.sdf.world import CollisionQueryBuffer, WorldCollisionConfig
from curobo.geom.sdf.utils import create_collision_checker
from curobo.geom.types import WorldConfig, Cuboid

from diffusion_policy.common.pytorch_util import dict_apply
from .ee_space import DiffusionUnetTimmPolicyEESpace

torch._dynamo.config.capture_dynamic_output_shape_ops = True

from embodisteer.guidance.diffusion import (
    flatten_obstacle_info,
)

from embodisteer.kinematics.rotation import (
    axis_angle_to_matrix,
    matrix_to_quaternion,
    matrix_to_rotation_6d as matrix_to_rot6d,
    quaternion_to_matrix,
    rotation_6d_to_matrix as rot6d_to_matrix,
)
from embodisteer.collision import (
    aggregate_signed_distance,
    collision_penalty,
    curobo_signed_distance_to_cbf_h,
)
from embodisteer.kinematics import (
    damped_least_squares_pinv,
    twist6_from_matrices_fast,
)
from embodisteer.kinematics.robot_config import infer_robot_cfg_name
from embodisteer.guidance import (
    guidance_scale_at,
    solve_batched_cbf_qp,
)


class _JointSpacePolicyRuntime(DiffusionUnetTimmPolicyEESpace):
    """Shared robot resources and I/O, not a selectable inference method.

    Concrete sibling policies own their samplers: JointSpace for no guidance
    or GD, EmbodiSteer for the paper's CBF method, and the post-hoc baselines.
    The CBF query/QP adapters are also used by post-hoc baselines.
    """

    def __init__(
        self,
        *args,
        robot_uid: Optional[str] = None,
        robot_cfg_name: Optional[str] = None,
        robot_urdf_path: Optional[str] = None,
        ee_link_name: Optional[str] = None,
        arm_dof: int = -1,
        ik_num_seeds: int = 1,
        jacobian_damping: float = 0.001,
        ik_position_threshold: float = 5e-4,
        ik_rotation_threshold: float = 5e-3,
        max_dq_per_step: float = 0.5,
        jac_noise_alpha: float = 0.1,
        # guidance parameters (only used when guidance_method != "")
        guidance_method: str = "",
        guidance_scale: float = 1.0,
        guidance_safety_margin: float = 0.01,
        guidance_activation_distance: float = 0.02,
        guidance_grad_clip: float = 1.0,
        guidance_loss_power: float = 2.0,
        guidance_use_schedule: bool = True,
        guidance_schedule_midpoint: float = 0.7,
        guidance_schedule_steepness: float = 50.0,
        guidance_cbf_lambda: float = 0.01,
        guidance_sdf_agg: str = "topk",
        guidance_sdf_softmax_temp: float = 20.0,
        guidance_sdf_topk: int = 4,
        guidance_task_pos_weight: float = 1.0,
        guidance_task_rot_weight: float = 1.0,
        **kwargs,
    ):
        if robot_cfg_name is None:
            robot_cfg_name = infer_robot_cfg_name(robot_uid)

        super().__init__(*args, **kwargs)

        self.robot_uid = robot_uid
        self.robot_cfg_name = robot_cfg_name
        self.robot_urdf_path = robot_urdf_path
        self.ee_link_name = ee_link_name
        self.arm_dof = int(arm_dof)
        self.ik_num_seeds = int(ik_num_seeds)
        self.jacobian_damping = float(jacobian_damping)
        self.ik_position_threshold = float(ik_position_threshold)
        self.ik_rotation_threshold = float(ik_rotation_threshold)
        self.max_dq_per_step = float(max_dq_per_step)
        self.jac_noise_alpha = float(jac_noise_alpha)

        # guidance config
        guidance_method = str(guidance_method).lower().strip()
        if guidance_method not in ("", "cbf", "gd"):
            raise ValueError(
                f"Unsupported guidance_method={guidance_method}. "
                "Use one of: ['', 'cbf', 'gd']"
            )
        self.guidance_method = guidance_method

        self.guidance_scale = float(guidance_scale)
        self.guidance_safety_margin = float(guidance_safety_margin)
        self.guidance_activation_distance = float(guidance_activation_distance)
        self.guidance_grad_clip = float(guidance_grad_clip)
        self.guidance_loss_power = float(guidance_loss_power)
        self.guidance_use_schedule = bool(guidance_use_schedule)
        self.guidance_schedule_midpoint = float(guidance_schedule_midpoint)
        self.guidance_schedule_steepness = float(guidance_schedule_steepness)
        self.guidance_cbf_lambda = float(guidance_cbf_lambda)
        guidance_sdf_agg = str(guidance_sdf_agg).lower()
        if guidance_sdf_agg not in ("max", "topk"):
            raise ValueError(f"Unsupported guidance_sdf_agg={guidance_sdf_agg}")
        self.guidance_sdf_agg = guidance_sdf_agg
        self.guidance_sdf_softmax_temp = float(guidance_sdf_softmax_temp)
        self.guidance_sdf_topk = int(max(guidance_sdf_topk, 1))
        self.guidance_task_pos_weight = float(guidance_task_pos_weight)
        self.guidance_task_rot_weight = float(guidance_task_rot_weight)

        # kinematics state (lazy init)
        self._ik_solver = None
        self._kin_model = None
        self._pk_chain = None
        self._robot_dof = None
        self._tensor_args = None
        self._last_joint_traj = None

        # collision state (lazy init, only for guidance)
        self._world_collision = None
        self._coll_query_buffer = None
        self._coll_weight = None
        self._coll_activation_distance = None
        self._cached_world_key = None
        self._coll_env_query_idx = None

    def _joint_traj_to_env_action(self, q_traj: torch.Tensor) -> torch.Tensor:
        arm_q = q_traj[..., : self.arm_dof]
        grip = q_traj[..., self._robot_dof : self._robot_dof + 1]
        return torch.cat([arm_q, grip], dim=-1)

    def _ensure_kinematics(self, device: torch.device):
        if self._ik_solver is not None:
            return

        self._tensor_args = TensorDeviceType(device=device)
        robot_cfg_dict = load_yaml(join_path(get_robot_configs_path(), self.robot_cfg_name))["robot_cfg"]
        kin_cfg = robot_cfg_dict.get("kinematics", {})
        if self.ee_link_name is None:
            self.ee_link_name = kin_cfg.get("ee_link", "eef")
        if self.robot_urdf_path is None:
            urdf_rel = kin_cfg.get("urdf_path")
            if urdf_rel is None:
                raise ValueError(f"urdf_path missing in robot config {self.robot_cfg_name}")
            self.robot_urdf_path = join_path(get_assets_path(), urdf_rel)

        robot_cfg = RobotConfig.from_dict(robot_cfg_dict, self._tensor_args)

        ik_config = IKSolverConfig.load_from_robot_config(
            robot_cfg,
            None,
            rotation_threshold=self.ik_rotation_threshold,
            position_threshold=self.ik_position_threshold,
            num_seeds=self.ik_num_seeds,
            self_collision_check=False,
            self_collision_opt=False,
            tensor_args=self._tensor_args,
            use_cuda_graph=False,
        )
        self._ik_solver = IKSolver(ik_config)
        self._kin_model = self._ik_solver.kinematics
        self._robot_dof = int(self._kin_model.get_dof())
        pk_root_link_name = kin_cfg.get("base_link", "")

        with open(self.robot_urdf_path, "r") as f:
            urdf_str = f.read()
        try:
            self._pk_chain = pk.build_serial_chain_from_urdf(
                urdf_str, self.ee_link_name, root_link_name=pk_root_link_name
            )
        except ValueError:
            self._pk_chain = pk.build_serial_chain_from_urdf(
                urdf_str.encode("utf-8"), self.ee_link_name, root_link_name=pk_root_link_name
            )
        self._pk_chain = self._pk_chain.to(dtype=torch.float32, device=device)
        if self.arm_dof <= 0:
            self.arm_dof = int(self._pk_chain.n_joints)

        self._compile_hot_functions(device)
        if self.guidance_method != "":
            # Preserve the historical warm-up order for GD as well: these
            # dummy tensors consume RNG before trajectory initialization.
            self._compile_cbf_functions(device)

    def _compile_hot_functions(self, device: torch.device):
        if getattr(self, "_compile_done", False):
            return
        self._compile_done = True

        arm_dof = min(self.arm_dof, 7)
        dummy_q = torch.randn(17, arm_dof, device=device, dtype=torch.float32)

        compiled_jac = torch.compile(
            self._pk_chain.jacobian_tensor, fullgraph=True, mode="reduce-overhead",
        )
        _ = compiled_jac(dummy_q)
        self._jacobian_fn = compiled_jac


        bsz, horizon = 1, 16
        dummy_pos_c = torch.randn(bsz, horizon, 3, device=device, dtype=torch.float32)
        dummy_pos_t = torch.randn(bsz, horizon, 3, device=device, dtype=torch.float32)
        I3 = torch.eye(3, device=device, dtype=torch.float32)
        dummy_rot_c = I3.unsqueeze(0).unsqueeze(0).expand(bsz, horizon, 3, 3).contiguous()
        dummy_rot_t = dummy_rot_c.clone()
        compiled_twist = torch.compile(
            twist6_from_matrices_fast, fullgraph=True, mode="reduce-overhead",
        )
        _ = compiled_twist(dummy_pos_c, dummy_rot_c, dummy_pos_t, dummy_rot_t)
        self._twist_fn = compiled_twist

    def _compile_cbf_functions(self, device: torch.device):
        if getattr(self, "_compile_cbf_done", False):
            return
        self._compile_cbf_done = True

        robot_dof = int(self._robot_dof)
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
        _ = compiled_qp(dummy_jac, dummy_grad_h, dummy_h, dummy_scale)
        self._cbf_qp_fn = compiled_qp

    def _solve_start_joint_from_pose(self, chunk_start_pose: torch.Tensor) -> torch.Tensor:
        B = chunk_start_pose.shape[0]
        start_pos = chunk_start_pose[:, :3]
        start_rot = axis_angle_to_matrix(chunk_start_pose[:, 3:6])
        start_quat = matrix_to_quaternion(start_rot)
        seed = torch.zeros((B, self._robot_dof), device=chunk_start_pose.device, dtype=chunk_start_pose.dtype)
        goal_pose = Pose(position=start_pos, quaternion=start_quat)
        with torch.enable_grad():
            ik_result = self._ik_solver.solve_batch(goal_pose, retract_config=seed)
        q = ik_result.solution.squeeze(1)
        if hasattr(ik_result, "success"):
            succ = ik_result.success
            while succ.ndim > 1:
                succ = succ.squeeze(-1)
            q = torch.where(succ.unsqueeze(-1), q, seed)
        return q

    def _ik_from_absolute(
        self, abs_pos: torch.Tensor, abs_rot: torch.Tensor, seed_q: torch.Tensor,
    ) -> torch.Tensor:
        B, T, _ = abs_pos.shape
        pos_flat = abs_pos.reshape(-1, 3)
        quat_flat = matrix_to_quaternion(abs_rot.reshape(-1, 3, 3))
        seed_flat = seed_q.reshape(-1, self._robot_dof).contiguous()
        goal_pose = Pose(position=pos_flat, quaternion=quat_flat)
        # Force ALL num_seeds to be the caller-provided seed (e.g. current
        # joint pos) so curobo does not explore alternative IK branches via
        # random seeds. seed_config shape per curobo: (batch, n_seeds, dof).
        n_seeds = int(self.ik_num_seeds)
        seed_cfg = seed_flat.unsqueeze(1).expand(B * T, n_seeds, self._robot_dof).contiguous()
        with torch.enable_grad():
            ik_result = self._ik_solver.solve_batch(
                goal_pose,
                retract_config=seed_flat,
                seed_config=seed_cfg,
            )
        q_flat = ik_result.solution.squeeze(1)
        if hasattr(ik_result, "success"):
            succ = ik_result.success
            while succ.ndim > 1:
                succ = succ.squeeze(-1)
            q_flat = torch.where(succ.unsqueeze(-1), q_flat, seed_flat)
        return q_flat.reshape(B, T, self._robot_dof)

    def _fk_to_absolute(self, q_arm: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        B, T, _ = q_arm.shape
        q_flat = q_arm.reshape(-1, q_arm.shape[-1]).contiguous()
        kin_state = self._kin_model.get_state(q_flat)
        if hasattr(kin_state, "ee_position"):
            pos_flat = kin_state.ee_position
        else:
            pos_flat = kin_state.ee_pose.position
        if hasattr(kin_state, "ee_quaternion"):
            quat_flat = kin_state.ee_quaternion
        else:
            quat_flat = kin_state.ee_pose.quaternion
        pos_flat = pos_flat.clone()
        quat_flat = quat_flat.clone()
        rot_flat = quaternion_to_matrix(quat_flat)
        return pos_flat.reshape(B, T, 3), rot_flat.reshape(B, T, 3, 3)

    def _jacobian(self, q_arm: torch.Tensor) -> torch.Tensor:
        qf = q_arm.reshape(-1, q_arm.shape[-1])
        n = qf.shape[0]

        arm_dof = min(self.arm_dof, qf.shape[-1])
        q_for_jac = qf[:, :arm_dof].detach().to(dtype=torch.float32)
        j_arm = self._jacobian_fn(q_for_jac).to(device=qf.device, dtype=qf.dtype)

        if qf.shape[-1] > arm_dof:
            pad = torch.zeros((n, 6, qf.shape[-1] - arm_dof), device=qf.device, dtype=qf.dtype)
            j = torch.cat([j_arm, pad], dim=-1)
        else:
            j = j_arm
        return j

    def _dls_pinv_map(self, twist: torch.Tensor, jacobian: torch.Tensor) -> torch.Tensor:
        return damped_least_squares_pinv(twist, jacobian, self.jacobian_damping)

    def _build_world_collision(self, obstacle_info: Any, device: torch.device, dtype: torch.dtype):
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

        n_env = 1
        normalized_obstacles = []
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
            self._coll_env_query_idx = torch.arange(n_env, device=device, dtype=torch.int32)
            return

        world_config = [WorldConfig(cuboid=cuboids) for cuboids in world_cuboids_by_env]
        world_coll_config = WorldCollisionConfig(
            tensor_args=self._tensor_args, 
            world_model=world_config,
            max_distance=10.0,
        )
        self._world_collision = create_collision_checker(world_coll_config)
        self._coll_query_buffer = CollisionQueryBuffer()
        self._coll_weight = self._tensor_args.to_device(torch.tensor([1.0], dtype=dtype))
        self._coll_activation_distance = self._tensor_args.to_device(
            torch.tensor([self.guidance_activation_distance], dtype=dtype)
        )
        self._cached_world_key = world_key.detach().clone()
        self._coll_env_query_idx = torch.arange(n_env, device=device, dtype=torch.int32)

    def _env_query_idx_for_batch(self, batch_size: int, device: torch.device):
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

    def _guidance_scale_at(self, idx: int, n_steps: int, t: torch.Tensor, dtype, device):
        return guidance_scale_at(
            idx,
            n_steps,
            self.guidance_scale,
            use_schedule=self.guidance_use_schedule,
            midpoint=self.guidance_schedule_midpoint,
            steepness=self.guidance_schedule_steepness,
            dtype=dtype,
            device=device,
        )

    def _query_worst_signed_distance(self, q_arm: torch.Tensor) -> torch.Tensor:
        B, T, D = q_arm.shape
        q_flat = q_arm.reshape(B * T, D).contiguous()
        kin_state = self._kin_model.get_state(q_flat)
        spheres = kin_state.link_spheres_tensor.reshape(B, T, -1, 4)
        self._coll_query_buffer.update_buffer_shape(
            spheres.shape, self._tensor_args, self._world_collision.collision_types,
        )
        dist = self._world_collision.get_sphere_distance(
            spheres, self._coll_query_buffer, self._coll_weight,
            self._coll_activation_distance,
            env_query_idx=self._env_query_idx_for_batch(B, spheres.device),
            return_loss=True, compute_esdf=True,
        )
        return self._aggregate_signed_distance(dist)

    def _aggregate_signed_distance(self, dist: torch.Tensor) -> torch.Tensor:
        return aggregate_signed_distance(
            dist,
            self.guidance_sdf_agg,
            self.guidance_sdf_topk,
            self.guidance_sdf_softmax_temp,
        )

    def _compute_cbf_linearization(
        self, q_arm: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self._world_collision is None:
            zq = torch.zeros_like(q_arm)
            zh = torch.zeros(q_arm.shape[:2], device=q_arm.device, dtype=q_arm.dtype)
            return zh, zq, zh
        with torch.enable_grad():
            q_req = q_arm.detach().clone().requires_grad_(True)
            dist_worst = self._query_worst_signed_distance(q_req)
            h = curobo_signed_distance_to_cbf_h(dist_worst)
            grad_h = torch.autograd.grad(h.sum(), q_req, allow_unused=True)[0]
            if grad_h is None:
                grad_h = torch.zeros_like(q_req)
            return h.detach(), grad_h.detach(), dist_worst.detach()

    def _solve_batched_cbf_qp(
        self, jac_pos: torch.Tensor, grad_h: torch.Tensor,
        h_value: torch.Tensor, constraint_scale: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Solve batched single-inequality QP in closed form.

        QP:
          min 1/2 * dq^T H dq, with H = J_task^T W J_task + λI
          s.t. grad_h^T dq >= d_safe - h
        """
        return solve_batched_cbf_qp(
            jac_pos,
            grad_h,
            h_value,
            constraint_scale,
            arm_dof=self.arm_dof,
            safety_margin=self.guidance_safety_margin,
            position_weight=self.guidance_task_pos_weight,
            rotation_weight=self.guidance_task_rot_weight,
            regularization=self.guidance_cbf_lambda,
        )

    def _initialize_joint_sample(
        self, condition_data, condition_mask, chunk_start_pose,
        current_joint_angles, base_pos, base_rot, generator,
    ):
        """Project initial noise and realize inpainted targets using this robot."""
        bsz, horizon = condition_data.shape[:2]
        # ── 1) Obtain starting joint configuration
        if current_joint_angles is None:
            q_start = self._solve_start_joint_from_pose(chunk_start_pose)
        else:
            q_start = current_joint_angles[..., : self._robot_dof].to(
                device=condition_data.device, dtype=condition_data.dtype
            )
        q_seed = q_start.unsqueeze(1).expand(bsz, horizon, self._robot_dof)

        # ── 2) Initialize noisy joint trajectory
        trajectory_cart_n = torch.randn(
            size=condition_data.shape,
            dtype=condition_data.dtype,
            device=condition_data.device,
            generator=generator,
        )
        trajectory_cart_n[condition_mask] = condition_data[condition_mask]
        trajectory_cart = self.normalizer["action"].unnormalize(trajectory_cart_n)
        grip = trajectory_cart[..., 9:10]

        jac_init = self._jacobian(q_seed)
        eps_twist = torch.randn(
            (bsz * horizon, 6), device=q_seed.device, dtype=q_seed.dtype, generator=generator,
        )
        dq_init = self._dls_pinv_map(eps_twist, jac_init).reshape(bsz, horizon, self._robot_dof)
        if self.max_dq_per_step > 0:
            dq_init = torch.clamp(dq_init, -self.max_dq_per_step, self.max_dq_per_step)
        q_arm = q_seed + dq_init * self.jac_noise_alpha

        q_traj = torch.cat([q_arm, grip], dim=-1)

        # ── 3) Precompute conditioned joint targets (for inpainting)
        cond_step_mask = condition_mask.any(dim=-1)
        q_cond = None
        if torch.any(cond_step_mask):
            cond_cart = self.normalizer["action"].unnormalize(condition_data)
            _rel9 = cond_cart[..., :9]
            _rel_pos = _rel9[..., :3]
            _rel_rot = rot6d_to_matrix(_rel9[..., 3:])
            cond_abs_pos = (base_rot.unsqueeze(1) @ _rel_pos.unsqueeze(-1)).squeeze(-1) + base_pos.unsqueeze(1)
            cond_abs_rot = base_rot.unsqueeze(1) @ _rel_rot
            q_cond_arm = self._ik_from_absolute(cond_abs_pos, cond_abs_rot, q_seed)
            q_cond = torch.cat([q_cond_arm, cond_cart[..., 9:10]], dim=-1)
            q_traj[cond_step_mask] = q_cond[cond_step_mask]

        return q_traj, q_cond, cond_step_mask

    def _joint_to_cartesian(self, q_traj, base_pos, base_rot_t):
        """Encode the joint trajectory in the frozen Cartesian model's frame."""
        bsz, horizon = q_traj.shape[:2]
        q_arm_curr = q_traj[..., : self._robot_dof]
        grip_curr = q_traj[..., self._robot_dof : self._robot_dof + 1]
        abs_pos_curr, abs_rot_curr = self._fk_to_absolute(q_arm_curr)

        # Relative Cartesian for U-Net
        rel_pos_curr = (base_rot_t.unsqueeze(1)
                        @ (abs_pos_curr - base_pos.unsqueeze(1)).unsqueeze(-1)).squeeze(-1)
        rel_rot_curr = base_rot_t.unsqueeze(1) @ abs_rot_curr
        rel_rot6d_curr = matrix_to_rot6d(rel_rot_curr.reshape(-1, 3, 3)).reshape(bsz, horizon, 6)
        cart_phys_curr = torch.cat([rel_pos_curr, rel_rot6d_curr, grip_curr], dim=-1)
        cart_n_curr = self.normalizer["action"].normalize(cart_phys_curr)
        return cart_n_curr, abs_pos_curr, abs_rot_curr

    def _finalize_joint_sample(
        self, q_traj, q_cond, cond_step_mask, condition_data, condition_mask,
        base_pos, base_rot_t,
    ):
        """Restore conditions and expose both joint and Cartesian trajectories."""
        if q_cond is not None:
            q_traj[cond_step_mask] = q_cond[cond_step_mask]
        self._last_joint_traj = q_traj.detach()
        cart_final_n, _, _ = self._joint_to_cartesian(q_traj, base_pos, base_rot_t)
        cart_final_n[condition_mask] = condition_data[condition_mask]
        return cart_final_n

    def predict_action(
        self,
        obs_dict: Dict[str, torch.Tensor],
        env_batched=False,
        chunk_start_pose: torch.Tensor = None,
        obstacle_info=None,
        current_joint_angles: Optional[torch.Tensor] = None,
    ) -> Dict[str, Any]:
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
            raise ValueError("chunk_start_pose must be provided for joint-space policy inference.")

        sample_result = self.conditional_sample(
            condition_data=cond_data,
            condition_mask=cond_mask,
            local_cond=None,
            global_cond=global_cond,
            chunk_start_pose=chunk_start_pose,
            obstacle_info=obstacle_info,
            current_joint_angles=current_joint_angles,
            **self.kwargs,
        )
        nsample = sample_result

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


class DiffusionUnetTimmPolicyJointSpace(_JointSpacePolicyRuntime):
    """Joint-space denoising comparison with no guidance (default) or GD.

    For the paper's CBF method, select DiffusionUnetTimmPolicyEmbodiSteer.
    """

    def __init__(self, *args, guidance_method: str = "", **kwargs):
        guidance_method = str(guidance_method).lower().strip()
        if guidance_method not in ("", "gd"):
            raise ValueError(
                "DiffusionUnetTimmPolicyJointSpace accepts only '' or 'gd'. "
                "Use DiffusionUnetTimmPolicyEmbodiSteer for CBF guidance."
            )
        super().__init__(*args, guidance_method=guidance_method, **kwargs)

    def _collision_penalty(self, dist: torch.Tensor) -> torch.Tensor:
        return collision_penalty(
            dist, self.guidance_safety_margin, self.guidance_loss_power
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
                spheres.shape, self._tensor_args, self._world_collision.collision_types,
            )
            dist = self._world_collision.get_sphere_distance(
                spheres, self._coll_query_buffer, self._coll_weight,
                self._coll_activation_distance,
                env_query_idx=self._env_query_idx_for_batch(B, spheres.device),
                return_loss=False, compute_esdf=True,
            )
            dist_agg = self._aggregate_signed_distance(dist)
            penalty = self._collision_penalty(dist_agg)
            loss = penalty.sum().sum()
            grad = torch.autograd.grad(loss, q_req, allow_unused=True)[0]
            if grad is None:
                grad = torch.zeros_like(q_req)
            return grad.detach(), loss.detach()

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
        if chunk_start_pose is None:
            raise ValueError("chunk_start_pose is required for joint-space inference.")

        if self.guidance_method not in ("", "gd"):
            raise ValueError("Use DiffusionUnetTimmPolicyEmbodiSteer for CBF guidance.")

        self._ensure_kinematics(condition_data.device)

        use_guidance = (self.guidance_method != "")
        if use_guidance:
            self._build_world_collision(
                obstacle_info=obstacle_info,
                device=condition_data.device,
                dtype=condition_data.dtype,
            )

        base_pos = chunk_start_pose[:, :3]
        base_rot = axis_angle_to_matrix(chunk_start_pose[:, 3:6])
        base_rot_t = base_rot.transpose(-2, -1)

        model = self.model
        scheduler = self.noise_scheduler
        bsz = condition_data.shape[0]
        horizon = condition_data.shape[1]

        q_traj, q_cond, cond_step_mask = self._initialize_joint_sample(
            condition_data, condition_mask, chunk_start_pose,
            current_joint_angles, base_pos, base_rot, generator,
        )

        # Cartesian denoising with joint-space realization.
        scheduler.set_timesteps(self.num_inference_steps)
        timesteps = list(scheduler.timesteps)
        n_steps = len(timesteps)

        for idx, t in enumerate(timesteps):
            if q_cond is not None:
                q_traj[cond_step_mask] = q_cond[cond_step_mask]

            q_arm_curr = q_traj[..., : self._robot_dof]
            cart_n_curr, abs_pos_curr, abs_rot_curr = self._joint_to_cartesian(
                q_traj, base_pos, base_rot_t
            )
            cart_n_curr[condition_mask] = condition_data[condition_mask]

            eps_cart_n = model(cart_n_curr, t, local_cond=local_cond, global_cond=global_cond)

            # Scheduler step in normalized Cartesian space
            cart_n_prev_tgt = scheduler.step(
                eps_cart_n, t, cart_n_curr, generator=generator, **kwargs,
            ).prev_sample
            cart_n_prev_tgt[condition_mask] = condition_data[condition_mask]
            cart_phys_prev_tgt = self.normalizer["action"].unnormalize(cart_n_prev_tgt)

            # Target absolute pose
            rel9_tgt = cart_phys_prev_tgt[..., :9]
            rel_pos_tgt = rel9_tgt[..., :3]
            rel_rot_tgt = rot6d_to_matrix(rel9_tgt[..., 3:])
            abs_pos_tgt = (base_rot.unsqueeze(1) @ rel_pos_tgt.unsqueeze(-1)).squeeze(-1) + base_pos.unsqueeze(1)
            abs_rot_tgt = base_rot.unsqueeze(1) @ rel_rot_tgt

            # World-frame twist
            twist6 = self._twist_fn(abs_pos_curr, abs_rot_curr, abs_pos_tgt, abs_rot_tgt).clone()

            # Clamped Jacobian step
            jac = self._jacobian(q_arm_curr)
            dq = self._dls_pinv_map(twist6.reshape(-1, 6), jac).reshape(bsz, horizon, self._robot_dof)
            if self.max_dq_per_step > 0:
                dq = torch.clamp(dq, -self.max_dq_per_step, self.max_dq_per_step)
            q_arm_new = q_arm_curr + dq

            # GD comparison: clip the gradient, then scale and subtract.
            if use_guidance and self._world_collision is not None:
                grad, _ = self._compute_collision_grad(q_arm_new)
                if torch.any(cond_step_mask):
                    grad = grad.clone()
                    grad[cond_step_mask] = 0.0
                if self.guidance_grad_clip > 0:
                    grad = torch.clamp(grad, -self.guidance_grad_clip, self.guidance_grad_clip)
                gamma = self._guidance_scale_at(idx, n_steps, t, q_arm_new.dtype, q_arm_new.device)
                q_arm_new = q_arm_new - gamma * grad

            q_traj = torch.cat([q_arm_new, cart_phys_prev_tgt[..., 9:10]], dim=-1)

        return self._finalize_joint_sample(
            q_traj, q_cond, cond_step_mask, condition_data, condition_mask,
            base_pos, base_rot_t,
        )


def __getattr__(name):
    # Keep the historical paper alias importable without a circular import.
    # It denotes CBF only; GD callers must use the explicit JointSpace class.
    if name == "EmbodiSteerJointPolicy":
        from .embodisteer import DiffusionUnetTimmPolicyEmbodiSteer

        return DiffusionUnetTimmPolicyEmbodiSteer
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["EmbodiSteerJointPolicy", "DiffusionUnetTimmPolicyJointSpace"]
