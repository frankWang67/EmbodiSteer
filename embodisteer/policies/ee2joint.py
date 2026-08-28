from typing import Dict, Optional, Tuple, List, Any
import os
import time

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
    get_pred_x0,
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
    absolute_pose_delta_to_twist6,
    absolute_pose_to_relative9,
    damped_least_squares_pinv,
    inv_se3,
    pose9d_to_mat,
    relative_pose9_to_absolute,
    twist6_from_matrices,
)
from embodisteer.guidance import (
    guidance_scale_at,
    solve_batched_cbf_qp,
    solve_batched_reverse_cbf_qcqp,
)


class DiffusionUnetTimmPolicyJointSpace(DiffusionUnetTimmPolicyEESpace):
    """
    Joint-space inference wrapper with optional collision guidance.

    guidance_method controls guidance behavior:
      - "" (empty, default): joint-space denoising without guidance
      - "cbf": batched CBF-QP guidance in joint space
      - "gd": gradient-descent guidance in joint space
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
        ik_refine_each_step: bool = False,
        ik_position_threshold: float = 5e-4,
        ik_rotation_threshold: float = 5e-3,
        init_noise_scale: float = 0.2,
        max_dq_per_step: float = 0.5,
        ik_refine_last_step: bool = False,
        noise_init_mode: str = "jacobian_projected",
        jac_noise_alpha: float = 0.1,
        cartesian_delta_mode: str = "geometric",
        # guidance parameters (only used when guidance_method != "")
        guidance_method: str = "",
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
        guidance_cbf_reverse_task_threshold: Optional[float] = None,
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

        super().__init__(*args, **kwargs)

        self.robot_uid = robot_uid
        self.robot_cfg_name = robot_cfg_name
        self.robot_urdf_path = robot_urdf_path
        self.ee_link_name = ee_link_name
        self.arm_dof = int(arm_dof)
        self.ik_num_seeds = int(ik_num_seeds)
        self.jacobian_damping = float(jacobian_damping)
        self.ik_refine_each_step = bool(ik_refine_each_step)
        self.ik_position_threshold = float(ik_position_threshold)
        self.ik_rotation_threshold = float(ik_rotation_threshold)
        self.init_noise_scale = float(init_noise_scale)
        self.max_dq_per_step = float(max_dq_per_step)
        self.ik_refine_last_step = bool(ik_refine_last_step)
        self.noise_init_mode = str(noise_init_mode)
        self.jac_noise_alpha = float(jac_noise_alpha)
        self.cartesian_delta_mode = str(cartesian_delta_mode)

        assert self.noise_init_mode in ("isotropic", "jacobian_projected", "jacobian_diagonal")
        assert self.cartesian_delta_mode in ("geometric", "se3_delta")

        # guidance config
        guidance_method = str(guidance_method).lower().strip()
        if guidance_method in ("gradient_descent", "gd"):
            guidance_method = "gd"
        elif guidance_method == "cbf":
            guidance_method = "cbf"
        elif guidance_method == "":
            guidance_method = ""
        else:
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
        self.guidance_apply_last_step_only = bool(guidance_apply_last_step_only)
        self.guidance_steps_per_denoise = int(max(guidance_steps_per_denoise, 1))
        self.guidance_use_clean_sample = bool(guidance_use_clean_sample)
        self.guidance_cbf_lambda = float(guidance_cbf_lambda)
        if guidance_cbf_reverse_task_threshold is None:
            self.guidance_cbf_reverse_task_threshold = None
        else:
            self.guidance_cbf_reverse_task_threshold = float(
                guidance_cbf_reverse_task_threshold
            )
            if self.guidance_cbf_reverse_task_threshold <= 0.0:
                raise ValueError(
                    "guidance_cbf_reverse_task_threshold must be positive."
                )
            if self.guidance_method != "cbf":
                raise ValueError(
                    "guidance_cbf_reverse_task_threshold requires guidance_method='cbf'."
                )

        guidance_sdf_agg = str(guidance_sdf_agg).lower()
        if guidance_sdf_agg == "softmax":
            guidance_sdf_agg = "topk"
        if guidance_sdf_agg not in ("max", "topk"):
            raise ValueError(f"Unsupported guidance_sdf_agg={guidance_sdf_agg}")
        self.guidance_sdf_agg = guidance_sdf_agg
        self.guidance_sdf_softmax_temp = float(guidance_sdf_softmax_temp)
        self.guidance_sdf_topk = int(max(guidance_sdf_topk, 1))
        self.guidance_task_pos_weight = float(guidance_task_pos_weight)
        self.guidance_task_rot_weight = float(guidance_task_rot_weight)
        self.guidance_reuse_jacobian = bool(guidance_reuse_jacobian)

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

    # ===========================
    # Robot config inference
    # ===========================
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

    # ===========================
    # Joint trajectory to env action
    # ===========================
    def _joint_traj_to_env_action(self, q_traj: torch.Tensor) -> torch.Tensor:
        arm_q = q_traj[..., : self.arm_dof]
        grip = q_traj[..., self._robot_dof : self._robot_dof + 1]
        return torch.cat([arm_q, grip], dim=-1)

    # ===========================
    # Fast rotation helpers (index-op-free, compilable)
    # ===========================
    @staticmethod
    def _sqrt_positive_part_fast(x: torch.Tensor) -> torch.Tensor:
        return torch.sqrt(torch.relu(x))

    @staticmethod
    def _matrix_to_quaternion_fast(matrix: torch.Tensor) -> torch.Tensor:
        batch_dim = matrix.shape[:-2]
        m00, m01, m02, m10, m11, m12, m20, m21, m22 = torch.unbind(
            matrix.reshape(batch_dim + (9,)), dim=-1,
        )
        q_abs = DiffusionUnetTimmPolicyJointSpace._sqrt_positive_part_fast(
            torch.stack([
                1.0 + m00 + m11 + m22, 1.0 + m00 - m11 - m22,
                1.0 - m00 + m11 - m22, 1.0 - m00 - m11 + m22,
            ], dim=-1),
        )
        quat_by_rijk = torch.stack([
            torch.stack([q_abs[..., 0] ** 2, m21 - m12, m02 - m20, m10 - m01], dim=-1),
            torch.stack([m21 - m12, q_abs[..., 1] ** 2, m10 + m01, m02 + m20], dim=-1),
            torch.stack([m02 - m20, m10 + m01, q_abs[..., 2] ** 2, m12 + m21], dim=-1),
            torch.stack([m10 - m01, m20 + m02, m21 + m12, q_abs[..., 3] ** 2], dim=-1),
        ], dim=-2)
        dtype = q_abs.dtype
        flr = torch.tensor(0.1, device=q_abs.device, dtype=dtype)
        quat_candidates = quat_by_rijk / (2.0 * q_abs[..., None].max(flr))
        weights = torch.nn.functional.one_hot(
            q_abs.argmax(dim=-1), num_classes=4,
        ).to(dtype=dtype)
        out = (quat_candidates * weights.unsqueeze(-1)).sum(dim=-2).reshape(
            batch_dim + (4,),
        )
        return torch.where(out[..., 0:1] < 0, -out, out)

    @staticmethod
    def _quaternion_to_axis_angle_fast(quaternions: torch.Tensor) -> torch.Tensor:
        quaternions = quaternions / torch.linalg.norm(
            quaternions, dim=-1, keepdim=True,
        )
        w, vec = quaternions[..., 0], quaternions[..., 1:]
        half_angles = torch.acos(torch.clamp(w, -1.0, 1.0))
        angles = 2.0 * half_angles
        small_mask = (angles < 1e-6).unsqueeze(-1)
        sin_half = torch.sin(half_angles).unsqueeze(-1)
        half_clamped = half_angles.clamp(min=1e-12).unsqueeze(-1)
        axis_raw = vec / torch.where(small_mask, 1.0, sin_half)
        taylor = vec / torch.where(small_mask, half_clamped, 1.0)
        axis = torch.where(small_mask, taylor, axis_raw)
        return angles.unsqueeze(-1) * axis

    @staticmethod
    def _matrix_to_axis_angle_fast(matrix: torch.Tensor) -> torch.Tensor:
        return DiffusionUnetTimmPolicyJointSpace._quaternion_to_axis_angle_fast(
            DiffusionUnetTimmPolicyJointSpace._matrix_to_quaternion_fast(matrix),
        )

    # ===========================
    # Kinematics initialization
    # ===========================
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
            self._twist6_from_matrices, fullgraph=True, mode="reduce-overhead",
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

        if self.guidance_cbf_reverse_task_threshold is None:
            def _cbf_qp_wrapper(jac, grad_h, h_value, constraint_scale):
                return self._solve_batched_cbf_qp(
                    jac_pos=jac,
                    grad_h=grad_h,
                    h_value=h_value,
                    constraint_scale=constraint_scale,
                )
        else:
            def _cbf_qp_wrapper(jac, grad_h, h_value, constraint_scale):
                return self._solve_batched_reverse_cbf_qcqp(
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

    @staticmethod
    def _inv_se3(mat: torch.Tensor) -> torch.Tensor:
        return inv_se3(mat)

    # ===========================
    # Pose conversions
    # ===========================
    def _relative_pose9_to_absolute(
        self, rel_pose9: torch.Tensor, chunk_start_pose: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return relative_pose9_to_absolute(rel_pose9, chunk_start_pose)

    def _absolute_pose_to_relative9(
        self, abs_pos: torch.Tensor, abs_rot: torch.Tensor, chunk_start_pose: torch.Tensor,
    ) -> torch.Tensor:
        return absolute_pose_to_relative9(abs_pos, abs_rot, chunk_start_pose)

    # ===========================
    # FK / IK / Jacobian helpers
    # ===========================
    def _solve_start_joint_from_pose(self, chunk_start_pose: torch.Tensor) -> torch.Tensor:
        import time
        B = chunk_start_pose.shape[0]
        start_pos = chunk_start_pose[:, :3]
        start_rot = axis_angle_to_matrix(chunk_start_pose[:, 3:6])
        start_quat = matrix_to_quaternion(start_rot)
        seed = torch.zeros((B, self._robot_dof), device=chunk_start_pose.device, dtype=chunk_start_pose.dtype)
        goal_pose = Pose(position=start_pos, quaternion=start_quat)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.enable_grad():
            ik_result = self._ik_solver.solve_batch(goal_pose, retract_config=seed)
        torch.cuda.synchronize()
        dt = time.perf_counter() - t0
        print(f"[IK timing] _solve_start_joint_from_pose: batch={B}, time={dt*1000:.1f}ms")
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
        import time
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
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.enable_grad():
            ik_result = self._ik_solver.solve_batch(
                goal_pose,
                retract_config=seed_flat,
                seed_config=seed_cfg,
            )
        torch.cuda.synchronize()
        dt = time.perf_counter() - t0
        q_flat = ik_result.solution.squeeze(1)
        n_succ = n_tot = 0
        if hasattr(ik_result, "success"):
            succ = ik_result.success
            while succ.ndim > 1:
                succ = succ.squeeze(-1)
            n_succ = int(succ.sum().item())
            n_tot = int(succ.numel())
            q_flat = torch.where(succ.unsqueeze(-1), q_flat, seed_flat)
        rate = (n_succ / max(n_tot, 1)) if n_tot > 0 else float("nan")
        # print(f"[IK timing] _ik_from_absolute: batch={B}x{T}={B*T}, "
        #       f"time={dt*1000:.1f}ms, success={n_succ}/{n_tot} ({rate*100:.1f}%)")
        return q_flat.reshape(B, T, self._robot_dof)

    def _ik_from_absolute_sequential(
        self, abs_pos: torch.Tensor, abs_rot: torch.Tensor, seed_q: torch.Tensor,
    ) -> torch.Tensor:
        """Sequential per-timestep IK: warm-start each step with the previous
        step's solution to keep IK on the same branch and avoid joint
        discontinuities for redundant / multi-solution arms.

        Args:
            abs_pos: (B, T, 3) target ee positions
            abs_rot: (B, T, 3, 3) target ee rotations
            seed_q:  (B, T, dof) seed joint angles; only seed_q[:, 0] is used
                     as the initial seed, subsequent steps warm-start from the
                     previous step's solution.

        Returns:
            (B, T, dof) joint trajectory.
        """
        import time
        B, T, _ = abs_pos.shape
        quat = matrix_to_quaternion(abs_rot.reshape(-1, 3, 3)).reshape(B, T, 4)

        torch.cuda.synchronize()
        t0 = time.perf_counter()

        q_seed_t = seed_q[:, 0].contiguous()
        q_out = torch.empty(B, T, self._robot_dof, device=abs_pos.device, dtype=abs_pos.dtype)
        n_succ_total = 0
        n_total = 0
        n_seeds = int(self.ik_num_seeds)
        for t in range(T):
            pos_t = abs_pos[:, t].contiguous()
            quat_t = quat[:, t].contiguous()
            goal_pose = Pose(position=pos_t, quaternion=quat_t)
            # Force ALL num_seeds to be the warm-start solution so curobo
            # does not explore alternative IK branches via random seeds.
            # seed_config shape per curobo: (batch, n_seeds, dof).
            # retract_config additionally penalizes deviation in null-space.
            seed_cfg = q_seed_t.unsqueeze(1).expand(B, n_seeds, self._robot_dof).contiguous()
            with torch.enable_grad():
                ik_result = self._ik_solver.solve_batch(
                    goal_pose,
                    retract_config=q_seed_t,
                    seed_config=seed_cfg,
                )
            q_t = ik_result.solution.squeeze(1)
            if hasattr(ik_result, "success"):
                succ = ik_result.success
                while succ.ndim > 1:
                    succ = succ.squeeze(-1)
                q_t = torch.where(succ.unsqueeze(-1), q_t, q_seed_t)
                n_succ_total += int(succ.sum().item())
                n_total += int(succ.numel())
            q_out[:, t] = q_t
            q_seed_t = q_t

        torch.cuda.synchronize()
        dt = time.perf_counter() - t0
        succ_rate = (n_succ_total / max(n_total, 1)) if n_total > 0 else float("nan")
        print(f"[IK timing] _ik_from_absolute_sequential: B={B}, T={T}, "
              f"total={dt*1000:.1f}ms, per-step={dt*1000/T:.1f}ms, "
              f"success={n_succ_total}/{n_total} ({succ_rate*100:.1f}%)")
        return q_out

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

    def _twist6_from_matrices(
        self, abs_pos_curr: torch.Tensor, abs_rot_curr: torch.Tensor,
        abs_pos_tgt: torch.Tensor, abs_rot_tgt: torch.Tensor,
    ) -> torch.Tensor:
        return twist6_from_matrices(
            abs_pos_curr, abs_rot_curr, abs_pos_tgt, abs_rot_tgt
        )

    def _absolute_pose_delta_to_twist6(
        self, abs_pose9_curr: torch.Tensor, abs_pose9_tgt: torch.Tensor,
    ) -> torch.Tensor:
        return absolute_pose_delta_to_twist6(
            abs_pose9_curr,
            abs_pose9_tgt,
            getattr(self, "cartesian_delta_mode", "geometric"),
        )

    # ===========================
    # Collision infrastructure (for guidance)
    # ===========================
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

    def _collision_penalty(self, dist: torch.Tensor) -> torch.Tensor:
        return collision_penalty(
            dist, self.guidance_safety_margin, self.guidance_loss_power
        )

    def _guidance_scale_at(self, idx: int, n_steps: int, t: torch.Tensor, dtype, device):
        return guidance_scale_at(
            idx,
            n_steps,
            self.guidance_scale,
            use_schedule=self.guidance_use_schedule,
            dtype=dtype,
            device=device,
        )

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
        return curobo_signed_distance_to_cbf_h(dist_signed)

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
            h = self._curobo_signed_distance_to_cbf_h(dist_worst)
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

    def _solve_batched_reverse_cbf_qcqp(
        self, jac_pos: torch.Tensor, grad_h: torch.Tensor,
        h_value: torch.Tensor, constraint_scale: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Solve the reviewer-requested reverse CBF problem in closed form.

        For each batch/time element, let a = grad_h and
        H = J_task^T W J_task + lambda I. The convex problem is

          min 1/2 * relu(r - a^T dq)^2
          s.t. sqrt(dq^T H dq) <= rho,

        where r = scale * relu(d_safe - h) and rho is
        guidance_cbf_reverse_task_threshold. Among the zero-collision-cost
        solutions, the closed form selects the minimum-task-disturbance one.
        """
        return solve_batched_reverse_cbf_qcqp(
            jac_pos,
            grad_h,
            h_value,
            constraint_scale,
            arm_dof=self.arm_dof,
            safety_margin=self.guidance_safety_margin,
            position_weight=self.guidance_task_pos_weight,
            rotation_weight=self.guidance_task_rot_weight,
            regularization=self.guidance_cbf_lambda,
            task_threshold=self.guidance_cbf_reverse_task_threshold,
            joint_clip=self.guidance_grad_clip,
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

    def _estimate_clean_joint_from_cartesian(
        self, q_arm_ref: torch.Tensor, cart_phys_clean: torch.Tensor, chunk_start_pose: torch.Tensor,
    ) -> torch.Tensor:
        bsz, horizon, _ = q_arm_ref.shape
        abs_pos_ref, abs_rot_ref = self._fk_to_absolute(q_arm_ref)
        abs_rot6d_ref = matrix_to_rot6d(abs_rot_ref.reshape(-1, 3, 3)).reshape(bsz, horizon, 6)
        abs_pose9_ref = torch.cat([abs_pos_ref, abs_rot6d_ref], dim=-1)
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
        return q_arm_ref + dq_clean

    # ===========================
    # Inference
    # ===========================
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

        if self.noise_init_mode == "jacobian_projected":
            jac_init = self._jacobian(q_seed)
            eps_twist = torch.randn(
                (bsz * horizon, 6), device=q_seed.device, dtype=q_seed.dtype, generator=generator,
            )
            dq_init = self._dls_pinv_map(eps_twist, jac_init).reshape(bsz, horizon, self._robot_dof)
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
                q_seed.shape, device=q_seed.device, dtype=q_seed.dtype, generator=generator,
            ) * per_joint_std
        else:
            q_arm = q_seed + torch.randn(
                q_seed.shape, device=q_seed.device, dtype=q_seed.dtype, generator=generator,
            ) * self.init_noise_scale

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

        # ── 4) Denoising in joint space
        scheduler.set_timesteps(self.num_inference_steps)
        debug = {
            "step_cart_l2": [],
            "guidance_loss": [],
            "guidance_grad_norm": [],
            "clean_sample_cart_err_before": [],
            "clean_sample_cart_err_after": [],
        }
        timing = {
            "guidance_total": 0.0,
            "guidance_grad": 0.0,
            "guidance_apply": 0.0,
        }
        timesteps = list(scheduler.timesteps)
        n_steps = len(timesteps)

        for idx, t in enumerate(timesteps):
            if q_cond is not None:
                q_traj[cond_step_mask] = q_cond[cond_step_mask]

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
            cart_n_curr[condition_mask] = condition_data[condition_mask]

            eps_cart_n = model(cart_n_curr, t, local_cond=local_cond, global_cond=global_cond)

            # Predicted clean Cartesian sample (for guidance_use_clean_sample)
            cart_phys_clean = None
            if use_guidance and self.guidance_use_clean_sample:
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
            jac_ref = jac
            q_ref_for_jac = q_arm_curr
            dq = self._dls_pinv_map(twist6.reshape(-1, 6), jac).reshape(bsz, horizon, self._robot_dof)
            if self.max_dq_per_step > 0:
                dq = torch.clamp(dq, -self.max_dq_per_step, self.max_dq_per_step)
            q_arm_new = q_arm_curr + dq

            # Optional IK refine
            is_last = (idx == n_steps - 1)
            if self.ik_refine_each_step or (is_last and self.ik_refine_last_step):
                q_arm_new = self._ik_from_absolute(abs_pos_tgt, abs_rot_tgt, q_arm_new)

            # ── Guidance application
            if use_guidance and self._world_collision is not None:
                apply_guidance = True
                if self.guidance_apply_last_step_only and not is_last:
                    apply_guidance = False

                if apply_guidance:
                    t0 = time.perf_counter()

                    if self.guidance_use_clean_sample and cart_phys_clean is not None:
                        q_guidance_state = self._estimate_clean_joint_from_cartesian(
                            q_arm_ref=q_arm_new, cart_phys_clean=cart_phys_clean,
                            chunk_start_pose=chunk_start_pose,
                        )
                    else:
                        q_guidance_state = q_arm_new

                    q_before_guidance = q_arm_new.clone()
                    for _ in range(self.guidance_steps_per_denoise):
                        if self.guidance_method == "cbf":
                            tg = time.perf_counter()
                            h_value, grad_h, _ = self._compute_cbf_linearization(q_guidance_state)
                            if self.guidance_reuse_jacobian and not self.guidance_use_clean_sample and q_ref_for_jac is not None:
                                state_change = (q_guidance_state - q_ref_for_jac).abs().max()
                                if state_change < 0.5:
                                    jac_lin = jac_ref.reshape(bsz, horizon, 6, self._robot_dof)
                                else:
                                    jac_lin = self._jacobian(q_guidance_state).reshape(bsz, horizon, 6, self._robot_dof)
                            else:
                                jac_lin = self._jacobian(q_guidance_state).reshape(bsz, horizon, 6, self._robot_dof)
                            gamma = self._guidance_scale_at(idx, n_steps, t, q_arm_new.dtype, q_arm_new.device)
                            dq_cbf, _, _, _ = self._cbf_qp_fn(jac_lin, grad_h, h_value, gamma)
                            timing["guidance_grad"] += time.perf_counter() - tg

                            if torch.any(cond_step_mask):
                                dq_cbf = dq_cbf.clone()
                                dq_cbf[cond_step_mask] = 0.0

                            ta = time.perf_counter()
                            if self.guidance_grad_clip > 0:
                                dq_cbf = torch.clamp(dq_cbf, -self.guidance_grad_clip, self.guidance_grad_clip)
                            q_arm_new = q_arm_new + dq_cbf
                            q_guidance_state = q_guidance_state + dq_cbf
                            timing["guidance_apply"] += time.perf_counter() - ta

                            if return_debug:
                                cbf_violation = torch.relu(self.guidance_safety_margin - h_value)
                                debug["guidance_loss"].append(cbf_violation.sum().detach().cpu())
                                debug["guidance_grad_norm"].append(
                                    torch.linalg.norm(grad_h.reshape(bsz, -1), dim=-1).detach().cpu()
                                )
                        else:  # gd
                            tg = time.perf_counter()
                            grad, guide_loss = self._compute_collision_grad(q_guidance_state)
                            timing["guidance_grad"] += time.perf_counter() - tg

                            if torch.any(cond_step_mask):
                                grad = grad.clone()
                                grad[cond_step_mask] = 0.0

                            ta = time.perf_counter()
                            if self.guidance_grad_clip > 0:
                                grad = torch.clamp(grad, -self.guidance_grad_clip, self.guidance_grad_clip)
                            gamma = self._guidance_scale_at(idx, n_steps, t, q_arm_new.dtype, q_arm_new.device)
                            q_arm_new = q_arm_new - gamma * grad
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

        # Return normalized Cartesian action (keeps external API unchanged)
        q_arm_final = q_traj[..., : self._robot_dof]
        grip_final = q_traj[..., self._robot_dof : self._robot_dof + 1]
        abs_p_f, abs_r_f = self._fk_to_absolute(q_arm_final)
        rel_pos_f = (base_rot_t.unsqueeze(1) @ (abs_p_f - base_pos.unsqueeze(1)).unsqueeze(-1)).squeeze(-1)
        rel_rot_f = base_rot_t.unsqueeze(1) @ abs_r_f
        rel_rot6d_f = matrix_to_rot6d(rel_rot_f.reshape(-1, 3, 3)).reshape(bsz, horizon, 6)
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
                device=self.device, dtype=self.dtype,
            )
        else:
            cond_data = torch.zeros(
                size=(B, self.action_horizon, self.action_dim),
                device=self.device, dtype=self.dtype,
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


EmbodiSteerJointPolicy = DiffusionUnetTimmPolicyJointSpace

__all__ = ["EmbodiSteerJointPolicy", "DiffusionUnetTimmPolicyJointSpace"]
