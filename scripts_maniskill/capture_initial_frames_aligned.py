"""
Capture initial episode frames for all 9 robots, all moved to panda's initial EE pose via IK.

Usage:
    python scripts_maniskill/capture_initial_frames_aligned.py --output-dir /tmp/robot_frames_aligned
"""
import os
import sys

ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
sys.path.append(ROOT_DIR)
os.chdir(ROOT_DIR)

import warnings
warnings.filterwarnings("ignore")

import argparse
import numpy as np
import torch
import gymnasium as gym
from PIL import Image

import mani_skill
from mani_skill.utils.wrappers.flatten import FlattenRGBDObservationWrapper

ROBOTS = [
    "ur5_robotiq_wristcam",
    "panda_robotiq_wristcam",
    "xarm6_robotiq_wristcam",
    "xarm7_robotiq_wristcam",
    "iiwa7_robotiq_wristcam",
    "gen3_6dof_robotiq_wristcam",
    "gen3_7dof_robotiq_wristcam",
    "rizon4_robotiq_wristcam",
    "sawyer_robotiq_wristcam",
]

ENV_ID = "MakeIcedCoffee-v1"
CONTROL_MODE = "pd_ee_delta_pose"


def make_env(robot_uid, seed, harder=True):
    kwargs = dict(
        robot_uids=robot_uid,
        control_mode=CONTROL_MODE,
        obs_mode="rgb",
        render_mode="rgb_array",
        max_episode_steps=2,
        sensor_configs=dict(shader_pack="default"),
    )
    if harder:
        kwargs["harder"] = True
    env = gym.make(ENV_ID, **kwargs)
    env = FlattenRGBDObservationWrapper(env)
    env.reset(seed=seed)
    return env


def get_arm_controller(env_unwrapped):
    ctrl = env_unwrapped.agent.controller
    if hasattr(ctrl, "controllers"):
        return ctrl.controllers["arm"]
    return ctrl


def get_panda_target_pose(seed):
    """Return panda's initial EE pose (world frame) as a Pose object."""
    env = make_env("panda_robotiq_wristcam", seed)
    arm_ctrl = get_arm_controller(env.unwrapped)
    target_pose = arm_ctrl.ee_pose  # world-frame Pose, shape [1, ...]
    env.close()
    return target_pose


def solve_ik_iterative(arm_ctrl, unwrapped, target_pose_world, n_iter=200, tol=3e-3):
    """Solve IK to move EE to target_pose_world.

    physx_cpu uses pinocchio which expects poses in the robot base frame,
    so we convert before calling compute_ik.
    """
    # Convert world-frame target to robot base frame (pinocchio expects base frame)
    base_pose = arm_ctrl.root_link.pose
    target_pose_base = base_pose.inv() * target_pose_world

    pos_error = float("inf")
    for i in range(n_iter):
        current_pose_base = arm_ctrl.ee_pose_at_base
        pos_error = torch.norm(target_pose_base.p - current_pose_base.p).item()
        if pos_error < tol:
            break
        q0 = unwrapped.agent.robot.get_qpos()
        target_qpos = arm_ctrl.kinematics.compute_ik(
            pose=target_pose_base,
            q0=q0,
            is_delta_pose=False,
            current_pose=current_pose_base,
            solver_config={"type": "levenberg_marquardt", "alpha": 1.0},
        )
        if target_qpos is None:
            break
        full_qpos = q0.clone()
        full_qpos[:, arm_ctrl.active_joint_indices] = target_qpos
        unwrapped.agent.robot.set_qpos(full_qpos)
    return arm_ctrl.ee_pose.p[0].cpu().numpy(), pos_error


def solve_ik_and_capture(robot_uid, target_pose_world, output_dir, seed, harder=True):
    env = make_env(robot_uid, seed, harder=harder)
    unwrapped = env.unwrapped
    arm_ctrl = get_arm_controller(unwrapped)

    final_pos, pos_error = solve_ik_iterative(arm_ctrl, unwrapped, target_pose_world)
    print(f"  IK done  pos_error={pos_error:.4f}  EE pos: {final_pos.round(4)}")

    frame = env.render()
    env.close()

    if hasattr(frame, "cpu"):
        frame = frame.cpu().numpy()
    frame = np.squeeze(frame)
    if frame.dtype != np.uint8:
        frame = np.clip(frame, 0, 255).astype(np.uint8)

    out_path = os.path.join(output_dir, f"{robot_uid}.png")
    Image.fromarray(frame).save(out_path)
    print(f"  Saved: {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", "-o",
                        default="scripts_maniskill/robot_initial_frames_aligned")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--robots", nargs="+", default=None)
    parser.add_argument("--no-obstacle", action="store_true",
                        help="Disable obstacles (harder=False)")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    robots = args.robots if args.robots else ROBOTS
    harder = not args.no_obstacle

    print("Getting panda's initial EE pose as IK target...")
    target_pose = get_panda_target_pose(args.seed)
    print(f"Target EE pos (world): {target_pose.p[0].cpu().numpy().round(4)}")

    for robot_uid in robots:
        print(f"\nCapturing: {robot_uid}")
        try:
            solve_ik_and_capture(robot_uid, target_pose, args.output_dir, args.seed,
                                 harder=harder)
        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback; traceback.print_exc()

    print(f"\nDone. Images saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
