"""
Capture the initial frame (episode start) for each robot on MakeIcedCoffee-v1 with obstacles.
Saves one PNG per robot to --output-dir.

Usage:
    python scripts_maniskill/capture_initial_frames.py --output-dir /tmp/robot_frames
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
import gymnasium as gym
from PIL import Image

import mani_skill  # registers ManiSkill envs
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


def capture_robot(robot_uid: str, output_dir: str, seed: int = 0):
    env = gym.make(
        ENV_ID,
        robot_uids=robot_uid,
        control_mode=CONTROL_MODE,
        obs_mode="rgb",
        render_mode="rgb_array",
        max_episode_steps=2,
        harder=True,
        sensor_configs=dict(shader_pack="default"),
    )
    env = FlattenRGBDObservationWrapper(env)

    obs, info = env.reset(seed=seed)
    frame = env.render()  # torch.Tensor [1, H, W, 3] or np.ndarray

    if hasattr(frame, "cpu"):
        frame = frame.cpu().numpy()
    frame = np.squeeze(frame)  # [H, W, 3]
    if frame.dtype != np.uint8:
        frame = np.clip(frame, 0, 255).astype(np.uint8)

    env.close()

    out_path = os.path.join(output_dir, f"{robot_uid}.png")
    Image.fromarray(frame).save(out_path)
    print(f"  Saved: {out_path}  shape={frame.shape}")
    return out_path


def main():
    parser = argparse.ArgumentParser(description="Capture initial episode frames for all robots")
    parser.add_argument("--output-dir", "-o", default="scripts_maniskill/robot_initial_frames",
                        help="Directory to save PNG images")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for env reset")
    parser.add_argument("--robots", nargs="+", default=None,
                        help="Subset of robot UIDs to capture (default: all 9)")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    robots = args.robots if args.robots else ROBOTS

    for robot_uid in robots:
        print(f"Capturing: {robot_uid}")
        try:
            capture_robot(robot_uid, args.output_dir, seed=args.seed)
        except Exception as e:
            print(f"  ERROR for {robot_uid}: {e}")

    print(f"\nDone. Images saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
