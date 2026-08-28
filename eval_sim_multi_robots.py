#!/usr/bin/env python3
"""Run the single-robot ManiSkill evaluator over a configured robot set."""

import os
import sys
import subprocess
import re
import shlex
from argparse import ArgumentParser
from datetime import datetime

from embodisteer.runtime_config import PolicyConfigError, load_policy_config

SCRIPT_PATH = os.path.join(os.path.dirname(__file__), "eval_sim_single_robot.py")

parser = ArgumentParser(description="Multi-Robot Policy Model Evaluation")
parser.add_argument("--input", "-i", type=str, required=True, help="Path to checkpoint and experiment results")
parser.add_argument("--ckpt-filename", "-f", type=str, required=True, help="Checkpoint filename within the experiment folder")
parser.add_argument("--env-id", "-e", type=str, required=True, help="Environment ID")
parser.add_argument("--robot-id", type=str, default="", help="Robot ID to evaluate (from tasks name). If empty, evaluate all robots.")
parser.add_argument("--sim-backend", "-s", type=str, default="physx_cpu", help="Simulation backend for ManiSkill env")
parser.add_argument("--control-mode", "-c", type=str, default=None, help="ManiSkill control mode; inferred from the policy config when omitted")
parser.add_argument("--num-env", "-n", type=int, default=10, help="Number of parallel environments")
parser.add_argument("--num-eval-episodes", "-ne", type=int, default=100, help="Number of evaluation episodes")
parser.add_argument("--env-seed", "--env_seed", dest="env_seed", type=int, default=2022, help="Base random seed for ManiSkill evaluation environments")
parser.add_argument("--obs-mode", "-o", type=str, default="rgb", help="Observation mode for ManiSkill env")
parser.add_argument("--render-mode", "-rm", type=str, default="rgb_array", help="Render mode for ManiSkill env")
parser.add_argument("--steps-per-inference", "-si", type=int, default=8, help="Number of predicted actions to execute per policy call. Use 0 to execute the checkpoint action horizon.")
parser.add_argument("--max-episode-steps", "-mes", type=int, default=500, help="Max episode steps for evaluation")
parser.add_argument("--obstacle", action="store_true", help="Whether to evaluate on harder env (with obstacles)")
parser.add_argument(
    "--obstacle-observation-noise",
    nargs=3,
    type=float,
    default=None,
    metavar=("POS_STD_M", "SIZE_STD_M", "ROT_STD_RAD"),
    help=(
        "Per-episode Gaussian noise for observed obstacle center, full edge "
        "length, and orientation angle"
    ),
)
parser.add_argument(
    "--policy-config",
    default=os.path.join(os.path.dirname(__file__), "configs", "policy", "embodisteer.yaml"),
    help=(
        "YAML file containing inference-space, guidance, baseline and IK settings"
    ),
)
args = parser.parse_args()

try:
    policy_settings = load_policy_config(args.policy_config)
except PolicyConfigError as exc:
    parser.error(str(exc))
args.policy_config = policy_settings["config_path"]
args.inference_space = policy_settings["inference_space"]
args.guidance = policy_settings["guidance"]
args.baseline_method = policy_settings["baseline_method"]
args.guidance_cbf_reverse_task_threshold = policy_settings[
    "guidance_cbf_reverse_task_threshold"
]
if args.control_mode is None:
    args.control_mode = (
        "pd_joint_pos"
        if args.inference_space == "joint" or args.baseline_method
        else "pd_ee_pose"
    )

# Validate flag combinations
if args.inference_space == "ee" and args.guidance == "cbf":
    parser.error("EE-space inference does not support 'cbf' guidance; use 'gd' or leave empty.")
if (args.guidance or args.baseline_method) and not args.obstacle:
    parser.error("The selected policy config requires --obstacle.")
if args.obstacle_observation_noise is not None:
    if any(std < 0.0 for std in args.obstacle_observation_noise):
        parser.error("--obstacle-observation-noise values must be non-negative.")
    if not any(std > 0.0 for std in args.obstacle_observation_noise):
        parser.error(
            "--obstacle-observation-noise must contain at least one positive value."
        )
    if not args.obstacle:
        parser.error("--obstacle-observation-noise requires --obstacle.")
    if not (args.guidance or args.baseline_method):
        parser.error(
            "--obstacle-observation-noise requires guidance or a baseline in "
            "--policy-config."
        )
if (
    args.inference_space == "joint" or args.baseline_method
) and not args.control_mode.startswith("pd_joint"):
    parser.error(
        "Joint-space and baseline policy configs require a pd_joint* control mode."
    )

# ================= 配置区域 =================
tasks = [
    {
        "name": "ur5",
        "robot_uid": "ur5_robotiq_wristcam",
        "script_path": SCRIPT_PATH
    },
    {
        "name": "panda",
        "robot_uid": "panda_robotiq_wristcam",
        "script_path": SCRIPT_PATH
    },
    {
        "name": "xarm6",
        "robot_uid": "xarm6_robotiq_wristcam",
        "script_path": SCRIPT_PATH
    },
    {
        "name": "xarm7",
        "robot_uid": "xarm7_robotiq_wristcam",
        "script_path": SCRIPT_PATH
    },
    # {
    #     "name": "floating_robotiq",
    #     "robot_uid": "floating_robotiq_2f_85_gripper_wristcam",
    #     "script_path": SCRIPT_PATH
    # },
    {
        "name": "iiwa7",
        "robot_uid": "iiwa7_robotiq_wristcam",
        "script_path": SCRIPT_PATH
    },
    {
        "name": "gen3_6dof",
        "robot_uid": "gen3_6dof_robotiq_wristcam",
        "script_path": SCRIPT_PATH
    },
    {
        "name": "gen3_7dof",
        "robot_uid": "gen3_7dof_robotiq_wristcam",
        "script_path": SCRIPT_PATH
    },
    {
        "name": "rizon4",
        "robot_uid": "rizon4_robotiq_wristcam",
        "script_path": SCRIPT_PATH
    },
    {
        "name": "sawyer",
        "robot_uid": "sawyer_robotiq_wristcam",
        "script_path": SCRIPT_PATH
    },
]

# ===========================================


def build_command(task):
    cmd = [
        sys.executable,
        task["script_path"],
        "--input", args.input,
        "--ckpt_filename", args.ckpt_filename,
        "--env_id", args.env_id,
        "--robot_uids", task["robot_uid"],
        "--sim_backend", args.sim_backend,
        "--control_mode", args.control_mode,
        "--num_env", str(args.num_env),
        "--num_eval_episodes", str(args.num_eval_episodes),
        "--env_seed", str(args.env_seed),
        "--obs_mode", args.obs_mode,
        "--render_mode", args.render_mode,
        "--steps_per_inference", str(args.steps_per_inference),
        "--max_episode_steps", str(args.max_episode_steps),
        "--policy-config", args.policy_config,
    ]
    if args.obstacle:
        cmd.append("--obstacle")
    if args.obstacle_observation_noise is not None:
        pos_std, size_std, rot_std = args.obstacle_observation_noise
        cmd.extend(
            [
                "--obstacle_observation_noise",
                f"{pos_std:.12g}",
                f"{size_std:.12g}",
                f"{rot_std:.12g}",
            ]
        )
    return cmd


def get_results_path(task):
    """Replicate the log dir logic from eval_sim_single_robot.py."""
    if args.baseline_method:
        subdir = f"obstacle_baseline_{args.baseline_method}"
    elif args.inference_space == "joint":
        if args.guidance:
            if args.guidance_cbf_reverse_task_threshold is not None:
                threshold = f"{args.guidance_cbf_reverse_task_threshold:.8g}"
                subdir = (
                    "obstacle_joint_space_guidance_reverse_cbf_"
                    f"task_threshold_{threshold}"
                )
            else:
                subdir = "obstacle_joint_space_guidance"
        else:
            subdir = "obstacle_joint_space" if args.obstacle else "no_obstacle_joint_space"
    else:
        if args.guidance:
            subdir = "obstacle_ee_space_guidance"
        else:
            subdir = "obstacle_ee_space" if args.obstacle else "no_obstacle_ee_space"
    if args.obstacle_observation_noise is not None:
        pos_std, size_std, rot_std = args.obstacle_observation_noise
        noise_subdir = (
            f"obs_noise_pos{pos_std:.6g}_size{size_std:.6g}_rot{rot_std:.6g}"
        )
        subdir = os.path.join(subdir, noise_subdir)
    return os.path.join(args.input, "eval_results", task["robot_uid"], subdir, "eval_results.txt")


def _markdown_float_token(value):
    """Format a float as a stable, filename-safe experiment token."""
    return (
        f"{value:.8g}"
        .replace("-", "m")
        .replace("+", "p")
        .replace(".", "p")
    )


def get_markdown_filename():
    """Choose an isolated summary file for incremental evaluation variants."""
    experiment_tags = []
    if args.obstacle_observation_noise is not None:
        pos_std, size_std, rot_std = args.obstacle_observation_noise
        experiment_tags.append(
            "obstacle_observation_noise_"
            f"pos{_markdown_float_token(pos_std)}_"
            f"size{_markdown_float_token(size_std)}_"
            f"rot{_markdown_float_token(rot_std)}"
        )
    if args.guidance_cbf_reverse_task_threshold is not None:
        experiment_tags.append(
            "guidance_cbf_reverse_task_threshold_"
            f"{_markdown_float_token(args.guidance_cbf_reverse_task_threshold)}"
        )

    if not experiment_tags:
        return "results.md"
    return f"results_{'__'.join(experiment_tags)}.md"


def parse_results(filepath):
    """Parse eval_results.txt to dict.  Returns empty dict if file missing."""
    if not os.path.exists(filepath):
        return {}
    results = {}
    with open(filepath) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            match = re.match(r"^([a-z_]+):\s+(.+)$", line)
            if match:
                key = match.group(1)
                val = match.group(2)
                if val == "N/A":
                    results[key] = None
                else:
                    try:
                        results[key] = float(val)
                    except ValueError:
                        results[key] = val
    return results


def run_command(cmd):
    print(f"Executing: {shlex.join(cmd)}")
    result = subprocess.run(cmd)
    if result.returncode != 0:
        print(f"Error executing command: {cmd}")
        sys.exit(1)


def format_pct(val):
    if val is None:
        return "N/A"
    return f"{val * 100:.1f}%"


def format_collision(val, std_val=None):
    if val is None:
        return "N/A"
    if std_val is not None:
        return f"{val:.1f}±{std_val:.1f}"
    return f"{val:.1f}"


def format_reward(val, std_val=None):
    if val is None:
        return "N/A"
    if std_val is not None:
        return f"{val:.3f}±{std_val:.3f}"
    return f"{val:.3f}"


def build_md_header(show_collisions):
    header_cols = ["Robot", "Success Once", "Success At End"]
    if show_collisions:
        header_cols.append("Collisions/Ep")
    header_cols.append("Max Reward/Ep")
    header_cols.append("Fail@All")
    header_cols.append("Fail@NeverSuccess")
    header_cols.append("FirstSuccessStep")
    header = "| " + " | ".join(header_cols) + " |"
    sep = "|" + "|".join([":-:" for _ in header_cols]) + "|"
    return header, sep


def build_data_row(r, show_collisions):
    cols = [r["name"], format_pct(r["success_once"]), format_pct(r["success_at_end"])]
    if show_collisions:
        cols.append(format_collision(r["collision_mean"], r["collision_std"]))
    cols.append(format_reward(r["max_reward_mean"], r["max_reward_std"]))
    cols.append(format_pct(r["fail_all"]))
    cols.append(format_pct(r["fail_in_never_success"]))
    cols.append(format_collision(r["first_success_step_mean"], r["first_success_step_std"]))
    return "| " + " | ".join(cols) + " |"


def build_avg_row(rows, show_collisions):
    valid_so = [r["success_once"] for r in rows if r["success_once"] is not None]
    valid_sae = [r["success_at_end"] for r in rows if r["success_at_end"] is not None]
    avg_cols = ["**Average**"]
    avg_cols.append(format_pct(sum(valid_so) / len(valid_so)) if valid_so else "N/A")
    avg_cols.append(format_pct(sum(valid_sae) / len(valid_sae)) if valid_sae else "N/A")
    if show_collisions:
        valid_cm = [r["collision_mean"] for r in rows if r["collision_mean"] is not None]
        valid_cs = [r["collision_std"] for r in rows if r["collision_std"] is not None]
        avg_cols.append(format_collision(
            sum(valid_cm) / len(valid_cm) if valid_cm else None,
            sum(valid_cs) / len(valid_cs) if valid_cs else None))
    valid_mr = [r["max_reward_mean"] for r in rows if r["max_reward_mean"] is not None]
    valid_mrs = [r["max_reward_std"] for r in rows if r["max_reward_std"] is not None]
    avg_cols.append(format_reward(
        sum(valid_mr) / len(valid_mr) if valid_mr else None,
        sum(valid_mrs) / len(valid_mrs) if valid_mrs else None))
    valid_fn_all = [r["fail_all"] for r in rows if r["fail_all"] is not None]
    avg_cols.append(format_pct(sum(valid_fn_all) / len(valid_fn_all)) if valid_fn_all else "N/A")
    valid_fn = [r["fail_in_never_success"] for r in rows if r["fail_in_never_success"] is not None]
    avg_cols.append(format_pct(sum(valid_fn) / len(valid_fn)) if valid_fn else "N/A")
    valid_fss = [r["first_success_step_mean"] for r in rows if r["first_success_step_mean"] is not None]
    valid_fsss = [r["first_success_step_std"] for r in rows if r["first_success_step_std"] is not None]
    avg_cols.append(format_collision(
        sum(valid_fss) / len(valid_fss) if valid_fss else None,
        sum(valid_fsss) / len(valid_fsss) if valid_fsss else None))
    return "| " + " | ".join(avg_cols) + " |"


def update_avg_row_in_md(md_path, avg_row_line):
    """Remove the Average row only from the current (last) section, then append the updated one."""
    with open(md_path, "r") as f:
        lines = f.readlines()

    # Find the start of the last section (last "## " heading)
    last_section_start = 0
    for i, l in enumerate(lines):
        if l.startswith("## "):
            last_section_start = i

    # Remove Average row only within the last section
    head = lines[:last_section_start]
    tail = [l for l in lines[last_section_start:] if not l.startswith("| **Average**")]

    lines_str = "".join(head + tail).rstrip("\n") + "\n" + avg_row_line + "\n"

    with open(md_path, "w") as f:
        f.write(lines_str)


def main():
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    full_cmd = "python " + " ".join(sys.argv)
    show_collisions = args.obstacle

    # Filter tasks by robot_id if specified
    eval_tasks = tasks
    if args.robot_id:
        eval_tasks = [t for t in tasks if t["name"] == args.robot_id]
        if not eval_tasks:
            print(f"Error: Robot ID '{args.robot_id}' not found in tasks. Available: {[t['name'] for t in tasks]}")
            sys.exit(1)

    results_dir = os.path.join(args.input, "eval_results")
    os.makedirs(results_dir, exist_ok=True)
    md_path = os.path.join(results_dir, get_markdown_filename())

    header, sep = build_md_header(show_collisions)

    # Write section header and table header once before the loop
    block_header = f"\n## {timestamp}\n\n"
    block_header += f"Command: `{full_cmd}`\n\n"
    block_header += header + "\n" + sep + "\n"

    if os.path.exists(md_path):
        with open(md_path, "a") as f:
            f.write(block_header)
    else:
        with open(md_path, "w") as f:
            f.write("# Evaluation Results\n")
            f.write(block_header)

    # Evaluate each robot and write its row immediately
    rows = []
    for task in eval_tasks:
        print(f"\n=== Evaluating Robot: {task['name']} ===")
        cmd = build_command(task)
        run_command(cmd)

        results_path = get_results_path(task)
        results = parse_results(results_path)
        row = {
            "name": task["name"],
            "success_once": results.get("success_once_rate"),
            "success_at_end": results.get("success_at_end_rate"),
            "collision_mean": results.get("avg_collision_per_episode"),
            "collision_std": results.get("std_collision_per_episode"),
            "max_reward_mean": results.get("avg_max_reward_per_episode"),
            "max_reward_std": results.get("std_max_reward_per_episode"),
            "fail_in_never_success": results.get("fail_rate_among_non_success"),
            "fail_all": results.get("fail_rate_all"),
            "first_success_step_mean": results.get("avg_first_success_step"),
            "first_success_step_std": results.get("std_first_success_step"),
        }
        rows.append(row)
        print(f"  Parsed results: {row}")

        data_row = build_data_row(row, show_collisions)
        with open(md_path, "a") as f:
            f.write(data_row + "\n")

        # Keep Average row at the end, updated after every robot
        if len(eval_tasks) > 1:
            if len(rows) == 1:
                with open(md_path, "a") as f:
                    f.write(build_avg_row(rows, show_collisions) + "\n")
            else:
                update_avg_row_in_md(md_path, build_avg_row(rows, show_collisions))

        print(f"  Results written to {md_path}")

    print(f"\nAll results saved to {md_path}")


if __name__ == "__main__":
    main()
