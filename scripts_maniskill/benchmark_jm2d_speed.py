#!/usr/bin/env python3
"""Synchronized B=1 inference benchmark for vanilla, EmbodiSteer, and JM2D."""

import argparse
import copy
import datetime as dt
import gc
import json
import os
import re
import statistics
import sys
import threading
import time
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))
os.chdir(ROOT_DIR)

import dill
import hydra
import numpy as np
import pytorch_kinematics as pk
import pynvml
import torch
from omegaconf import OmegaConf, open_dict

from curobo.util_file import (
    get_assets_path,
    get_robot_configs_path,
    join_path,
    load_yaml,
)
from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.workspace.base_workspace import BaseWorkspace
from mani_skill.utils.wrappers.flatten import FlattenRGBDObservationWrapper
from scripts_maniskill.utils import make_eval_envs, maniskill_to_umi_env_obs
from umi.real_world.real_inference_util import get_real_umi_obs_dict
from embodisteer.runtime_config import (
    ee_policy_overrides,
    joint_policy_overrides,
    load_policy_config,
)


OmegaConf.register_new_resolver("eval", eval, replace=True)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        required=True,
    )
    parser.add_argument(
        "--benchmark-input",
        default=None,
        help=(
            "Optional torch-saved CPU benchmark input. When provided, the "
            "benchmark process never creates a simulation environment."
        ),
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--trials", type=int, default=20)
    parser.add_argument("--seed", type=int, default=2022)
    parser.add_argument(
        "--policy-config",
        default=str(ROOT_DIR / "configs" / "policy" / "embodisteer.yaml"),
        help="Canonical policy YAML supplying all algorithm hyperparameters.",
    )
    parser.add_argument(
        "--physical-gpu-index",
        type=int,
        default=None,
        help="NVML physical GPU index; required unless --allow-shared-gpu is used.",
    )
    parser.add_argument("--exclusive-poll-interval", type=float, default=0.05)
    parser.add_argument("--allow-shared-gpu", action="store_true")
    args = parser.parse_args()
    if not args.allow_shared_gpu and args.physical_gpu_index is None:
        parser.error("--physical-gpu-index is required unless --allow-shared-gpu is used")
    return args


class GPUExclusivityMonitor:
    """Detect any foreign CUDA process during the complete benchmark window."""

    def __init__(self, physical_gpu_index, poll_interval_s):
        self.physical_gpu_index = int(physical_gpu_index)
        self.poll_interval_s = float(poll_interval_s)
        self.pid = os.getpid()
        self._stop_event = threading.Event()
        self._violation_lock = threading.Lock()
        self._violation = None
        self._thread = None
        self._handle = None

    def _foreign_processes(self):
        processes = pynvml.nvmlDeviceGetComputeRunningProcesses(self._handle)
        return [
            {
                "pid": int(process.pid),
                "used_gpu_memory_bytes": int(process.usedGpuMemory),
            }
            for process in processes
            if int(process.pid) != self.pid
        ]

    def _poll(self):
        while not self._stop_event.is_set():
            try:
                foreign = self._foreign_processes()
                if foreign:
                    with self._violation_lock:
                        if self._violation is None:
                            self._violation = {
                                "timestamp": dt.datetime.now().astimezone().isoformat(),
                                "foreign_compute_processes": foreign,
                            }
            except Exception as exc:
                with self._violation_lock:
                    if self._violation is None:
                        self._violation = {
                            "timestamp": dt.datetime.now().astimezone().isoformat(),
                            "monitor_error": repr(exc),
                        }
            self._stop_event.wait(self.poll_interval_s)

    def start(self):
        pynvml.nvmlInit()
        self._handle = pynvml.nvmlDeviceGetHandleByIndex(
            self.physical_gpu_index
        )
        foreign = self._foreign_processes()
        if foreign:
            pynvml.nvmlShutdown()
            raise RuntimeError(
                f"GPU {self.physical_gpu_index} is not exclusive: {foreign}"
            )
        self._thread = threading.Thread(
            target=self._poll,
            name="gpu-exclusivity-monitor",
            daemon=True,
        )
        self._thread.start()

    def assert_clean(self):
        with self._violation_lock:
            violation = copy.deepcopy(self._violation)
        if violation is not None:
            raise RuntimeError(
                "GPU exclusivity was violated during benchmark: "
                f"{violation}"
            )

    def stop(self):
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=max(1.0, 4.0 * self.poll_interval_s))
        try:
            self.assert_clean()
        finally:
            if self._handle is not None:
                pynvml.nvmlShutdown()


def infer_robot_kinematic_args(robot_cfg_name):
    robot_cfg = load_yaml(
        join_path(get_robot_configs_path(), robot_cfg_name)
    )["robot_cfg"]
    kin_cfg = robot_cfg.get("kinematics", {})
    urdf_rel = kin_cfg.get("urdf_path")
    ee_link_name = kin_cfg.get("ee_link", "eef")
    if urdf_rel is None:
        return None, ee_link_name, None

    urdf_path = join_path(get_assets_path(), urdf_rel)
    arm_dof = None
    try:
        with open(urdf_path, "r") as f:
            urdf_str = f.read()
        try:
            chain = pk.build_serial_chain_from_urdf(urdf_str, ee_link_name)
        except ValueError:
            chain = pk.build_serial_chain_from_urdf(
                urdf_str.encode("utf-8"), ee_link_name
            )
        arm_dof = int(chain.n_joints)
    except Exception:
        pass
    return urdf_path, ee_link_name, arm_dof


def percentile(values, q):
    return float(np.percentile(np.asarray(values, dtype=np.float64), q))


def summarize(latencies):
    mean_s = statistics.fmean(latencies)
    return {
        "num_trials": len(latencies),
        "latencies_s": latencies,
        "mean_s": mean_s,
        "std_s": statistics.stdev(latencies) if len(latencies) > 1 else 0.0,
        "median_s": statistics.median(latencies),
        "p95_s": percentile(latencies, 95),
        "min_s": min(latencies),
        "max_s": max(latencies),
        "frequency_hz": 1.0 / mean_s,
    }


def parse_jm2d_method(method):
    match = re.fullmatch(r"jm2d_n(\d+)_i(\d+)", method)
    if match is None:
        return None
    return int(match.group(1)), int(match.group(2))


def capture_benchmark_input(cfg, seed):
    env_kwargs = dict(
        robot_uids="panda_robotiq_wristcam",
        control_mode="pd_joint_pos",
        obs_mode="rgb",
        render_mode="rgb_array",
        max_episode_steps=500,
        sensor_configs=dict(shader_pack="default"),
        reward_mode="normalized_dense",
        harder=True,
    )
    env = make_eval_envs(
        "PickPlaceToasterToCounter-v1",
        1,
        "physx_cpu",
        env_kwargs,
        dict(obs_horizon=cfg.task.shape_meta.obs.camera0_rgb.horizon),
        video_dir=None,
        wrappers=[FlattenRGBDObservationWrapper],
        track_collisions=False,
        info_on_video=False,
    )
    try:
        raw_obs, _ = env.reset(seed=[seed])
        current_joint_angles = torch.as_tensor(
            raw_obs["joint_state"][:, -1, :], dtype=torch.float32
        )
        env_obs = maniskill_to_umi_env_obs(raw_obs)
        chunk_start_pose_np = np.concatenate(
            [
                env_obs["robot0_eef_pos"][:, -1],
                env_obs["robot0_eef_rot_axis_angle"][:, -1],
            ],
            axis=-1,
        )
        obs_dict_np = get_real_umi_obs_dict(
            env_obs=env_obs,
            shape_meta=cfg.task.shape_meta,
            obs_pose_repr=cfg.task.pose_repr.obs_pose_repr,
            episode_start_pose=[chunk_start_pose_np],
            batched=True,
        )
        obstacle_info = env.call("get_obstacles_info")
        return {
            "obs_dict_cpu": dict_apply(
                obs_dict_np, lambda x: torch.from_numpy(x).contiguous()
            ),
            "chunk_start_pose_cpu": torch.from_numpy(
                chunk_start_pose_np
            ).to(dtype=torch.float32),
            "current_joint_angles_cpu": current_joint_angles.contiguous(),
            "obstacle_info": obstacle_info,
        }
    finally:
        env.close()


def add_robot_config(cfg, include_robot_uid=True):
    robot_cfg_name = "panda_robotiq_wristcam.yml"
    urdf_path, ee_link_name, arm_dof = infer_robot_kinematic_args(robot_cfg_name)
    with open_dict(cfg.policy):
        if include_robot_uid:
            cfg.policy.robot_uid = "panda_robotiq_wristcam"
        cfg.policy.robot_cfg_name = robot_cfg_name
        if urdf_path is not None:
            cfg.policy.robot_urdf_path = urdf_path
        cfg.policy.ee_link_name = ee_link_name
        if arm_dof is not None:
            cfg.policy.arm_dof = arm_dof


def add_robot_and_guidance_config(cfg, policy_settings):
    add_robot_config(cfg)
    with open_dict(cfg.policy):
        for key, value in joint_policy_overrides(policy_settings).items():
            cfg.policy[key] = value


def configure_method(base_cfg, method, policy_settings):
    cfg = copy.deepcopy(base_cfg)
    with open_dict(cfg.policy):
        cfg.policy.num_inference_steps = policy_settings["num_inference_steps"]
    if method == "vanilla":
        cfg.policy._target_ = (
            "embodisteer.policies.ee_space."
            "EmbodiSteerEESpacePolicy"
        )
        with open_dict(cfg.policy):
            for key, value in ee_policy_overrides(policy_settings).items():
                cfg.policy[key] = value
            cfg.policy.use_ee_guidance = False
    elif method == "joint_space_no_guidance":
        cfg.policy._target_ = (
            "embodisteer.policies.ee2joint."
            "EmbodiSteerJointPolicy"
        )
        add_robot_config(cfg, include_robot_uid=False)
        with open_dict(cfg.policy):
            for key, value in joint_policy_overrides(policy_settings).items():
                cfg.policy[key] = value
            cfg.policy.guidance_method = ""
    elif method == "embodisteer":
        cfg.policy._target_ = (
            "embodisteer.policies.ee2joint."
            "EmbodiSteerJointPolicy"
        )
        add_robot_and_guidance_config(cfg, policy_settings)
        with open_dict(cfg.policy):
            cfg.policy.guidance_method = policy_settings["guidance"]
            cfg.policy.guidance_use_clean_sample = False
            cfg.policy.guidance_apply_last_step_only = False
    elif method.startswith("batch_sampling_"):
        num_samples = int(method.rsplit("_", 1)[1])
        cfg.policy._target_ = (
            "embodisteer.policies.baselines."
            "DiffusionUnetTimmPolicyBaseline"
        )
        add_robot_and_guidance_config(cfg, policy_settings)
        with open_dict(cfg.policy):
            cfg.policy.baseline_method = "batch_sampling"
            cfg.policy.batch_sampling_num = num_samples
    elif parse_jm2d_method(method) is not None:
        num_samples, _ = parse_jm2d_method(method)
        cfg.policy._target_ = (
            "embodisteer.policies.jm2d."
            "DiffusionUnetTimmPolicyJM2D"
        )
        add_robot_and_guidance_config(cfg, policy_settings)
        with open_dict(cfg.policy):
            cfg.policy.jm2d_num_samples = num_samples
            cfg.policy.jm2d_temperature = policy_settings["jm2d_temperature"]
            cfg.policy.jm2d_eta = policy_settings["jm2d_eta"]
    else:
        raise ValueError(method)
    return cfg


def load_policy(payload, cfg, device, method, policy_settings):
    workspace_cls = hydra.utils.get_class(cfg._target_)
    workspace = workspace_cls(cfg)
    workspace: BaseWorkspace
    workspace.load_payload(payload, exclude_keys=None, include_keys=None)
    policy = workspace.ema_model if cfg.training.use_ema else workspace.model
    jm2d_config = parse_jm2d_method(method)
    policy.num_inference_steps = (
        policy_settings["num_inference_steps"]
        if jm2d_config is None
        else jm2d_config[1]
    )
    policy.eval().to(device)
    return workspace, policy


def run_one(policy, benchmark_input, method, seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    policy.reset()
    with torch.no_grad():
        result = policy.predict_action(
            benchmark_input["obs_dict"],
            env_batched=False,
            chunk_start_pose=benchmark_input["chunk_start_pose"],
            obstacle_info=benchmark_input["obstacle_info"],
            current_joint_angles=benchmark_input["current_joint_angles"],
        )
    output_key = "action_pred" if method == "vanilla" else "joint_action_pred"
    output = result[output_key]
    if not bool(torch.isfinite(output).all().item()):
        raise RuntimeError(f"{method} produced non-finite output")
    return result


def benchmark_method(
    payload,
    base_cfg,
    benchmark_input_cpu,
    method,
    warmup,
    trials,
    seed,
    policy_settings,
    device,
    exclusivity_monitor,
):
    cfg = configure_method(base_cfg, method, policy_settings)
    workspace, policy = load_policy(payload, cfg, device, method, policy_settings)
    benchmark_input = {
        "obs_dict": dict_apply(
            benchmark_input_cpu["obs_dict_cpu"], lambda x: x.to(device)
        ),
        "chunk_start_pose": benchmark_input_cpu["chunk_start_pose_cpu"].to(device),
        "current_joint_angles": benchmark_input_cpu[
            "current_joint_angles_cpu"
        ].to(device),
        "obstacle_info": benchmark_input_cpu["obstacle_info"],
    }

    for i in range(warmup):
        if exclusivity_monitor is not None:
            exclusivity_monitor.assert_clean()
        run_one(policy, benchmark_input, method, seed + i)
        torch.cuda.synchronize(device)
        if exclusivity_monitor is not None:
            exclusivity_monitor.assert_clean()

    torch.cuda.reset_peak_memory_stats(device)
    latencies = []
    ik_pose_rates = []
    ik_trajectory_rates = []
    for i in range(trials):
        trial_seed = seed + 1000 + i
        torch.manual_seed(trial_seed)
        torch.cuda.manual_seed_all(trial_seed)
        policy.reset()
        if exclusivity_monitor is not None:
            exclusivity_monitor.assert_clean()
        torch.cuda.synchronize(device)
        start = time.perf_counter()
        with torch.no_grad():
            result = policy.predict_action(
                benchmark_input["obs_dict"],
                env_batched=False,
                chunk_start_pose=benchmark_input["chunk_start_pose"],
                obstacle_info=benchmark_input["obstacle_info"],
                current_joint_angles=benchmark_input["current_joint_angles"],
            )
        torch.cuda.synchronize(device)
        latency = time.perf_counter() - start
        if exclusivity_monitor is not None:
            exclusivity_monitor.assert_clean()
        latencies.append(latency)

        output_key = "action_pred" if method == "vanilla" else "joint_action_pred"
        if not bool(torch.isfinite(result[output_key]).all().item()):
            raise RuntimeError(f"{method} produced non-finite output")

        if parse_jm2d_method(method) is not None and policy._last_jm2d_stats is not None:
            stats = policy._last_jm2d_stats
            ik_pose_rates.append(
                float(stats["ik_pose_success_rate"].float().mean().item())
            )
            ik_trajectory_rates.append(
                float(
                    stats["ik_trajectory_success_rate"].float().mean().item()
                )
            )

    summary = summarize(latencies)
    summary["warmup_calls"] = warmup
    summary["peak_torch_memory_allocated_bytes"] = int(
        torch.cuda.max_memory_allocated(device)
    )
    if ik_pose_rates:
        summary["jm2d_ik_pose_success_rate_mean"] = statistics.fmean(
            ik_pose_rates
        )
        summary["jm2d_ik_trajectory_success_rate_mean"] = statistics.fmean(
            ik_trajectory_rates
        )
        summary["jm2d_ik_pose_success_rates"] = ik_pose_rates
        summary["jm2d_ik_trajectory_success_rates"] = ik_trajectory_rates

    del policy, workspace, cfg, benchmark_input
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize(device)
    return summary


def main():
    args = parse_args()
    policy_settings = load_policy_config(args.policy_config)
    checkpoint = Path(args.checkpoint).resolve()
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)

    payload = torch.load(
        checkpoint.open("rb"), map_location="cpu", pickle_module=dill
    )
    base_cfg = payload["cfg"]
    benchmark_input_path = None
    if args.benchmark_input is None:
        benchmark_input = capture_benchmark_input(base_cfg, args.seed)
        benchmark_input_source = "captured_in_process_then_environment_closed"
        simulation_created_in_benchmark_process = True
    else:
        benchmark_input_path = Path(args.benchmark_input).resolve()
        benchmark_input = torch.load(
            benchmark_input_path.open("rb"),
            map_location="cpu",
            weights_only=False,
        )
        required_input_keys = {
            "obs_dict_cpu",
            "chunk_start_pose_cpu",
            "current_joint_angles_cpu",
            "obstacle_info",
        }
        missing_input_keys = required_input_keys.difference(benchmark_input)
        if missing_input_keys:
            raise ValueError(
                "Benchmark input is missing required keys: "
                f"{sorted(missing_input_keys)}"
            )
        benchmark_input_source = "pre_captured_cpu_fixture"
        simulation_created_in_benchmark_process = False

    device = torch.device("cuda:0")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")

    method_order = [
        "vanilla",
        "joint_space_no_guidance",
        "embodisteer",
        "batch_sampling_4",
        "batch_sampling_8",
        "batch_sampling_16",
        "batch_sampling_32",
        "jm2d_n4_i16",
        "jm2d_n8_i16",
        "jm2d_n16_i16",
        "jm2d_n32_i16",
        "jm2d_n16_i4",
        "jm2d_n16_i8",
        "jm2d_n16_i12",
    ]
    jm2d_reference = (
        f"jm2d_n{policy_settings['jm2d_num_samples']}_"
        f"i{policy_settings['num_inference_steps']}"
    )
    # Keep the standard comparison methods, but always include the configured
    # JM2D reference when a profile changes sample count or inference steps.
    if jm2d_reference not in method_order:
        method_order.append(jm2d_reference)

    report = {
        "timestamp": dt.datetime.now().astimezone().isoformat(),
        "checkpoint": str(checkpoint),
        "checkpoint_size_bytes": checkpoint.stat().st_size,
        "environment": "PickPlaceToasterToCounter-v1",
        "robot": "panda_robotiq_wristcam",
        "batch_size": 1,
        "observation_seed": args.seed,
        "benchmark_input_source": benchmark_input_source,
        "benchmark_input_path": (
            str(benchmark_input_path) if benchmark_input_path is not None else None
        ),
        "simulation_created_in_benchmark_process": (
            simulation_created_in_benchmark_process
        ),
        "action_horizon": int(base_cfg.task.action_horizon),
        "default_diffusion_steps": policy_settings["num_inference_steps"],
        "policy_config": policy_settings["config_path"],
        "timing_scope": (
            "policy.predict_action wall time with torch.cuda.synchronize() "
            "immediately before and after every measured call"
        ),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "gpu_name": torch.cuda.get_device_name(device),
        "warmup_calls_per_method": args.warmup,
        "measured_calls_per_method": args.trials,
        "gpu_exclusivity": {
            "required": not args.allow_shared_gpu,
            "physical_gpu_index": args.physical_gpu_index,
            "nvml_poll_interval_s": args.exclusive_poll_interval,
            "allowed_compute_pid": os.getpid(),
        },
        "method_order": method_order,
        "method_configs": {
            "vanilla": {
                "space": "Cartesian",
                "guidance": "none",
            },
            "joint_space_no_guidance": {
                "space": "joint",
                "guidance": "none",
                "purpose": "diagnostic decomposition of joint mapping overhead",
            },
            "embodisteer": {
                "space": "joint",
                "guidance": "cbf",
                "policy_class": "EmbodiSteerJointPolicy",
                "guidance_scale": policy_settings["guidance_scale"],
                "guidance_safety_margin": policy_settings["guidance_safety_margin"],
                "guidance_activation_distance": policy_settings["guidance_activation_distance"],
                "guidance_grad_clip": policy_settings["guidance_grad_clip"],
                "guidance_cbf_lambda": policy_settings["guidance_cbf_lambda"],
                "guidance_task_pos_weight": policy_settings["guidance_task_pos_weight"],
                "guidance_task_rot_weight": policy_settings["guidance_task_rot_weight"],
            },
            "batch_sampling_4": {
                "space": "Cartesian samples with post-hoc IK/SDF selection",
                "num_samples": 4,
            },
            "batch_sampling_8": {
                "space": "Cartesian samples with post-hoc IK/SDF selection",
                "num_samples": 8,
            },
            "batch_sampling_16": {
                "space": "Cartesian samples with post-hoc IK/SDF selection",
                "num_samples": 16,
            },
            "batch_sampling_32": {
                "space": "Cartesian samples with post-hoc IK/SDF selection",
                "num_samples": 32,
            },
        },
        "methods": {},
    }

    for method in report["method_order"]:
        jm2d_config = parse_jm2d_method(method)
        if jm2d_config is None:
            continue
        num_samples, num_steps = jm2d_config
        rollout_stages = num_steps * (num_steps + 1) // 2
        report["method_configs"][method] = {
            "space": "Cartesian diffusion with IK/SDF importance weighting",
            "num_samples": num_samples,
            "diffusion_steps": num_steps,
            "temperature": policy_settings["jm2d_temperature"],
            "ddim_eta": policy_settings["jm2d_eta"],
            "final_post_hoc_cbf": True,
            "guidance_safety_margin": policy_settings["guidance_safety_margin"],
            "guidance_activation_distance": policy_settings["guidance_activation_distance"],
            "guidance_cbf_lambda": policy_settings["guidance_cbf_lambda"],
            "guidance_task_pos_weight": policy_settings["guidance_task_pos_weight"],
            "guidance_task_rot_weight": policy_settings["guidance_task_rot_weight"],
            "sequential_clean_rollout_unet_stages": rollout_stages,
            "candidate_sample_forward_equivalents": (
                num_samples * rollout_stages
            ),
            "relative_model_sample_work_vs_vanilla": (
                num_samples
                * rollout_stages
                / float(policy_settings["num_inference_steps"])
            ),
            "candidate_pose_ik_targets_per_call": (
                num_samples * num_steps * int(base_cfg.task.action_horizon)
            ),
        }

    exclusivity_monitor = None
    if not args.allow_shared_gpu:
        exclusivity_monitor = GPUExclusivityMonitor(
            physical_gpu_index=args.physical_gpu_index,
            poll_interval_s=args.exclusive_poll_interval,
        )
        exclusivity_monitor.start()

    try:
        for method in report["method_order"]:
            print(f"Benchmarking {method}...", flush=True)
            report["methods"][method] = benchmark_method(
                payload=payload,
                base_cfg=base_cfg,
                benchmark_input_cpu=benchmark_input,
                method=method,
                warmup=args.warmup,
                trials=args.trials,
                seed=args.seed,
                policy_settings=policy_settings,
                device=device,
                exclusivity_monitor=exclusivity_monitor,
            )
            print(json.dumps(report["methods"][method], indent=2), flush=True)
    finally:
        if exclusivity_monitor is not None:
            exclusivity_monitor.stop()

    if jm2d_reference not in report["methods"]:
        raise RuntimeError(
            "The configured JM2D reference was not benchmarked: "
            f"{jm2d_reference}"
        )
    report["jm2d_reference_method"] = jm2d_reference
    jm2d_mean = report["methods"][jm2d_reference]["mean_s"]
    for method in report["method_order"]:
        if method == jm2d_reference:
            continue
        report["methods"][method]["jm2d_latency_ratio"] = (
            jm2d_mean / report["methods"][method]["mean_s"]
        )

    with output.open("w") as f:
        json.dump(report, f, indent=2)
        f.write("\n")
    print(f"Saved benchmark report to {output}", flush=True)


if __name__ == "__main__":
    main()
