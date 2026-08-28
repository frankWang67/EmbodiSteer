#!/usr/bin/env python3
"""Diagnose batch shapes and CUDA timing semantics for EmbodiSteer inference."""

import argparse
import json
import os
import statistics
import sys
import time
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))
os.chdir(ROOT_DIR)

import dill
import torch

from diffusion_policy.common.pytorch_util import dict_apply
from scripts_maniskill.benchmark_jm2d_speed import configure_method, load_policy


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        default="scripts_maniskill/evals/PickPlaceToasterToCounter_0512/ckpt/latest.ckpt",
    )
    parser.add_argument(
        "--benchmark-input",
        default="/tmp/jm2d_benchmark_input_seed2022.pt",
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--trials", type=int, default=20)
    parser.add_argument("--num-inference-steps", type=int, default=16)
    parser.add_argument("--cudnn-benchmark", action="store_true")
    parser.add_argument("--disable-hot-compile", action="store_true")
    parser.add_argument("--compile-unet", action="store_true")
    parser.add_argument(
        "--matmul-precision",
        choices=("highest", "high", "medium"),
        default="highest",
    )
    return parser.parse_args()


def stats(values):
    return {
        "values_ms": values,
        "mean_ms": statistics.fmean(values),
        "std_ms": statistics.stdev(values) if len(values) > 1 else 0.0,
        "median_ms": statistics.median(values),
        "min_ms": min(values),
        "max_ms": max(values),
    }


def tensor_shapes(value):
    if torch.is_tensor(value):
        return {"shape": list(value.shape), "dtype": str(value.dtype)}
    if isinstance(value, dict):
        return {str(k): tensor_shapes(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [tensor_shapes(v) for v in value]
    return type(value).__name__


def callable_info(value):
    return {
        "type": f"{type(value).__module__}.{type(value).__qualname__}",
        "module": getattr(value, "__module__", None),
        "has_torchdynamo_original": hasattr(value, "_torchdynamo_orig_callable"),
    }


def call_policy(policy, benchmark_input, seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    policy.reset()
    with torch.no_grad():
        return policy.predict_action(
            benchmark_input["obs_dict"],
            env_batched=False,
            chunk_start_pose=benchmark_input["chunk_start_pose"],
            obstacle_info=benchmark_input["obstacle_info"],
            current_joint_angles=benchmark_input["current_joint_angles"],
        )


def benchmark_policy(policy, benchmark_input, warmup, trials):
    for i in range(warmup):
        call_policy(policy, benchmark_input, 100 + i)
        torch.cuda.synchronize()

    legacy_cpu_ms = []
    synchronized_wall_ms = []
    cuda_event_ms = []
    result_to_cpu_ms = []

    for i in range(trials):
        torch.cuda.synchronize()
        start = time.perf_counter()
        call_policy(policy, benchmark_input, 1000 + i)
        legacy_cpu_ms.append((time.perf_counter() - start) * 1000.0)
        torch.cuda.synchronize()

    for i in range(trials):
        torch.cuda.synchronize()
        start = time.perf_counter()
        call_policy(policy, benchmark_input, 2000 + i)
        torch.cuda.synchronize()
        synchronized_wall_ms.append((time.perf_counter() - start) * 1000.0)

    for i in range(trials):
        torch.cuda.synchronize()
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()
        call_policy(policy, benchmark_input, 3000 + i)
        end_event.record()
        end_event.synchronize()
        cuda_event_ms.append(start_event.elapsed_time(end_event))

    for i in range(trials):
        torch.cuda.synchronize()
        start = time.perf_counter()
        result = call_policy(policy, benchmark_input, 4000 + i)
        result["joint_action_pred"][0].detach().to("cpu").numpy()
        result_to_cpu_ms.append((time.perf_counter() - start) * 1000.0)

    return {
        "legacy_cpu_timer_without_post_sync": stats(legacy_cpu_ms),
        "synchronized_wall_timer": stats(synchronized_wall_ms),
        "cuda_event_timer": stats(cuda_event_ms),
        "predict_plus_result_cpu_transfer": stats(result_to_cpu_ms),
    }


def benchmark_callable(fn, warmup, trials, *, legacy=False):
    for _ in range(warmup):
        fn()
        torch.cuda.synchronize()
    values = []
    for _ in range(trials):
        torch.cuda.synchronize()
        start = time.perf_counter()
        fn()
        if not legacy:
            torch.cuda.synchronize()
        values.append((time.perf_counter() - start) * 1000.0)
        if legacy:
            torch.cuda.synchronize()
    return stats(values)


def main():
    args = parse_args()
    torch.set_float32_matmul_precision(args.matmul_precision)
    torch.backends.cudnn.benchmark = bool(args.cudnn_benchmark)
    device = torch.device("cuda:0")

    checkpoint = Path(args.checkpoint).resolve()
    fixture_path = Path(args.benchmark_input).resolve()
    payload = torch.load(
        checkpoint.open("rb"), map_location="cpu", pickle_module=dill
    )
    fixture = torch.load(
        fixture_path.open("rb"), map_location="cpu", weights_only=False
    )
    cfg = configure_method(payload["cfg"], "embodisteer", 16)
    workspace, policy = load_policy(payload, cfg, device, "embodisteer")
    policy.num_inference_steps = args.num_inference_steps
    benchmark_input = {
        "obs_dict": dict_apply(
            fixture["obs_dict_cpu"], lambda x: x.to(device)
        ),
        "chunk_start_pose": fixture["chunk_start_pose_cpu"].to(device),
        "current_joint_angles": fixture["current_joint_angles_cpu"].to(device),
        "obstacle_info": fixture["obstacle_info"],
    }

    observed = {"obs_encoder_calls": [], "unet_calls": []}

    def encoder_pre_hook(module, args_):
        if not observed["obs_encoder_calls"]:
            observed["obs_encoder_calls"].append(tensor_shapes(args_[0]))

    def unet_pre_hook(module, args_, kwargs_):
        if not observed["unet_calls"]:
            observed["unet_calls"].append(
                {
                    "sample": tensor_shapes(args_[0]),
                    "timestep": tensor_shapes(args_[1]),
                    "global_cond": tensor_shapes(kwargs_.get("global_cond")),
                }
            )

    encoder_handle = policy.obs_encoder.register_forward_pre_hook(encoder_pre_hook)
    unet_handle = policy.model.register_forward_pre_hook(
        unet_pre_hook, with_kwargs=True
    )
    call_policy(policy, benchmark_input, 1)
    torch.cuda.synchronize()
    encoder_handle.remove()
    unet_handle.remove()

    compile_state_after_initialization = {
        "compile_done": bool(getattr(policy, "_compile_done", False)),
        "compile_cbf_done": bool(getattr(policy, "_compile_cbf_done", False)),
        "jacobian_fn": callable_info(policy._jacobian_fn),
        "twist_fn": callable_info(policy._twist_fn),
        "cbf_qp_fn": callable_info(policy._cbf_qp_fn),
        "unet": callable_info(policy.model),
    }

    if args.disable_hot_compile:
        policy._jacobian_fn = policy._pk_chain.jacobian_tensor
        policy._twist_fn = policy._twist6_from_matrices

        def eager_cbf_qp(jac, grad_h, h_value, constraint_scale):
            return policy._solve_batched_cbf_qp(
                jac_pos=jac,
                grad_h=grad_h,
                h_value=h_value,
                constraint_scale=constraint_scale,
            )

        policy._cbf_qp_fn = eager_cbf_qp

    if args.compile_unet:
        policy.model = torch.compile(
            policy.model, fullgraph=True, mode="reduce-overhead"
        )

    effective_compile_state = {
        "hot_compile_disabled_for_measurement": args.disable_hot_compile,
        "unet_compile_requested": args.compile_unet,
        "jacobian_fn": callable_info(policy._jacobian_fn),
        "twist_fn": callable_info(policy._twist_fn),
        "cbf_qp_fn": callable_info(policy._cbf_qp_fn),
        "unet": callable_info(policy.model),
    }

    with torch.no_grad():
        normalized_obs = policy.normalizer.normalize(benchmark_input["obs_dict"])

    def encoder_call():
        with torch.no_grad():
            policy.obs_encoder(normalized_obs)

    with torch.no_grad():
        global_cond = policy.obs_encoder(normalized_obs)
    torch.cuda.synchronize()
    unet_sample = torch.zeros(
        (1, policy.action_horizon, policy.action_dim),
        device=device,
        dtype=policy.dtype,
    )
    policy.noise_scheduler.set_timesteps(policy.num_inference_steps)
    timesteps = list(policy.noise_scheduler.timesteps)

    def unet_16_call():
        with torch.no_grad():
            for timestep in timesteps:
                policy.model(
                    unet_sample,
                    timestep,
                    local_cond=None,
                    global_cond=global_cond,
                )

    report = {
        "checkpoint": str(checkpoint),
        "fixture": str(fixture_path),
        "gpu": torch.cuda.get_device_name(device),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version(),
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
        "matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
        "policy_dtype": str(policy.dtype),
        "env_batched": False,
        "configured_batch_size": 1,
        "action_horizon": policy.action_horizon,
        "action_dim": policy.action_dim,
        "num_inference_steps": policy.num_inference_steps,
        "input_shapes": tensor_shapes(benchmark_input),
        "observed_forward_shapes": observed,
        "compile_state_after_initialization": compile_state_after_initialization,
        "effective_compile_state": effective_compile_state,
        "policy_timing": benchmark_policy(
            policy, benchmark_input, args.warmup, args.trials
        ),
        "encoder_synchronized": benchmark_callable(
            encoder_call, args.warmup, args.trials
        ),
        "encoder_legacy_without_post_sync": benchmark_callable(
            encoder_call, args.warmup, args.trials, legacy=True
        ),
        "unet_16_steps_synchronized": benchmark_callable(
            unet_16_call, args.warmup, args.trials
        ),
        "unet_16_steps_legacy_without_post_sync": benchmark_callable(
            unet_16_call, args.warmup, args.trials, legacy=True
        ),
    }

    output = Path(args.output).resolve()
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)
    print(f"Saved {output}", flush=True)


if __name__ == "__main__":
    main()
