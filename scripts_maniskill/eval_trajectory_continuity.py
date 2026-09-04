#!/usr/bin/env python3
"""Small, isolated qualitative study of CBF trajectory continuity.

This script intentionally does not modify the policy or the standard evaluation
loop.  At runtime it wraps ``policy.predict_action`` to obtain a noise-matched
counterfactual with CBF disabled, while the environment still executes the
original CBF action chunk.

The default experiment evaluates one fixed-seed episode for Panda and UR5 and
writes raw arrays, a compact Markdown table, and a worst-chunk figure to a
separate continuity-results directory.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Sequence

import numpy as np


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from embodisteer.runtime_config import load_policy_config

ROBOT_UIDS = {
    "panda": "panda_robotiq_wristcam",
    "ur5": "ur5_robotiq_wristcam",
}


def resolve_checkpoint(input_path: str | os.PathLike[str], ckpt_filename: str) -> Path:
    """Resolve checkpoints with the same suffix rules as eval_sim_single_robot.py."""
    input_path = Path(input_path).expanduser().resolve()
    if input_path.is_file():
        return input_path

    candidate = input_path / "ckpt" / ckpt_filename
    candidates = [candidate]
    if candidate.suffix not in (".ckpt", ".pth"):
        candidates.extend([candidate.with_suffix(".ckpt"), candidate.with_suffix(".pth")])
    for path in candidates:
        if path.exists():
            return path.resolve()
    raise FileNotFoundError(
        f"Cannot find checkpoint {ckpt_filename!r} under {input_path / 'ckpt'}"
    )


def _joint_step_arrays(
    q_current: np.ndarray,
    q_traj: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return first and second differences, including current->first action."""
    anchored = np.concatenate([q_current[:, None, :], q_traj], axis=1)
    step = np.diff(anchored, axis=1)
    second = np.diff(step, axis=1)
    return step, second


def compute_continuity_summary(arrays: Mapping[str, np.ndarray]) -> Dict[str, Any]:
    """Compute the intentionally small set of metrics used in the report."""
    q_current = np.asarray(arrays["q_current"], dtype=np.float64)
    q_cbf = np.asarray(arrays["q_cbf"], dtype=np.float64)
    q_no_cbf = np.asarray(arrays["q_no_cbf"], dtype=np.float64)

    if q_cbf.ndim != 3 or q_no_cbf.shape != q_cbf.shape:
        raise ValueError("q_cbf and q_no_cbf must both have shape [chunks, horizon, dof]")
    if q_current.shape != (q_cbf.shape[0], q_cbf.shape[2]):
        raise ValueError("q_current must have shape [chunks, dof]")

    step_cbf, second_cbf = _joint_step_arrays(q_current, q_cbf)
    step_no, second_no = _joint_step_arrays(q_current, q_no_cbf)
    correction = q_cbf - q_no_cbf
    correction_with_zero = np.concatenate(
        [np.zeros_like(correction[:, :1]), correction], axis=1
    )
    correction_variation = np.diff(correction_with_zero, axis=1)

    dof_scale = np.sqrt(float(q_cbf.shape[-1]))
    step_l2_cbf = np.linalg.norm(step_cbf, axis=-1) / dof_scale
    step_l2_no = np.linalg.norm(step_no, axis=-1) / dof_scale
    second_l2_cbf = np.linalg.norm(second_cbf, axis=-1) / dof_scale
    second_l2_no = np.linalg.norm(second_no, axis=-1) / dof_scale
    correction_variation_l2 = np.linalg.norm(correction_variation, axis=-1)

    per_chunk_worst = correction_variation_l2.max(axis=1)
    worst_chunk = int(np.argmax(per_chunk_worst))
    max_second_no = float(second_l2_no.max()) if second_l2_no.size else 0.0
    max_second_cbf = float(second_l2_cbf.max()) if second_l2_cbf.size else 0.0

    active = np.asarray(arrays.get("qp_active", np.zeros(q_cbf.shape[:2], dtype=bool)))
    return {
        "num_chunks": int(q_cbf.shape[0]),
        "horizon": int(q_cbf.shape[1]),
        "arm_dof": int(q_cbf.shape[2]),
        "max_abs_joint_step_no_cbf_rad": float(np.abs(step_no).max()),
        "max_abs_joint_step_cbf_rad": float(np.abs(step_cbf).max()),
        "p95_joint_step_l2_no_cbf_rad": float(np.percentile(step_l2_no, 95)),
        "p95_joint_step_l2_cbf_rad": float(np.percentile(step_l2_cbf, 95)),
        "max_second_difference_no_cbf_rad": max_second_no,
        "max_second_difference_cbf_rad": max_second_cbf,
        "max_second_difference_ratio_cbf_over_no_cbf": (
            max_second_cbf / max(max_second_no, 1e-12)
        ),
        "max_cbf_correction_variation_rad": float(correction_variation_l2.max()),
        "qp_active_fraction_final_denoise": float(active.mean()),
        "worst_chunk_index": worst_chunk,
    }


def _load_npz(path: Path) -> Dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


def _format_float(value: float) -> str:
    return f"{value:.4f}"


def write_report(output_dir: Path, robots: Sequence[str]) -> None:
    """Create the two-row table and objective worst-chunk visualization."""
    output_dir.mkdir(parents=True, exist_ok=True)
    summaries: Dict[str, Dict[str, Any]] = {}
    arrays_by_robot: Dict[str, Dict[str, np.ndarray]] = {}
    for robot in robots:
        raw_path = output_dir / f"raw_{robot}.npz"
        arrays = _load_npz(raw_path)
        arrays_by_robot[robot] = arrays
        summaries[robot] = compute_continuity_summary(arrays)

    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summaries, indent=2), encoding="utf-8")

    lines = [
        "# Qualitative CBF Trajectory Continuity",
        "",
        (
            "Fixed-seed, noise-matched comparison. The environment executes the CBF "
            "trajectory; the no-CBF trajectory is a counterfactual generated from the "
            "same observation, current joint state, and diffusion RNG state."
        ),
        "",
        "| Robot | Chunks | Max joint step no-CBF (rad) | Max joint step CBF (rad) | Max second-difference ratio | Max CBF correction variation (rad) |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for robot in robots:
        s = summaries[robot]
        lines.append(
            "| "
            + " | ".join(
                [
                    robot.upper(),
                    str(s["num_chunks"]),
                    _format_float(s["max_abs_joint_step_no_cbf_rad"]),
                    _format_float(s["max_abs_joint_step_cbf_rad"]),
                    _format_float(s["max_second_difference_ratio_cbf_over_no_cbf"]),
                    _format_float(s["max_cbf_correction_variation_rad"]),
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            (
                "The plotted chunk for each robot is selected automatically as the chunk "
                "with the largest adjacent variation in the total CBF correction."
            ),
            "",
            "![Worst continuity chunks](continuity_worst_chunks.png)",
            "",
        ]
    )
    (output_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    fig, axes = plt.subplots(
        len(robots), 3, figsize=(12.0, 3.4 * len(robots)), squeeze=False
    )
    for row, robot in enumerate(robots):
        arrays = arrays_by_robot[robot]
        summary = summaries[robot]
        idx = summary["worst_chunk_index"]
        q_current = arrays["q_current"][idx]
        q_cbf = arrays["q_cbf"][idx]
        q_no = arrays["q_no_cbf"][idx]
        h_pre = arrays["h_pre_qp"][idx]
        active = arrays["qp_active"][idx].astype(bool)
        safety_margin = float(np.asarray(arrays["guidance_safety_margin"]).reshape(-1)[0])

        horizon = q_cbf.shape[0]
        dof = q_cbf.shape[1]
        action_x = np.arange(horizon)
        anchored_x = np.arange(-1, horizon)
        anchored_cbf = np.concatenate([q_current[None], q_cbf], axis=0)
        anchored_no = np.concatenate([q_current[None], q_no], axis=0)
        colors = plt.cm.tab10(np.linspace(0.0, 1.0, max(dof, 2)))

        ax = axes[row, 0]
        for joint_idx in range(dof):
            ax.plot(
                anchored_x,
                anchored_no[:, joint_idx],
                linestyle="--",
                linewidth=1.0,
                color=colors[joint_idx],
                alpha=0.55,
            )
            ax.plot(
                anchored_x,
                anchored_cbf[:, joint_idx],
                linestyle="-",
                linewidth=1.2,
                color=colors[joint_idx],
            )
        ax.set_title(f"{robot.upper()} worst chunk #{idx}: joint positions")
        ax.set_xlabel("action index (-1 = current)")
        ax.set_ylabel("joint position (rad)")
        ax.grid(alpha=0.2)
        ax.legend(
            handles=[
                Line2D([0], [0], color="black", linestyle="-", label="CBF"),
                Line2D([0], [0], color="black", linestyle="--", label="no-CBF"),
            ],
            loc="best",
            frameon=False,
        )

        step_cbf, _ = _joint_step_arrays(q_current[None], q_cbf[None])
        step_no, _ = _joint_step_arrays(q_current[None], q_no[None])
        ax = axes[row, 1]
        ax.plot(
            action_x,
            np.linalg.norm(step_no[0], axis=-1) / np.sqrt(dof),
            "--o",
            markersize=3,
            label="no-CBF",
        )
        ax.plot(
            action_x,
            np.linalg.norm(step_cbf[0], axis=-1) / np.sqrt(dof),
            "-o",
            markersize=3,
            label="CBF",
        )
        ax.set_title("Adjacent joint displacement")
        ax.set_xlabel("action index")
        ax.set_ylabel(r"$\|\Delta q\|_2/\sqrt{D}$ (rad)")
        ax.grid(alpha=0.2)
        ax.legend(frameon=False)

        correction = q_cbf - q_no
        correction_norm = np.linalg.norm(correction, axis=-1)
        ax = axes[row, 2]
        correction_line = ax.plot(
            action_x, correction_norm, "-o", markersize=3, color="tab:blue", label="CBF correction"
        )[0]
        ax.set_xlabel("action index")
        ax.set_ylabel(r"$\|q^{CBF}-q^{noCBF}\|_2$ (rad)", color="tab:blue")
        ax.tick_params(axis="y", labelcolor="tab:blue")
        ax.grid(alpha=0.2)
        ax_clearance = ax.twinx()
        clearance_line = ax_clearance.plot(
            action_x, h_pre, "--", color="tab:orange", label="pre-QP clearance"
        )[0]
        margin_line = ax_clearance.axhline(
            safety_margin, color="tab:red", linestyle=":", label="safety margin"
        )
        if np.any(active):
            ax_clearance.scatter(
                action_x[active], h_pre[active], color="tab:red", s=22, zorder=4, label="QP active"
            )
        ax_clearance.set_ylabel("clearance before final QP (m)", color="tab:orange")
        ax_clearance.tick_params(axis="y", labelcolor="tab:orange")
        handles = [correction_line, clearance_line, margin_line]
        labels = [handle.get_label() for handle in handles]
        if np.any(active):
            handles.append(
                Line2D([0], [0], marker="o", linestyle="", color="tab:red", label="QP active")
            )
            labels.append("QP active")
        ax.legend(handles, labels, loc="best", frameon=False, fontsize=8)
        ax.set_title("Correction and final-denoise QP activity")

    fig.tight_layout()
    fig.savefig(output_dir / "continuity_worst_chunks.png", dpi=180)
    plt.close(fig)


def _capture_rng_state(torch_module) -> Dict[str, Any]:
    state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch_module.get_rng_state(),
        "torch_cuda": None,
    }
    if torch_module.cuda.is_available():
        state["torch_cuda"] = torch_module.cuda.get_rng_state_all()
    return state


def _restore_rng_state(torch_module, state: Mapping[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch_module.set_rng_state(state["torch_cpu"])
    if state["torch_cuda"] is not None:
        torch_module.cuda.set_rng_state_all(state["torch_cuda"])


class ContinuityPolicyProxy:
    """Record CBF and exact-noise no-CBF chunks without changing execution."""

    def __init__(self, policy, torch_module):
        self._policy = policy
        self._torch = torch_module
        self._records: list[Dict[str, np.ndarray]] = []

    def __getattr__(self, name: str):
        return getattr(self._policy, name)

    @property
    def records(self) -> Sequence[Mapping[str, np.ndarray]]:
        return self._records

    def predict_action(self, *args, **kwargs):
        policy = self._policy
        torch = self._torch
        if getattr(policy, "guidance_method", None) != "cbf":
            raise RuntimeError("ContinuityPolicyProxy requires guidance_method='cbf'.")
        if not hasattr(policy, "_cbf_qp_fn"):
            raise RuntimeError("CBF QP function is not initialized on the policy.")
        current_joint_angles = kwargs.get("current_joint_angles")
        if current_joint_angles is None:
            raise ValueError("current_joint_angles is required for continuity analysis.")

        rng_before = _capture_rng_state(torch)
        original_qp_fn = policy._cbf_qp_fn
        last_qp: Dict[str, Any] = {}

        def traced_qp(jac, grad_h, h_value, constraint_scale):
            output = original_qp_fn(jac, grad_h, h_value, constraint_scale)
            dq, rhs, denom, feasible = output
            last_qp["h"] = h_value.detach().clone()
            last_qp["dq"] = dq.detach().clone()
            last_qp["rhs"] = rhs.detach().clone()
            last_qp["denom"] = denom.detach().clone()
            last_qp["feasible"] = feasible.detach().clone()
            return output

        policy._cbf_qp_fn = traced_qp
        try:
            result_cbf = policy.predict_action(*args, **kwargs)
        finally:
            policy._cbf_qp_fn = original_qp_fn

        rng_after_cbf = _capture_rng_state(torch)
        q_cbf_full = policy._last_joint_traj.detach().clone()
        if not last_qp:
            raise RuntimeError("No QP call was observed during CBF inference.")

        original_guidance_method = policy.guidance_method
        try:
            _restore_rng_state(torch, rng_before)
            policy.guidance_method = ""
            policy.predict_action(*args, **kwargs)
            q_no_cbf_full = policy._last_joint_traj.detach().clone()
        finally:
            policy.guidance_method = original_guidance_method
            policy._last_joint_traj = q_cbf_full
            _restore_rng_state(torch, rng_after_cbf)

        arm_dof = int(policy.arm_dof)
        q_current = current_joint_angles[..., :arm_dof]
        q_cbf = q_cbf_full[..., :arm_dof]
        q_no_cbf = q_no_cbf_full[..., :arm_dof]
        dq_final = last_qp["dq"][..., :arm_dof]
        if float(policy.guidance_grad_clip) > 0.0:
            dq_final = torch.clamp(
                dq_final,
                min=-float(policy.guidance_grad_clip),
                max=float(policy.guidance_grad_clip),
            )
        qp_active = (last_qp["rhs"] > 0.0) & last_qp["feasible"]

        batch_size = q_cbf.shape[0]
        for batch_idx in range(batch_size):
            self._records.append(
                {
                    "q_current": q_current[batch_idx].detach().cpu().numpy().astype(np.float32),
                    "q_cbf": q_cbf[batch_idx].detach().cpu().numpy().astype(np.float32),
                    "q_no_cbf": q_no_cbf[batch_idx].detach().cpu().numpy().astype(np.float32),
                    "h_pre_qp": last_qp["h"][batch_idx].detach().cpu().numpy().astype(np.float32),
                    "qp_rhs": last_qp["rhs"][batch_idx].detach().cpu().numpy().astype(np.float32),
                    "qp_active": qp_active[batch_idx].detach().cpu().numpy().astype(bool),
                    "dq_final_qp": dq_final[batch_idx].detach().cpu().numpy().astype(np.float32),
                }
            )
        return result_cbf


def save_proxy_records(
    path: Path,
    proxy: ContinuityPolicyProxy,
    metadata: Mapping[str, Any],
) -> None:
    if not proxy.records:
        raise RuntimeError("No continuity records were collected.")
    keys = proxy.records[0].keys()
    arrays = {
        key: np.stack([record[key] for record in proxy.records], axis=0)
        for key in keys
    }
    arrays["guidance_safety_margin"] = np.asarray(
        [metadata["guidance_safety_margin"]], dtype=np.float32
    )
    arrays["metadata_json"] = np.asarray(json.dumps(dict(metadata)))
    np.savez_compressed(path, **arrays)


def _prepare_runtime_checkpoint(runtime_root: Path, checkpoint: Path) -> str:
    ckpt_dir = runtime_root / "ckpt"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    link_path = ckpt_dir / checkpoint.name
    if link_path.exists() or link_path.is_symlink():
        if not link_path.is_symlink() or link_path.resolve() != checkpoint.resolve():
            raise FileExistsError(f"Unexpected runtime checkpoint already exists: {link_path}")
    else:
        link_path.symlink_to(checkpoint)
    return link_path.name


def run_worker(args: argparse.Namespace) -> None:
    import torch

    import eval_sim_single_robot as eval_model

    robot = args.worker_robot
    policy_settings = load_policy_config(args.policy_config)
    robot_uid = ROBOT_UIDS[robot]
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_path = output_dir / f"raw_{robot}.npz"
    if raw_path.exists() and not args.overwrite:
        raise FileExistsError(f"Refusing to overwrite {raw_path}; pass --overwrite to replace it.")

    checkpoint = resolve_checkpoint(args.input, args.ckpt_filename)
    runtime_root = output_dir / "_runtime" / robot
    runtime_ckpt_name = _prepare_runtime_checkpoint(runtime_root, checkpoint)

    original_evaluate = eval_model.evaluate
    original_make_eval_envs = eval_model.make_eval_envs

    def make_eval_envs_without_video(*make_args, **make_kwargs):
        make_kwargs["video_dir"] = None
        return original_make_eval_envs(*make_args, **make_kwargs)

    def evaluate_with_continuity(
        n,
        cfg,
        policy,
        eval_envs,
        steps_per_inference,
        add_guidance,
        joint_space,
        device,
        **evaluate_kwargs,
    ):
        proxy = ContinuityPolicyProxy(policy, torch)
        metrics = original_evaluate(
            n,
            cfg,
            proxy,
            eval_envs,
            steps_per_inference,
            add_guidance,
            joint_space,
            device,
            **evaluate_kwargs,
        )
        metadata = {
            "robot": robot,
            "robot_uid": robot_uid,
            "checkpoint": str(checkpoint),
            "env_id": args.env_id,
            "num_env": args.num_env,
            "num_eval_episodes": args.num_eval_episodes,
            "steps_per_inference": args.steps_per_inference,
            "max_episode_steps": args.max_episode_steps,
            "policy_config": args.policy_config,
            "guidance_safety_margin": policy_settings["guidance_safety_margin"],
            "guidance_task_rot_weight": policy_settings[
                "guidance_task_rot_weight"
            ],
            "control_mode": args.control_mode,
        }
        save_proxy_records(raw_path, proxy, metadata)
        return metrics

    eval_model.evaluate = evaluate_with_continuity
    eval_model.make_eval_envs = make_eval_envs_without_video
    click_args = [
        "--input", str(runtime_root),
        "--ckpt_filename", runtime_ckpt_name,
        "--env_id", args.env_id,
        "--robot_uids", robot_uid,
        "--sim_backend", args.sim_backend,
        "--control_mode", args.control_mode,
        "--num_env", str(args.num_env),
        "--num_eval_episodes", str(args.num_eval_episodes),
        "--obs_mode", args.obs_mode,
        "--render_mode", args.render_mode,
        "--steps_per_inference", str(args.steps_per_inference),
        "--max_episode_steps", str(args.max_episode_steps),
        "--obstacle",
        "--policy-config", args.policy_config,
    ]
    eval_model.main.main(args=click_args, standalone_mode=False)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Qualitative, noise-matched trajectory-continuity analysis."
    )
    parser.add_argument("--input", "-i", required=True)
    parser.add_argument("--ckpt-filename", "-f", default="latest")
    parser.add_argument("--env-id", "-e", default="MakeIcedCoffee-v1")
    parser.add_argument("--robots", nargs="+", choices=tuple(ROBOT_UIDS), default=["panda", "ur5"])
    parser.add_argument("--sim-backend", default="physx_cpu")
    parser.add_argument("--control-mode", "-c", default="pd_joint_pos")
    parser.add_argument("--num-env", "-n", type=int, default=1)
    parser.add_argument("--num-eval-episodes", "-ne", type=int, default=1)
    parser.add_argument("--obs-mode", default="rgb")
    parser.add_argument("--render-mode", default="rgb_array")
    parser.add_argument("--steps-per-inference", "-si", type=int, default=0)
    parser.add_argument("--max-episode-steps", "-mes", type=int, default=500)
    parser.add_argument(
        "--policy-config",
        default=str(ROOT_DIR / "configs" / "policy" / "continuity_cbf.yaml"),
        help="YAML file containing the joint-space guidance settings.",
    )
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--worker-robot", choices=tuple(ROBOT_UIDS), default=None, help=argparse.SUPPRESS)
    return parser


def _default_output_dir(input_path: str) -> Path:
    input_resolved = Path(input_path).expanduser().resolve()
    experiment_root = input_resolved if input_resolved.is_dir() else input_resolved.parent.parent
    return experiment_root / "continuity_results" / "panda_ur5"


def _worker_command(args: argparse.Namespace, robot: str, output_dir: Path) -> list[str]:
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--input", str(Path(args.input).expanduser().resolve()),
        "--ckpt-filename", args.ckpt_filename,
        "--env-id", args.env_id,
        "--sim-backend", args.sim_backend,
        "--control-mode", args.control_mode,
        "--num-env", str(args.num_env),
        "--num-eval-episodes", str(args.num_eval_episodes),
        "--obs-mode", args.obs_mode,
        "--render-mode", args.render_mode,
        "--steps-per-inference", str(args.steps_per_inference),
        "--max-episode-steps", str(args.max_episode_steps),
        "--policy-config", args.policy_config,
        "--output-dir", str(output_dir),
        "--worker-robot", robot,
    ]
    if args.overwrite:
        command.append("--overwrite")
    return command


def main(argv: Iterable[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    args.policy_config = load_policy_config(args.policy_config)["config_path"]
    if args.num_env < 1:
        raise ValueError("--num-env must be positive.")
    if args.num_eval_episodes < 1:
        raise ValueError("--num-eval-episodes must be positive.")
    if args.num_eval_episodes % args.num_env != 0:
        raise ValueError("--num-eval-episodes must be divisible by --num-env.")
    output_dir = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir is not None
        else _default_output_dir(args.input)
    )

    if args.worker_robot is not None:
        args.output_dir = str(output_dir)
        run_worker(args)
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    for robot in args.robots:
        raw_path = output_dir / f"raw_{robot}.npz"
        if raw_path.exists() and not args.overwrite:
            raise FileExistsError(
                f"Refusing to overwrite {raw_path}; pass --overwrite to replace existing results."
            )
        subprocess.run(
            _worker_command(args, robot, output_dir),
            check=True,
            cwd=ROOT_DIR,
        )
    write_report(output_dir, args.robots)
    print(f"Continuity report: {output_dir / 'report.md'}")
    print(f"Continuity figure: {output_dir / 'continuity_worst_chunks.png'}")


if __name__ == "__main__":
    main()
