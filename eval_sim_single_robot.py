#!/usr/bin/env python3
"""Evaluate one robot in ManiSkill using a shared EmbodiSteer policy YAML."""

# %%
import sys
import os
import json
from datetime import datetime, timezone
from functools import partial

ROOT_DIR = os.path.dirname(__file__)
sys.path.append(ROOT_DIR)
os.chdir(ROOT_DIR)

import warnings
warnings.filterwarnings("ignore", message=".*pkg_resources is deprecated.*")
warnings.filterwarnings("ignore", message=".*CUDA reports that you have.*")

# %%
import click
import dill
import hydra
import numpy as np
import torch
import pytorch_kinematics as pk
from omegaconf import OmegaConf
from omegaconf import open_dict
from embodisteer.runtime_config import (
    PolicyConfigError,
    ee_policy_overrides,
    joint_policy_overrides,
    load_policy_config,
    policy_target,
)
from embodisteer.evaluation import evaluation_subdir, workflow_result_dir
from embodisteer.kinematics.robot_config import infer_robot_cfg_name
from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.workspace.base_workspace import BaseWorkspace
from umi.real_world.real_inference_util import (
    get_real_umi_obs_dict,
)
from curobo.util_file import get_robot_configs_path, get_assets_path, join_path, load_yaml
from scripts_maniskill.utils import (
    make_eval_envs, 
    maniskill_to_umi_env_obs, 
    get_maniskill_umi_action, 
    get_maniskill_joint_action,
    evaluate,
)
from scripts_maniskill.obstacle_observation_noise import (
    ObstacleObservationNoiseWrapper,
)

from mani_skill.utils.wrappers.flatten import FlattenRGBDObservationWrapper

OmegaConf.register_new_resolver("eval", eval, replace=True)


def _infer_robot_kinematic_args(robot_cfg_name: str):
    robot_cfg_dict = load_yaml(join_path(get_robot_configs_path(), robot_cfg_name))["robot_cfg"]
    kin_cfg = robot_cfg_dict.get("kinematics", {})
    urdf_rel = kin_cfg.get("urdf_path")
    if urdf_rel is None:
        return None, kin_cfg.get("ee_link", "eef"), None

    robot_urdf_path = join_path(get_assets_path(), urdf_rel)
    ee_link_name = kin_cfg.get("ee_link", "eef")

    arm_dof = None
    try:
        with open(robot_urdf_path, "r") as f:
            urdf_str = f.read()
        try:
            chain = pk.build_serial_chain_from_urdf(urdf_str, ee_link_name)
        except ValueError:
            chain = pk.build_serial_chain_from_urdf(urdf_str.encode("utf-8"), ee_link_name)
        arm_dof = int(chain.n_joints)
    except Exception:
        arm_dof = None

    return robot_urdf_path, ee_link_name, arm_dof


@click.command()
@click.option('--input', '-i', required=True, help='Path to checkpoint and experiment results')
@click.option('--ckpt_filename', '-f', required=True, help='Checkpoint filename within the experiment folder')
@click.option(
    '--output-dir',
    type=click.Path(file_okay=False),
    default=None,
    help='Workflow result root. Omit to preserve the experiment-local legacy layout.',
)
@click.option(
    '--profile-name',
    default=None,
    help='Workflow profile name used to isolate results under --output-dir.',
)
@click.option('--env_id', '-e', required=True, help='ManiSkill environment id')
@click.option('--robot_uids', '-r', required=True, help='Robot UIDs in ManiSkill env')
@click.option('--sim_backend', '-s', default='physx_cpu', help='Simulation backend for ManiSkill env')
@click.option('--control_mode', '-c', default=None, help='ManiSkill control mode; inferred from the policy config when omitted.')
@click.option('--num_env', '-n', default=10, type=int, help='Number of parallel environments')
@click.option('--num_eval_episodes', '-ne', default=100, type=int, help='Number of evaluation episodes')
@click.option('--obs_mode', '-o', default='rgb', help='Observation mode for ManiSkill env')
@click.option('--render_mode', '-rm', default='all', help='Render mode for ManiSkill env')
@click.option('--steps_per_inference', '-si', default=0, type=int, help="Number of predicted actions to execute per policy call. Use 0 to execute cfg.task.action_horizon.")
@click.option('--max_episode_steps', '-mes', default=500, type=int, help="Max episode steps for evaluation.")
@click.option('--obstacle', is_flag=True, help="Whether to use harder environment setting with more obstacles and narrower workspace.")
@click.option(
    '--obstacle-observation-noise',
    '--obstacle_observation_noise',
    nargs=3,
    type=float,
    default=None,
    metavar='POS_STD_M SIZE_STD_M ROT_STD_RAD',
    help=(
        "Per-episode Gaussian noise for observed obstacle center, full edge "
        "length, and orientation angle. Omit to use exact obstacle geometry."
    ),
)
@click.option(
    '--policy-config',
    type=click.Path(dir_okay=False),
    default=os.path.join(ROOT_DIR, 'configs', 'policy', 'embodisteer.yaml'),
    show_default=True,
    help='YAML file containing inference-space, guidance, baseline and IK settings.',
)
def main(
    input,
    ckpt_filename,
    output_dir,
    profile_name,
    env_id,
    robot_uids,
    sim_backend,
    control_mode,
    num_env,
    num_eval_episodes,
    obs_mode,
    render_mode,
    steps_per_inference,
    max_episode_steps,
    obstacle,
    obstacle_observation_noise,
    policy_config,
):
    started_at = datetime.now(timezone.utc).isoformat()
    if (output_dir is None) != (profile_name is None):
        raise click.BadParameter(
            '--output-dir and --profile-name must be supplied together.'
        )
    try:
        policy_settings = load_policy_config(policy_config)
    except PolicyConfigError as exc:
        raise click.BadParameter(str(exc), param_hint='--policy-config') from exc

    inference_space = policy_settings['inference_space']
    guidance = policy_settings['guidance']
    baseline_method = policy_settings['baseline_method']
    batch_sampling_num = policy_settings['batch_sampling_num']
    jm2d_num_samples = policy_settings['jm2d_num_samples']
    jm2d_temperature = policy_settings['jm2d_temperature']
    jm2d_eta = policy_settings['jm2d_eta']

    # Validate method/runtime combinations before touching checkpoint data or
    # creating a simulation environment. Configuration mistakes should remain
    # deterministic and safe even when the checkpoint path is unavailable.
    use_baseline = (baseline_method != '')
    use_guidance = (guidance != '')
    is_joint_space = (inference_space == 'joint') or use_baseline
    if control_mode is None:
        control_mode = 'pd_joint_pos' if is_joint_space else 'pd_ee_pose'
    if is_joint_space and not control_mode.startswith('pd_joint'):
        raise click.BadParameter(
            "Joint-space and baseline policy configs require a pd_joint* "
            "control mode so the configured joint trajectory is executed."
        )
    policy_needs_obstacles = use_baseline or use_guidance
    if policy_needs_obstacles and not obstacle:
        raise click.BadParameter(
            "The selected policy config requires --obstacle."
        )
    obstacle_noise_enabled = obstacle_observation_noise is not None
    if obstacle_noise_enabled:
        if any(std < 0.0 for std in obstacle_observation_noise):
            raise click.BadParameter(
                "--obstacle-observation-noise values must be non-negative."
            )
        if not any(std > 0.0 for std in obstacle_observation_noise):
            raise click.BadParameter(
                "--obstacle-observation-noise must contain at least one positive value."
            )
        if not obstacle:
            raise click.BadParameter(
                "--obstacle-observation-noise requires --obstacle."
            )
        if not policy_needs_obstacles:
            raise click.BadParameter(
                "--obstacle-observation-noise requires guidance or a baseline "
                "in --policy-config."
            )

    # load checkpoint
    exp_path = input
    if os.path.isfile(input):
        ckpt_path = input
        exp_path = os.path.dirname(os.path.dirname(ckpt_path))
    else:
        ckpt_path = os.path.join(exp_path, 'ckpt', ckpt_filename)
        if not (ckpt_path.endswith('.ckpt') or ckpt_path.endswith('.pth')):
            ckpt_path_ckpt = ckpt_path + '.ckpt'
            ckpt_path_pth = ckpt_path + '.pth'
            if os.path.exists(ckpt_path_ckpt):
                ckpt_path = ckpt_path_ckpt
            elif os.path.exists(ckpt_path_pth):
                ckpt_path = ckpt_path_pth
            else:
                ckpt_path = ckpt_path_ckpt
    assert os.path.exists(ckpt_path), f"Checkpoint {ckpt_path} does not exist."
    payload = torch.load(open(ckpt_path, 'rb'), map_location='cpu', pickle_module=dill)
    cfg = payload['cfg']
    if steps_per_inference <= 0:
        steps_per_inference = int(cfg.task.action_horizon)

    cfg.policy._target_ = policy_target(policy_settings)
    if use_baseline:
        robot_cfg_name = infer_robot_cfg_name(robot_uids)
        robot_urdf_path, ee_link_name, arm_dof = _infer_robot_kinematic_args(robot_cfg_name)
        with open_dict(cfg.policy):
            for key, value in joint_policy_overrides(policy_settings).items():
                cfg.policy[key] = value
            cfg.policy.baseline_method = baseline_method
            cfg.policy.batch_sampling_num = batch_sampling_num
            # JM2D-specific constructor arguments are only injected for the
            # JM2D target.  Keeping them out of the other baseline configs
            # avoids leaking unrelated fields through ``**kwargs``.
            if baseline_method == 'jm2d':
                cfg.policy.jm2d_num_samples = jm2d_num_samples
                cfg.policy.jm2d_temperature = jm2d_temperature
                cfg.policy.jm2d_eta = jm2d_eta
            cfg.policy.robot_uid = robot_uids
            cfg.policy.robot_cfg_name = robot_cfg_name
            if robot_urdf_path is not None:
                cfg.policy.robot_urdf_path = robot_urdf_path
            cfg.policy.ee_link_name = ee_link_name
            if arm_dof is not None:
                cfg.policy.arm_dof = arm_dof
    elif inference_space == 'joint':
        robot_cfg_name = infer_robot_cfg_name(robot_uids)
        robot_urdf_path, ee_link_name, arm_dof = _infer_robot_kinematic_args(robot_cfg_name)
        with open_dict(cfg.policy):
            for key, value in joint_policy_overrides(policy_settings).items():
                cfg.policy[key] = value
            cfg.policy.robot_uid = robot_uids
            cfg.policy.robot_cfg_name = robot_cfg_name
            if robot_urdf_path is not None:
                cfg.policy.robot_urdf_path = robot_urdf_path
            cfg.policy.ee_link_name = ee_link_name
            if arm_dof is not None:
                cfg.policy.arm_dof = arm_dof
    else:
        # EE-space inference (default). guidance is either '' or 'gd'.
        with open_dict(cfg.policy):
            for key, value in ee_policy_overrides(policy_settings).items():
                cfg.policy[key] = value
    # Ensure every target (including checkpoints with older stored configs)
    # receives the canonical inference step count from the selected profile.
    with open_dict(cfg.policy):
        cfg.policy.num_inference_steps = policy_settings['num_inference_steps']
    print("policy_config:", policy_settings['config_path'])
    print(
        "method:",
        f"inference_space={inference_space}",
        f"guidance={guidance or 'none'}",
        f"baseline={baseline_method or 'none'}",
    )
    print("model_name:", cfg.policy.obs_encoder.model_name)
    print("dataset_path:", cfg.task.dataset.dataset_path)

    # ── Compute subdir name (shared by video_dir and log_dir) ───────────
    eval_subdir = evaluation_subdir(
        inference_space=inference_space,
        guidance=guidance,
        baseline_method=baseline_method,
        obstacle=obstacle,
        obstacle_observation_noise=(
            obstacle_observation_noise if obstacle_noise_enabled else None
        ),
    )

    if output_dir is not None:
        log_dir = str(workflow_result_dir(output_dir, profile_name, robot_uids))
    else:
        log_dir = os.path.join(exp_path, 'eval_results', robot_uids, eval_subdir)
    video_dir = os.path.join(log_dir, 'videos')
    os.makedirs(video_dir, exist_ok=True)
    env_kwargs = dict(
        robot_uids=robot_uids,
        control_mode=control_mode,
        obs_mode=obs_mode,
        render_mode=render_mode,
        max_episode_steps=max_episode_steps,
        sensor_configs=dict(shader_pack="default"),
        reward_mode="normalized_dense",
    )
    if obstacle:
        env_kwargs['harder'] = True
    track_collisions = obstacle
    other_kwargs = dict(obs_horizon=cfg.task.shape_meta.obs.camera0_rgb.horizon)
    env_wrappers = [FlattenRGBDObservationWrapper]
    if obstacle_noise_enabled:
        pos_std, size_std, rot_std = obstacle_observation_noise
        env_wrappers.append(
            partial(
                ObstacleObservationNoiseWrapper,
                position_std_m=pos_std,
                size_std_m=size_std,
                rotation_std_rad=rot_std,
            )
        )
        print(
            "Obstacle observation noise enabled: "
            f"position_std={pos_std:.6g} m, "
            f"size_std={size_std:.6g} m (full edge length), "
            f"rotation_std={rot_std:.6g} rad; sampled once per episode."
        )
    env = make_eval_envs(
        env_id,
        num_env,
        sim_backend,
        env_kwargs,
        other_kwargs,
        video_dir=video_dir,
        wrappers=env_wrappers,
        track_collisions=track_collisions,
        info_on_video=True,
    )

    # creating model
    # have to be done after fork to prevent 
    # duplicating CUDA context with ffmpeg nvenc
    cls = hydra.utils.get_class(cfg._target_)
    workspace = cls(cfg)
    workspace: BaseWorkspace
    workspace.load_payload(payload, exclude_keys=None, include_keys=None)

    policy = workspace.model
    if cfg.training.use_ema:
        policy = workspace.ema_model
    obs_pose_rep = cfg.task.pose_repr.obs_pose_repr
    action_pose_repr = cfg.task.pose_repr.action_pose_repr
    print('obs_pose_rep', obs_pose_rep)
    print('action_pose_repr', action_pose_repr)

    device = torch.device('cuda')
    policy.eval().to(device)

    print("Warming up policy inference")
    obs, info = env.reset()
    # 获取当前关节角度（在转换obs之前）
    current_joint_angles_warmup = torch.tensor(obs["joint_state"][:, -1, :]).to(device) if is_joint_space else None
    obs = maniskill_to_umi_env_obs(obs)
    with torch.no_grad():
        policy.reset()
        # [新增] 获取当前位姿作为临时的 start_pose 用于预热
        episode_start_pose = [np.concatenate([
            obs['robot0_eef_pos'][:, -1],
            obs['robot0_eef_rot_axis_angle'][:, -1]
        ], axis=-1)]
        obs_dict_np = get_real_umi_obs_dict(
            env_obs=obs, shape_meta=cfg.task.shape_meta,
            obs_pose_repr=obs_pose_rep,
            episode_start_pose=episode_start_pose,
            batched=True,
        )
        obs_dict = dict_apply(obs_dict_np,
            lambda x: torch.from_numpy(x).to(device))
        if is_joint_space or policy_needs_obstacles:
            episode_start_pose_tensor = torch.from_numpy(episode_start_pose[0]).to(device)
        else:
            episode_start_pose_tensor = None
        result = policy.predict_action(obs_dict, env_batched=False, chunk_start_pose=episode_start_pose_tensor, current_joint_angles=current_joint_angles_warmup)
        if control_mode.startswith("pd_joint"):
            action = result["joint_action_pred"].detach().to("cpu").numpy()
            single_action_space = getattr(env, "single_action_space", env.action_space)
            action = get_maniskill_joint_action(action, action_space=single_action_space)
            assert action.shape[-1] == single_action_space.shape[-1]
        else:
            action = result['action_pred'].detach().to('cpu').numpy()
            assert action.shape[-1] == 10
            action = get_maniskill_umi_action(action, obs, action_pose_repr, batched=True)
            single_action_space = getattr(env, "single_action_space", env.action_space)
            assert action.shape[-1] == single_action_space.shape[-1]
        del result

    print('Ready! Start evaluation.')
    eval_metrics = evaluate(
        num_eval_episodes, cfg, policy, env, steps_per_inference,
        policy_needs_obstacles,  # add_guidance param: triggers obstacle_info fetch + chunk_pose
        is_joint_space,          # joint_space param: triggers chunk_pose
        device,
        control_mode=control_mode,
    )
    jm2d_ik_pose_success_rate = None
    jm2d_ik_trajectory_success_rate = None
    if baseline_method == 'jm2d' and 'jm2d_ik_pose_success_rate' in eval_metrics:
        jm2d_ik_pose_success_rate = float(
            np.asarray(eval_metrics['jm2d_ik_pose_success_rate']).mean()
        )
        jm2d_ik_trajectory_success_rate = float(
            np.asarray(eval_metrics['jm2d_ik_trajectory_success_rate']).mean()
        )
        print(
            "JM2D IK success (all evaluation inferences): "
            f"pose={jm2d_ik_pose_success_rate:.4f}, "
            f"full_trajectory={jm2d_ik_trajectory_success_rate:.4f}"
        )
    print("Evaluation results over {} episodes:".format(num_eval_episodes))
    success_once_rate = float(np.mean(eval_metrics["success_once"]))
    success_at_end_rate = float(np.mean(eval_metrics["success_at_end"]))
    print(f"{success_once_rate=}")
    print(f"{success_at_end_rate=}")

    # ---- collisions ----
    if "collision_count" in eval_metrics:
        avg_collision = float(np.mean(eval_metrics["collision_count"]))
        std_collision = float(np.std(eval_metrics["collision_count"]))
        print(f"avg collisions/ep: {avg_collision:.2f}  std: {std_collision:.2f}")
    else:
        avg_collision = None
        std_collision = None

    # ---- max reward ----
    if "max_reward" in eval_metrics:
        avg_max_reward = float(np.mean(eval_metrics["max_reward"]))
        std_max_reward = float(np.std(eval_metrics["max_reward"]))
        print(f"avg max reward/ep: {avg_max_reward:.4f}  std: {std_max_reward:.4f}")
    else:
        avg_max_reward = None
        std_max_reward = None

    # ---- first success step ----
    if "first_success_step" in eval_metrics:
        fss = eval_metrics["first_success_step"].flatten()
        success_mask = fss >= 0
        if success_mask.sum() > 0:
            avg_first_success_step = float(np.mean(fss[success_mask]))
            std_first_success_step = float(np.std(fss[success_mask]))
        else:
            avg_first_success_step = None
            std_first_success_step = None
        print(f"avg first success step (among successes): {avg_first_success_step}  std: {std_first_success_step}")
    else:
        avg_first_success_step = None
        std_first_success_step = None

    # ---- fail rate among all episodes ----
    if "collision_count" in eval_metrics:
        fail_rate_all = float((eval_metrics["collision_count"] > 0).mean())
        print(f"fail rate among all episodes: {fail_rate_all}")
    else:
        fail_rate_all = None

    # ---- fail rate among non-success episodes ----
    if "fail_once" in eval_metrics:
        success_once = eval_metrics["success_once"]
        fail_once = eval_metrics["fail_once"]
        non_success_mask = success_once == 0
        if non_success_mask.sum() > 0:
            fail_rate_among_non_success = float(fail_once[non_success_mask].mean())
        else:
            fail_rate_among_non_success = None  # undefined when all episodes succeeded
        print(f"fail rate among non-success episodes: {fail_rate_among_non_success}")
    else:
        fail_rate_among_non_success = None

    log_filename = os.path.join(log_dir, 'eval_results.txt')
    os.makedirs(os.path.dirname(log_filename), exist_ok=True)
    with open(log_filename, "w") as f:
        f.write("reset_protocol: unseeded_env_reset\n")
        if obstacle_noise_enabled:
            pos_std, size_std, rot_std = obstacle_observation_noise
            f.write(f"obstacle_position_noise_std_m: {pos_std}\n")
            f.write(f"obstacle_size_noise_std_m: {size_std}\n")
            f.write(f"obstacle_rotation_noise_std_rad: {rot_std}\n")
            f.write("obstacle_noise_temporal_mode: per_episode\n")
        f.write(f"success_once_rate: {success_once_rate}\n")
        f.write(f"success_at_end_rate: {success_at_end_rate}\n")
        if jm2d_ik_pose_success_rate is not None:
            f.write(
                f"jm2d_ik_pose_success_rate: {jm2d_ik_pose_success_rate}\n"
            )
            f.write(
                "jm2d_ik_trajectory_success_rate: "
                f"{jm2d_ik_trajectory_success_rate}\n"
            )

        if avg_collision is not None:
            f.write(f"avg_collision_per_episode: {avg_collision}\n")
            f.write(f"std_collision_per_episode: {std_collision}\n")
        else:
            f.write("avg_collision_per_episode: N/A\n")
            f.write("std_collision_per_episode: N/A\n")

        if avg_max_reward is not None:
            f.write(f"avg_max_reward_per_episode: {avg_max_reward}\n")
            f.write(f"std_max_reward_per_episode: {std_max_reward}\n")
        else:
            f.write("avg_max_reward_per_episode: N/A\n")
            f.write("std_max_reward_per_episode: N/A\n")

        if fail_rate_among_non_success is not None:
            f.write(f"fail_rate_among_non_success: {fail_rate_among_non_success}\n")
        else:
            f.write("fail_rate_among_non_success: N/A\n")

        if fail_rate_all is not None:
            f.write(f"fail_rate_all: {fail_rate_all}\n")
        else:
            f.write("fail_rate_all: N/A\n")

        if avg_first_success_step is not None:
            f.write(f"avg_first_success_step: {avg_first_success_step}\n")
            f.write(f"std_first_success_step: {std_first_success_step}\n")
        else:
            f.write("avg_first_success_step: N/A\n")
            f.write("std_first_success_step: N/A\n")

    # Preserve episode-level numeric arrays for statistically correct pooling
    # and future analysis. The text file above remains backward-compatible with
    # eval_sim_multi_robots.py.
    episode_metrics_path = os.path.join(log_dir, 'episode_metrics.npz')
    np.savez_compressed(
        episode_metrics_path,
        **{
            key: np.asarray(value)
            for key, value in eval_metrics.items()
            if np.asarray(value).dtype != object
        },
    )
    structured_metrics = {
        'success_once_rate': success_once_rate,
        'success_at_end_rate': success_at_end_rate,
        'avg_collision_per_episode': avg_collision,
        'std_collision_per_episode': std_collision,
        'avg_max_reward_per_episode': avg_max_reward,
        'std_max_reward_per_episode': std_max_reward,
        'fail_rate_among_non_success': fail_rate_among_non_success,
        'fail_rate_all': fail_rate_all,
        'avg_first_success_step': avg_first_success_step,
        'std_first_success_step': std_first_success_step,
        'inference_calls': int(np.asarray(eval_metrics['inference_calls']).sum()),
        'inference_total_time': float(np.asarray(eval_metrics['inference_total_time']).sum()),
        'inference_frequency': float(np.asarray(eval_metrics['inference_frequency']).mean()),
    }
    if jm2d_ik_pose_success_rate is not None:
        structured_metrics.update(
            {
                'jm2d_ik_pose_success_rate': jm2d_ik_pose_success_rate,
                'jm2d_ik_trajectory_success_rate': jm2d_ik_trajectory_success_rate,
                'jm2d_ik_pose_success_rate_sample_count': int(
                    np.asarray(eval_metrics['jm2d_ik_pose_success_rate_sample_count']).sum()
                ),
                'jm2d_ik_trajectory_success_rate_sample_count': int(
                    np.asarray(eval_metrics['jm2d_ik_trajectory_success_rate_sample_count']).sum()
                ),
            }
        )
    payload = {
        'schema_version': 1,
        'status': 'completed',
        'started_at': started_at,
        'finished_at': datetime.now(timezone.utc).isoformat(),
        'profile_name': profile_name,
        'policy_config': policy_settings,
        'checkpoint': {
            'path': os.path.abspath(ckpt_path),
        },
        'evaluation': {
            'env_id': env_id,
            'robot_uid': robot_uids,
            'sim_backend': sim_backend,
            'control_mode': control_mode,
            'num_env': num_env,
            'num_eval_episodes': num_eval_episodes,
            'reset_protocol': 'unseeded_env_reset',
            'obs_mode': obs_mode,
            'render_mode': render_mode,
            'steps_per_inference': steps_per_inference,
            'max_episode_steps': max_episode_steps,
            'obstacle': obstacle,
            'obstacle_observation_noise': obstacle_observation_noise,
        },
        'metrics': structured_metrics,
        'episode_metrics': os.path.basename(episode_metrics_path),
    }
    metrics_tmp = os.path.join(log_dir, '.metrics.json.tmp')
    with open(metrics_tmp, 'w', encoding='utf-8') as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write('\n')
    os.replace(metrics_tmp, os.path.join(log_dir, 'metrics.json'))

# %%
if __name__ == '__main__':
    main()
