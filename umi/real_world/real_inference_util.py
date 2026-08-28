from typing import Dict, Callable, Tuple, List
import numpy as np
import collections
from diffusion_policy.common.cv2_util import get_image_transform
from diffusion_policy.common.pose_repr_util import (
    compute_relative_pose, 
    convert_pose_mat_rep, 
)
from embodisteer.kinematics import batched_convert_pose_mat_rep
from umi.common.pose_util import (
    pose_to_mat, mat_to_pose, 
    mat_to_pose10d, pose10d_to_mat)
from diffusion_policy.model.common.rotation_transformer import \
    RotationTransformer
import time

def env_obs_to_pose_mat(
    env_obs: Dict[str, np.ndarray],
    robot_prefix: str='robot0',
) -> np.ndarray:
    if env_obs[f'{robot_prefix}_eef_pos'].ndim == 2:
        batched = False
        pos = env_obs[f'{robot_prefix}_eef_pos']
        rot = env_obs[f'{robot_prefix}_eef_rot_axis_angle']
    elif env_obs[f'{robot_prefix}_eef_pos'].ndim == 3:
        batched = True
        batch_size = env_obs[f'{robot_prefix}_eef_pos'].shape[0]
        time_horizon = env_obs[f'{robot_prefix}_eef_pos'].shape[1]
        pos = env_obs[f'{robot_prefix}_eef_pos'].reshape(-1,3)
        rot = env_obs[f'{robot_prefix}_eef_rot_axis_angle'].reshape(-1,3)
    else:
        raise ValueError("Invalid pose shape")
    pose = np.concatenate([pos, rot], axis=-1)
    pose_mat = pose_to_mat(pose)
    if batched:
        pose_mat = pose_mat.reshape(batch_size, time_horizon, 4, 4)
    return pose_mat

def get_real_obs_resolution(
        shape_meta: dict
        ) -> Tuple[int, int]:
    out_res = None
    obs_shape_meta = shape_meta['obs']
    for key, attr in obs_shape_meta.items():
        type = attr.get('type', 'low_dim')
        shape = attr.get('shape')
        if type == 'rgb':
            co,ho,wo = shape
            if out_res is None:
                out_res = (wo, ho)
            assert out_res == (wo, ho)
    return out_res


def get_real_obs_dict(
        env_obs: Dict[str, np.ndarray], 
        shape_meta: dict,
        ) -> Dict[str, np.ndarray]:
    obs_dict_np = dict()
    obs_shape_meta = shape_meta['obs']
    for key, attr in obs_shape_meta.items():
        type = attr.get('type', 'low_dim')
        shape = attr.get('shape')
        if type == 'rgb':
            this_imgs_in = env_obs[key]
            t,hi,wi,ci = this_imgs_in.shape
            co,ho,wo = shape
            assert ci == co
            out_imgs = this_imgs_in
            if (ho != hi) or (wo != wi) or (this_imgs_in.dtype == np.uint8):
                tf = get_image_transform(
                    input_res=(wi,hi), 
                    output_res=(wo,ho), 
                    bgr_to_rgb=False)
                out_imgs = np.stack([tf(x) for x in this_imgs_in])
                if this_imgs_in.dtype == np.uint8:
                    out_imgs = out_imgs.astype(np.float32) / 255
            # THWC to TCHW
            obs_dict_np[key] = np.moveaxis(out_imgs,-1,1)
        elif type == 'low_dim':
            this_data_in = env_obs[key]
            obs_dict_np[key] = this_data_in
    return obs_dict_np


def get_real_umi_obs_dict(
        env_obs: Dict[str, np.ndarray], 
        shape_meta: dict,
        obs_pose_repr: str='abs',
        tx_robot1_robot0: np.ndarray=None,
        episode_start_pose: List[np.ndarray]=None,
        batched: bool=False
        ) -> Dict[str, np.ndarray]:
    obs_dict_np = dict()
    # process non-pose
    obs_shape_meta = shape_meta['obs']
    robot_prefix_map = collections.defaultdict(list)
    for key, attr in obs_shape_meta.items():
        type = attr.get('type', 'low_dim')
        shape = attr.get('shape')
        if type == 'rgb':
            this_imgs_in = env_obs[key]
            hi,wi,ci = this_imgs_in.shape[-3:]
            co,ho,wo = shape
            assert ci == co
            out_imgs = this_imgs_in
            if (ho != hi) or (wo != wi):
                tf = get_image_transform(
                    input_res=(wi,hi), 
                    output_res=(wo,ho), 
                    bgr_to_rgb=False)
                out_imgs = np.stack([tf(x) for x in this_imgs_in])
            if this_imgs_in.dtype == np.uint8:
                out_imgs = out_imgs.astype(np.float32) / 255
            # THWC to TCHW
            obs_dict_np[key] = np.moveaxis(out_imgs,-1,-3)
        elif type == 'low_dim' and ('eef' not in key):
            this_data_in = env_obs[key]
            obs_dict_np[key] = this_data_in
            # handle multi-robots
            ks = key.split('_')
            if ks[0].startswith('robot'):
                robot_prefix_map[ks[0]].append(key)

    convert_pose_mat_rep_func = batched_convert_pose_mat_rep if batched else convert_pose_mat_rep

    # generate relative pose
    for robot_prefix in robot_prefix_map.keys():
        # convert pose to mat
        pose_mat = env_obs_to_pose_mat(env_obs, robot_prefix)

        # solve reltaive obs
        obs_pose_mat = convert_pose_mat_rep_func(
            pose_mat, 
            base_pose_mat=pose_mat[:, -1] if batched else pose_mat[-1],
            pose_rep=obs_pose_repr,
            backward=False)

        obs_pose = mat_to_pose10d(obs_pose_mat)
        obs_dict_np[robot_prefix + '_eef_pos'] = obs_pose[...,:3]
        obs_dict_np[robot_prefix + '_eef_rot_axis_angle'] = obs_pose[...,3:]
    
    # generate pose relative to other robot
    n_robots = len(robot_prefix_map)
    for robot_id in range(n_robots):
        # convert pose to mat
        assert f'robot{robot_id}' in robot_prefix_map
        tx_robota_tcpa = env_obs_to_pose_mat(env_obs, f'robot{robot_id}')
        for other_robot_id in range(n_robots):
            if robot_id == other_robot_id:
                continue
            tx_robotb_tcpb = pose_to_mat(np.concatenate([
                env_obs[f'robot{other_robot_id}_eef_pos'],
                env_obs[f'robot{other_robot_id}_eef_rot_axis_angle']
            ], axis=-1))
            tx_robota_robotb = tx_robot1_robot0
            if robot_id == 0:
                tx_robota_robotb = np.linalg.inv(tx_robot1_robot0)
            tx_robota_tcpb = tx_robota_robotb @ tx_robotb_tcpb

            rel_obs_pose_mat = convert_pose_mat_rep_func(
                tx_robota_tcpa,
                base_pose_mat=tx_robota_tcpb[:, -1] if batched else tx_robota_tcpb[-1],
                pose_rep='relative',
                backward=False)
            rel_obs_pose = mat_to_pose10d(rel_obs_pose_mat)
            obs_dict_np[f'robot{robot_id}_eef_pos_wrt{other_robot_id}'] = rel_obs_pose[...,:3]
            obs_dict_np[f'robot{robot_id}_eef_rot_axis_angle_wrt{other_robot_id}'] = rel_obs_pose[...,3:]

    # generate relative pose with respect to episode start
    if episode_start_pose is not None:
        for robot_id in range(n_robots):        
            # convert pose to mat
            pose_mat = env_obs_to_pose_mat(env_obs, f'robot{robot_id}')
            
            # get start pose
            start_pose = episode_start_pose[robot_id]
            start_pose_mat = pose_to_mat(start_pose)
            rel_obs_pose_mat = convert_pose_mat_rep_func(
                pose_mat,
                base_pose_mat=start_pose_mat,
                pose_rep='relative',
                backward=False)
            
            rel_obs_pose = mat_to_pose10d(rel_obs_pose_mat)
            # obs_dict_np[f'robot{robot_id}_eef_pos_wrt_start'] = rel_obs_pose[...,:3]
            obs_dict_np[f'robot{robot_id}_eef_rot_axis_angle_wrt_start'] = rel_obs_pose[...,3:]

    return obs_dict_np

def get_real_umi_action(
        action: np.ndarray,
        env_obs: Dict[str, np.ndarray], 
        action_pose_repr: str='abs'
    ):

    n_robots = int(action.shape[-1] // 10)
    env_action = list()
    for robot_idx in range(n_robots):
        # convert pose to mat
        pose_mat = pose_to_mat(np.concatenate([
            env_obs[f'robot{robot_idx}_eef_pos'][-1],
            env_obs[f'robot{robot_idx}_eef_rot_axis_angle'][-1]
        ], axis=-1))

        start = robot_idx * 10
        action_pose10d = action[..., start:start+9]
        action_grip = action[..., start+9:start+10]
        action_pose_mat = pose10d_to_mat(action_pose10d)

        # solve relative action
        action_mat = convert_pose_mat_rep(
            action_pose_mat, 
            base_pose_mat=pose_mat,
            pose_rep=action_pose_repr,
            backward=True)

        # convert action to pose
        action_pose = mat_to_pose(action_mat)
        env_action.append(action_pose)
        env_action.append(action_grip)

    env_action = np.concatenate(env_action, axis=-1)
    return env_action

# Added by WSHF, used for sync wait
def wait_until_still(env, velocity_threshold=0.01, timeout=5.0):
    t_start = time.time()
    while time.time() - t_start < timeout:
        state = env.get_robot_state()
        tcp_vel = np.linalg.norm(state['ActualTCPSpeed'])

        if tcp_vel < velocity_threshold:
            return True
        
        time.sleep(0.01)

    print(f"[WARNING] Wait for still timed out! Vel: {tcp_vel:.3f}")
    return False

def get_robot_cfg_name(robot_type: str) -> str:
    robot_type = str(robot_type).lower()
    robot_cfg_name_map = {
        'ur5': 'ur5_robotiq_umi.yml',
        'franka': 'panda_robotiq_umi.yml',
    }
    if robot_type not in robot_cfg_name_map:
        raise KeyError(f"Unsupported robot_type for joint-space policy adaptation: {robot_type}")
    return robot_cfg_name_map[robot_type]


def robotiq_width_to_joint_angles(
    gripper_width,
    max_width: float = 0.085,
    outer_knuckle_max: float = 0.81,
    inner_knuckle_max: float = 0.8757,
):
    """
    Convert parallel jaw opening width (meters) into the 6-D Robotiq joint state
    expected by `ur5_robotiq_umi.yml`:
    [left_outer_knuckle, left_inner_knuckle, left_inner_finger,
     right_outer_knuckle, right_inner_knuckle, right_inner_finger].

    The 2F-85 URDF uses one scalar closure state replicated across the finger
    joints, with `inner_finger` rotating in the opposite direction.
    """
    width = np.asarray(gripper_width, dtype=np.float32)
    width = np.clip(width, 0.0, max_width)
    close_ratio = 1.0 - (width / max_width)

    outer_knuckle = close_ratio * outer_knuckle_max
    inner_knuckle = close_ratio * inner_knuckle_max
    inner_finger = -inner_knuckle

    return np.stack([
        outer_knuckle,
        inner_knuckle,
        inner_finger,
        outer_knuckle,
        inner_knuckle,
        inner_finger,
    ], axis=-1).astype(np.float32)


def build_policy_current_joint_angles(
    arm_joint_angles,
    gripper_width,
    robot_cfg_name: str,
):
    arm_joint_angles = np.asarray(arm_joint_angles, dtype=np.float32)
    if robot_cfg_name in ['ur5_robotiq_umi.yml', 'panda_robotiq_umi.yml']:
        gripper_joint_angles = robotiq_width_to_joint_angles(gripper_width)
        return np.concatenate([arm_joint_angles, gripper_joint_angles], axis=-1)
    return arm_joint_angles


def get_current_action_base_pose(obs, robot_id: int = 0) -> np.ndarray:
    return np.concatenate([
        obs[f'robot{robot_id}_eef_pos'][-1],
        obs[f'robot{robot_id}_eef_rot_axis_angle'][-1],
    ], axis=-1).astype(np.float32)
