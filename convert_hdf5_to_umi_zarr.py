#!/usr/bin/env python3
"""
Convert HDF5 dataset to UMI zarr format.

Usage:
python convert_hdf5_to_umi_zarr.py \
    --input-h5-path xx.h5 \
    --output-path umi_dataset.zarr.zip \
    --camera-name hand_camera \
    --image-size 224
"""

import os
import cv2
import h5py
import zarr
import argparse
import numpy as np
from tqdm import tqdm
from scipy.spatial.transform import Rotation as R
from diffusion_policy.common.replay_buffer import ReplayBuffer
from diffusion_policy.codecs.imagecodecs_numcodecs import register_codecs, Jpeg2k

register_codecs()

def quaternion_to_axis_angle(quat):
    """Convert quaternion (w,x,y,z) to axis-angle representation."""
    # Ensure quaternion is normalized
    quat = quat / np.linalg.norm(quat, axis=-1, keepdims=True)
    
    # Convert to rotation object using scipy (expects x,y,z,w format)
    rotation = R.from_quat(quat[..., [1, 2, 3, 0]])  # Convert from w,x,y,z to x,y,z,w
    
    # Convert to axis-angle (rotvec)
    axis_angle = rotation.as_rotvec()
    return axis_angle

def euler_to_axis_angle(euler_angles):
    """Convert euler angles (roll, pitch, yaw) to axis-angle representation."""
    rotation = R.from_euler("xyz", euler_angles, degrees=False)
    axis_angle = rotation.as_rotvec()
    return axis_angle

def resize_image(image, target_size=224):
    """Resize image to target size keeping HWC format."""
    # image is (H, W, C), keep it as HWC for UMI format
    resized = cv2.resize(image, (target_size, target_size))
    return resized

def convert_dataset(input_h5_path, output_path, camera_name='hand_camera', image_size=224):
    """Convert HDF5 dataset to UMI zarr format."""
    assert os.path.exists(input_h5_path), f"Input path {input_h5_path} does not exist."
    assert input_h5_path.endswith('.h5') or input_h5_path.endswith('.hdf5'), "Input path must be an HDF5 file."
    
    # Create replay buffer with memory store
    replay_buffer = ReplayBuffer.create_empty_zarr(storage=zarr.MemoryStore())
    
    print("Loading episodes...")

    with h5py.File(input_h5_path, 'r') as f:
        for i in tqdm(range(len(f)), desc="Converting episodes"):
            traj_key = f"/traj_{i}"
            tcp_key = f"{traj_key}/obs/extra/tcp_pose"
            qpos_key = f"{traj_key}/obs/agent/qpos"
            img_key = f"{traj_key}/obs/sensor_data/{camera_name}/rgb"
            action_key = f"{traj_key}/actions"

            tcp_pose = f[tcp_key][()]  # (T, 7)
            gripper = f[qpos_key][()][:, -1:]  # (T, 1)
            images = f[img_key][()]  # (T, H, W, C)
            
            # Load actions
            action = f[action_key][()]  # (T, 8): similar structure to qpos
            
            # Extract components from qpos (observations)
            ee_pos = tcp_pose[:, :3]  # (T, 3)
            ee_quat = tcp_pose[:, 3:7]  # (T, 4) - quaternion in w,x,y,z format
            
            # Extract components from action
            action_ee_pos = action[:, :3]  # (T, 3)
            # ===== CHANGE FROM EULER TO QUATERNION =====
            # action_ee_euler = action[:, 3:6]  # (T, 3) - euler angles
            action_ee_quat = action[:, 3:7]  # (T, 4) - quaternion in w,x,y,z format
            # ===========================================
            action_gripper = action[:, 7:8]  # (T, 1)
            
            # Convert quaternions to axis-angle
            ee_rot_axis_angle = quaternion_to_axis_angle(ee_quat)  # (T, 3)
            # action_ee_rot_axis_angle = euler_to_axis_angle(action_ee_euler)  # (T, 3)
            action_ee_rot_axis_angle = quaternion_to_axis_angle(action_ee_quat)  # (T, 3)

            episode_data = {
                'ee_pos': ee_pos.astype(np.float32),
                'ee_rot_axis_angle': ee_rot_axis_angle.astype(np.float32),
                'gripper': gripper.astype(np.float32),
                'action_ee_pos': action_ee_pos.astype(np.float32),
                'action_ee_rot_axis_angle': action_ee_rot_axis_angle.astype(np.float32),
                'action_gripper': action_gripper.astype(np.float32),
                'images': images
            }

            episode_length = len(episode_data['ee_pos'])
            
            # Process images
            episode_images = []
            for img in episode_data['images']:
                resized_img = resize_image(img, image_size)
                episode_images.append(resized_img)
            episode_images = np.stack(episode_images)
            
            # Ensure images are uint8 (0-255 range)
            if episode_images.dtype != np.uint8:
                episode_images = (episode_images * 255).astype(np.uint8)
            
            # Process actions - concatenate [pos(3), axis_angle(3), gripper(1)] = (7,)
            episode_actions = np.concatenate([
                episode_data['action_ee_pos'],
                episode_data['action_ee_rot_axis_angle'],
                episode_data['action_gripper']
            ], axis=1)
            # copy the last action to match episode length
            episode_actions = np.vstack([episode_actions, episode_actions[-1:]])[:episode_length]
            
            # Create demo start and end poses (6D: pos + axis_angle)
            demo_start_pose = np.concatenate([
                episode_data['ee_pos'][0:1],  # First timestep position (1, 3)
                episode_data['ee_rot_axis_angle'][0:1]  # First timestep rotation (1, 3)
            ], axis=1)  # Shape: (1, 6)
            
            demo_end_pose = np.concatenate([
                episode_data['ee_pos'][-1:],  # Last timestep position (1, 3)  
                episode_data['ee_rot_axis_angle'][-1:]  # Last timestep rotation (1, 3)
            ], axis=1)  # Shape: (1, 6)
            
            # Expand to match episode length for storage
            demo_start_poses = np.tile(demo_start_pose, (episode_length, 1))  # (T, 6)
            demo_end_poses = np.tile(demo_end_pose, (episode_length, 1))  # (T, 6)
            
            # Prepare episode data for adding to replay buffer
            episode_dict = {
                'robot0_eef_pos': episode_data['ee_pos'],
                'robot0_eef_rot_axis_angle': episode_data['ee_rot_axis_angle'],
                'robot0_gripper_width': episode_data['gripper'],
                'robot0_demo_start_pose': demo_start_poses,
                'robot0_demo_end_pose': demo_end_poses,
                'camera0_rgb': episode_images,
                'action': episode_actions
            }
            
            # Define compression for each data type
            compressors = {
                'robot0_eef_pos': None,
                'robot0_eef_rot_axis_angle': None,
                'robot0_gripper_width': None,
                'robot0_demo_start_pose': None,
                'robot0_demo_end_pose': None,
                'action': None,
                'camera0_rgb': Jpeg2k(level=50)
            }
            
            # Define chunks for each data type
            chunks = {
                'robot0_eef_pos': episode_data['ee_pos'].shape,
                'robot0_eef_rot_axis_angle': episode_data['ee_rot_axis_angle'].shape,
                'robot0_gripper_width': episode_data['gripper'].shape,
                'robot0_demo_start_pose': demo_start_poses.shape,
                'robot0_demo_end_pose': demo_end_poses.shape,
                'action': episode_actions.shape,
                'camera0_rgb': (1,) + episode_images.shape[1:]  # Chunk per timestep for images
            }
            
            # Add episode to replay buffer
            replay_buffer.add_episode(
                data=episode_dict,
                chunks=chunks,
                compressors=compressors
            )
    
    print(f"Total steps: {replay_buffer.n_steps}")
    print(f"Total episodes: {replay_buffer.n_episodes}")
    print(f"Data shapes:")
    for key in replay_buffer.keys():
        print(f"  {key}: {replay_buffer[key].shape}")
    
    # Verify action dimensionality
    action_shape = replay_buffer['action'].shape
    print(f"Action shape: {action_shape}")
    if action_shape[-1] != 7:
        print(f"WARNING: Expected action dimension 7, got {action_shape[-1]}")
    
    # Save to disk using ReplayBuffer's built-in method
    print(f"Saving dataset to: {output_path}")
    if output_path.endswith('.zip'):
        # Use ZipStore directly for .zip files
        with zarr.ZipStore(output_path, mode='w') as zip_store:
            replay_buffer.save_to_store(
                store=zip_store
            )
        print(f"Dataset converted and saved to: {output_path}")
    else:
        replay_buffer.save_to_path(output_path)
        print(f"Dataset converted and saved to: {output_path}")
    
    return output_path

def main():
    parser = argparse.ArgumentParser(description='Convert HDF5 dataset to UMI zarr format')
    parser.add_argument('--input-h5-path', '-i', required=True, 
                      help='Input HDF5 file path containing episode data')
    parser.add_argument('--output-path', '-o', required=True,
                      help='Output zarr file path (e.g., dataset.zarr.zip)')
    parser.add_argument('--camera-name', default='hand_camera',
                      help='Camera name in HDF5 files (default: hand_camera)')
    parser.add_argument('--image-size', type=int, default=224,
                      help='Target image size (default: 224)')
    
    args = parser.parse_args()
    
    convert_dataset(
        input_h5_path=args.input_h5_path,
        output_path=args.output_path,
        camera_name=args.camera_name,
        image_size=args.image_size
    )

if __name__ == '__main__':
    main() 