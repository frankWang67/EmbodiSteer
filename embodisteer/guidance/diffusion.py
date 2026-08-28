"""Diffusion-time clean-sample and Cartesian collision guidance helpers."""

import torch
from embodisteer.kinematics.rotation import axis_angle_to_matrix

def quat_conjugate(quat):
    """
    计算四元数的共轭 (用于逆旋转)
    假设格式: [w, x, y, z] (ManiSkill/Sapien 默认是 w 在前)
    共轭: [w, -x, -y, -z]
    """
    # 如果是 batched input (..., 4)
    w, x, y, z = quat.unbind(dim=-1)
    return torch.stack([w, -x, -y, -z], dim=-1)

def quat_apply(quat, vec):
    """
    将四元数旋转应用到向量上 (Standard vectorization implementation)
    quat: (..., 4) [w, x, y, z] or [x, y, z, w], 需注意你的模型输出格式
    vec: (..., 3)
    """
    # 假设 quaternion 格式为 [w, x, y, z] (实部在前)
    # 如果你的模型输出是 [x, y, z, w]，请在此处调整切片
    w, x, y, z = quat.unbind(dim=-1)
    
    # 临时变量辅助计算
    two_s = 2.0 / (quat * quat).sum(dim=-1) # normalization factor if not normalized
    two_s = 2.0 # 假设输出已归一化，通常直接取2
    
    # 简化的旋转逻辑 (PyTorch 官方常用写法)
    uv = torch.cross(quat[..., 1:], vec, dim=-1)
    uuv = torch.cross(quat[..., 1:], uv, dim=-1)
    return vec + two_s * (w.unsqueeze(-1) * uv + uuv)

def rot6d_to_matrix(rot6d):
    """
    将 6D 表示转换为旋转矩阵
    rot6d: (..., 6) 前3维是 x_axis，后3维是 y_axis 的未归一化向量
    输出: (..., 3, 3) 旋转矩阵
    """
    x_raw = rot6d[..., :3]
    y_raw = rot6d[..., 3:]

    # 归一化 x_axis
    x_axis = torch.nn.functional.normalize(x_raw, dim=-1)
    
    # 从 y_raw 中去除 x_axis 的分量，得到正交的 y_axis
    y_proj = (y_raw * x_axis).sum(dim=-1, keepdim=True) * x_axis
    y_axis = torch.nn.functional.normalize(y_raw - y_proj, dim=-1)
    
    # z_axis 是 x 和 y 的叉积
    z_axis = torch.cross(x_axis, y_axis, dim=-1)
    
    # 堆叠成旋转矩阵
    rot_mat = torch.stack([x_axis, y_axis, z_axis], dim=-2) # (..., 3, 3)
    
    return rot_mat

def matrix_to_rot6d(rot_mat):
    """
    将旋转矩阵转换回 6D 表示
    rot_mat: (..., 3, 3)
    输出: (..., 6) 前3维是 x_axis，后3维是 y_axis 的未归一化向量
    """
    x_axis = rot_mat[..., 0, :] # (..., 3)
    y_axis = rot_mat[..., 1, :] # (..., 3)
    
    rot6d = torch.cat([x_axis, y_axis], dim=-1) # (..., 6)
    return rot6d

def pose9d_to_mat(pose9d):
    """
    将 9D pose (pos + 6d rot) 转换为 4x4 变换矩阵
    """
    pos = pose9d[..., :3] # (B, 9) -> (B, 3)
    rot6d = pose9d[..., 3:] # (B, 9) -> (B, 6)
    
    # 将 6D 表示转换为旋转矩阵
    rot_mat = rot6d_to_matrix(rot6d) # (B, 3, 3)
    
    # 构建 4x4 变换矩阵
    batch_size = pose9d.shape[0]
    mat = torch.eye(4).unsqueeze(0).repeat(batch_size, 1, 1).to(pose9d.device) # (B, 4, 4)
    mat[:, :3, :3] = rot_mat
    mat[:, :3, 3] = pos
    
    return mat

def mat_to_pose9d(mat):
    """
    将 4x4 变换矩阵转换回 9D pose (pos + 6d rot)
    """
    pos = mat[:, :3, 3] # (B, 4, 4) -> (B, 3)
    rot_mat = mat[:, :3, :3] # (B, 4, 4) -> (B, 3, 3)
    
    rot6d = matrix_to_rot6d(rot_mat) # (B, 3, 3) -> (B, 6)
    
    pose9d = torch.cat([pos, rot6d], dim=-1) # (B, 9)
    return pose9d

def point_obb_distance(points, box_center, box_quat, box_extent):
    """
    计算点到旋转长方体 (OBB) 的距离。
    
    Args:
        points: (B, T, N_k, 3) 机器人关键点 (世界坐标)
        box_center: (B, 3) 障碍物中心
        box_quat: (B, 4) 障碍物旋转四元数
        box_extent: (B, 3) 障碍物半长轴 (half_size)
        
    Returns:
        dist: (B, T, N_k) 距离
    """
    # 1. 广播维度处理 (Broadcasting)
    # 假设 points 是 (B, T, N_k, 3)
    # 障碍物 pose 通常是 (B, 3)，需要扩充到 (B, 1, 1, 3) 以便对齐 T 和 N_k
    if box_center.ndim == 2: # (B, 3)
        center_expanded = box_center.unsqueeze(1).unsqueeze(1) # (B, 1, 1, 3)
        quat_expanded = box_quat.unsqueeze(1).unsqueeze(1)     # (B, 1, 1, 4)
        box_extent = box_extent.unsqueeze(1).unsqueeze(1) # (B, 1, 1, 3)
    else: # static single env
        center_expanded = box_center
        quat_expanded = box_quat
        box_extent = box_extent

    # 2. 平移变换 (Translate)
    # 得到相对于 Box 中心的向量 (世界坐标系方向)
    p_centered = points - center_expanded
    
    # 3. 旋转变换 (Rotate into Local Frame)
    # 使用 Box 的逆旋转 (共轭四元数)
    quat_inv = quat_conjugate(quat_expanded)
    p_local = quat_apply(quat_inv, p_centered)
    
    # 4. 在局部坐标系下计算 AABB 距离 (即之前的逻辑)
    # 此时 Box 在局部坐标系下是以原点为中心，轴对齐的
    q = torch.abs(p_local) - box_extent.to(points.device)
    
    # 计算外部距离
    dist_outside = torch.norm(torch.clamp(q, min=0.0), dim=-1)

    # 内部距离部分 (标量, 取最大的那个负分量，即离最近表面的距离)
    # max(q, dim=-1)[0] 找到了离得最近的那个面的距离（在内部都是负数）
    # min(..., 0.0) 确保只在内部生效
    dist_inside = torch.min(torch.max(q, dim=-1)[0], torch.tensor(0.0).to(points.device))
    
    return dist_outside + dist_inside

def relative_trajectory_to_absolute(current_state, action_pred):
    """
    可微地将相对动作序列转化为世界坐标系轨迹。
    
    Args:
        current_state: (B, 6) [x, y, z, ax, ay, az] (当前机器人状态)
        action_pred:   (B, T, 9) [dx, dy, dz, r1, r2, r3, r4, r5, r6] (模型预测)
        
    Returns:
        abs_traj_pos: (B, T, 3) 世界坐标系下的位置序列
        abs_traj_rot: (B, T, 3, 3) 世界坐标系下的旋转矩阵序列
    """
    B, T, _ = action_pred.shape
    
    # 1. 初始化当前状态
    curr_pos = current_state[:, :3]  # (B, 3)
    curr_rot_axis = current_state[:, 3:6]
    curr_rot_mat = axis_angle_to_matrix(curr_rot_axis) # (B, 3, 3)
    curr_pose_mat = torch.eye(4).unsqueeze(0).repeat(B, 1, 1).to(action_pred.device) # (B, 4, 4)
    curr_pose_mat[:, :3, :3] = curr_rot_mat
    curr_pose_mat[:, :3, 3] = curr_pos
    
    # 2. 将动作转化为变换矩阵
    traj_mat = pose9d_to_mat(action_pred.view(B*T, 9)).view(B, T, 4, 4) # (B, T, 4, 4)

    # 3. 应用变换
    abs_traj_mat = curr_pose_mat.unsqueeze(1) @ traj_mat # (B, T, 4, 4)

    # 4. 提取位置和旋转矩阵
    abs_traj_pos = abs_traj_mat[:, :, :3, 3] # (B, T, 3)
    abs_traj_rot = abs_traj_mat[:, :, :3, :3] # (B, T, 3, 3)

    return abs_traj_pos, abs_traj_rot

def get_pred_x0(noise_pred, noisy_action_seq, alpha_prod_k):
    return (noisy_action_seq - torch.sqrt(1 - alpha_prod_k) * noise_pred) / torch.sqrt(alpha_prod_k)

def flatten_obstacle_info(obstacles):
    if isinstance(obstacles, list) and (
        len(obstacles) == 0 or isinstance(obstacles[0], dict)
    ):
        return obstacles

    res = []
    n_envs = len(obstacles)
    n_obstacles = len(obstacles[0])
    for i in range(n_obstacles):
        res.append({
            'center': [],
            'quat': [],
            'extent': [],
        })
        for j in range(n_envs):
            res[i]['center'].append(obstacles[j][i]['center'])
            res[i]['quat'].append(obstacles[j][i]['quat'])
            res[i]['extent'].append(obstacles[j][i]['extent'])
        res[i]['center'] = torch.cat(res[i]['center'], dim=0)
        res[i]['quat'] = torch.cat(res[i]['quat'], dim=0)
        res[i]['extent'] = torch.cat(res[i]['extent'], dim=0)
    return res

def rel_action_obstacle_loss(
    action_pred,        # (B, T, 6) 需要求导的变量
    current_state,      # (B, 6) Constant
    robot_corners,      # (N, 3) 夹爪局部角点
    obstacles,          # 障碍物信息
    safety_margin=0.04
):
    # 1. 将 Relative Pose 变为 Absolute Pose
    # 梯度会穿过这里，回传给 action_pred
    traj_pos, traj_rot = relative_trajectory_to_absolute(current_state, action_pred)
    
    # 2. 【运动学变换】计算角点在世界坐标系的位置
    # traj_pos: (B, T, 3)
    # traj_rot: (B, T, 3, 3)
    # corners:  (N, 3)
    
    # Expand dims for broadcasting
    # (B, T, 1, 3, 3)
    rot_expanded = traj_rot.unsqueeze(2) 
    # (1, 1, N, 3, 1) -> 转置为列向量做矩阵乘法
    corners_expanded = robot_corners.view(1, 1, robot_corners.shape[0], 3, 1)
    
    # 旋转: R * p_local
    # result: (B, T, N, 3, 1) -> squeeze -> (B, T, N, 3)
    corners_rotated = torch.matmul(rot_expanded, corners_expanded).squeeze(-1)
    
    # 平移: + pos
    # (B, T, 1, 3) + (B, T, N, 3)
    corners_world = corners_rotated + traj_pos.unsqueeze(2)
    
    # 3. 【SDF Loss】计算距离场 Loss
    total_loss = torch.zeros(1, device=action_pred.device)
    if len(obstacles) == 0:
        total_loss.requires_grad_(True)
        return total_loss
    obstacles = flatten_obstacle_info(obstacles)
    for obs in obstacles:
        # 调用之前定义的点到Box距离函数
        dists = point_obb_distance(
            corners_world, 
            obs['center'].to(action_pred.device),
            obs['quat'].to(action_pred.device), 
            obs['extent'].to(action_pred.device),
        )
        
        penetration = safety_margin - dists
        # cost = torch.relu(penetration).pow(2)
        cost = torch.relu(penetration)
        total_loss += cost.sum()
        
    return total_loss

def get_guidance_strength(k, num_diffusion_steps):
    h1 = 1.0
    h2 = 50.0
    h3 = 0.7
    
    t = k / num_diffusion_steps
    gamma = h1 / (1 + torch.exp(-h2 * (h3 - t)))

    return gamma


__all__ = [
    "flatten_obstacle_info",
    "get_guidance_strength",
    "get_pred_x0",
    "rel_action_obstacle_loss",
]
