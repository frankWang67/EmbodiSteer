import torch

from diffusion_policy.policy.diffusion_unet_timm_policy import DiffusionUnetTimmPolicy
from embodisteer.guidance.diffusion import (
    get_pred_x0,
    rel_action_obstacle_loss,
    get_guidance_strength,
)

class DiffusionUnetTimmPolicyWithGuidance(DiffusionUnetTimmPolicy):
    eef_corner_pts = torch.tensor([
        [ 0.01,  0.043, 0.01],
        [ 0.01, -0.043, 0.01],
        [-0.01,  0.043, 0.01],
        [-0.01, -0.043, 0.01],

        [ 0.01,  0.043, -0.03],
        [ 0.01, -0.043, -0.03],
        [-0.01,  0.043, -0.03],
        [-0.01, -0.043, -0.03],

        # [ 0.04,  0.0,  -0.15],
        # [-0.04,  0.0,  -0.15],
        # [ 0.0 ,  0.04, -0.15],
        # [ 0.0 , -0.04, -0.15],
        
        # [0.0, 0.0, 0.0],
    ])

    def to(self, device):
        super().to(device)
        self.eef_corner_pts = self.eef_corner_pts.to(device)

    def conditional_sample(self, 
        condition_data,
        condition_mask,
        local_cond=None,
        global_cond=None,
        generator=None,
        chunk_start_pose=None,
        obstacle_info=[],
        current_joint_angles=None,
        # keyword arguments to scheduler.step
        **kwargs
    ):
        assert chunk_start_pose is not None, "chunk_start_pose is required for guided diffusion"

        model = self.model
        scheduler = self.noise_scheduler

        trajectory = torch.randn(
            size=condition_data.shape, 
            dtype=condition_data.dtype,
            device=condition_data.device,
            generator=generator)
    
        # set step values
        scheduler.set_timesteps(self.num_inference_steps)

        for t in scheduler.timesteps:
            # 1. apply conditioning
            trajectory[condition_mask] = condition_data[condition_mask]

            # 2. predict model output
            model_output = model(trajectory, t, 
                local_cond=local_cond, global_cond=global_cond)
            
            # ===============================================================
            # Guided Diffusion
            # ===============================================================
            # 1. 准备数据
            alpha_prod_t = self.noise_scheduler.alphas_cumprod[t]
            # beta_t = self.noise_scheduler.betas[t]

            # 开启梯度计算
            with torch.enable_grad():
                # 2. 预测 x0 (clean action prediction)
                # 这里的 pred_action_delta 应该是反归一化后的物理值
                x_in = trajectory.detach().clone().requires_grad_(True)
                pred_action_normalized = get_pred_x0(model_output, x_in, alpha_prod_t)
                
                # !重要!: 反归一化 (Un-Normalize)
                # Loss 计算必须在物理空间 (米/弧度)
                x_physical = self.normalizer['action'].unnormalize(pred_action_normalized)
                x_physical = x_physical[:, :, :9] # 只取前 9 维 (3 维位置 + 6 维姿态) 来计算损失
                
                # 计算 Loss
                loss = rel_action_obstacle_loss(
                    action_pred=x_physical, 
                    current_state=chunk_start_pose, # 传入当前的真实机器人状态
                    robot_corners=self.eef_corner_pts,
                    obstacles=obstacle_info,
                )
                
                # 计算梯度
                grad = torch.autograd.grad(loss, x_in, allow_unused=True)[0]
                if grad is None:
                    grad = torch.zeros_like(x_in)
                grad = torch.clamp(grad, -0.1, 0.1) # 梯度裁剪，防止过大扰动
                
            # 3. Apply Guidance
            # 修改 noisy_action 或 epsilon
            # 注意: 如果是在 Delta 空间做引导，梯度往往会改变整条轨迹的形态
            gamma = get_guidance_strength(t, self.num_inference_steps)
            # model_output += torch.sqrt(1 - alpha_prod_t) * gamma * grad
            # ===============================================================

            # 3. compute previous image: x_t -> x_t-1
            trajectory = scheduler.step(
                model_output, t, trajectory, 
                generator=generator,
                **kwargs
                ).prev_sample
            # Apply guidance (在更新后的 trajectory 上应用梯度引导)
            # trajectory = trajectory - beta_t * gamma * grad
            trajectory = trajectory - gamma * grad
        
        # finally make sure conditioning is enforced
        trajectory[condition_mask] = condition_data[condition_mask]

        return trajectory
