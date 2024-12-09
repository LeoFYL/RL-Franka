import torch
from omni.isaac.lab.assets import AssetBase
from omni.isaac.lab.managers import SceneEntityCfg


class DynamicObstacleManager:
    def __init__(self, obstacle_cfg: SceneEntityCfg, env_device: str):
        """初始化动态障碍管理器"""
        self.obstacle_cfg = obstacle_cfg
        self.env_device = env_device
        self.obstacle = None
        self.time_step = 0

    def initialize(self, scene):
        """初始化障碍物"""
        self.obstacle = scene[self.obstacle_cfg.name]

    def update_position(self, total_steps, speed=0.1, max_distance=0.5):
        """更新障碍物位置"""
        if self.obstacle is None:
            raise ValueError("Obstacle not initialized. Call initialize() first.")
        
        # 使用正弦波移动障碍物
        new_x = max_distance * torch.sin(2 * torch.pi * speed * self.time_step / total_steps)
        new_y = max_distance * torch.cos(2 * torch.pi * speed * self.time_step / total_steps)
        new_position = torch.tensor([new_x, new_y, 0.25], device=self.env_device)
        
        # 更新障碍物位置
        self.obstacle.set_world_poses(position=new_position)
        self.time_step += 1
