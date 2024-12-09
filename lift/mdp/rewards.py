from __future__ import annotations
import torch
import math
from typing import TYPE_CHECKING

from omni.isaac.lab.assets import RigidObject
from omni.isaac.lab.managers import SceneEntityCfg
from omni.isaac.lab.sensors import FrameTransformer
from omni.isaac.lab.utils.math import combine_frame_transforms

if TYPE_CHECKING:
    from omni.isaac.lab.envs import ManagerBasedRLEnv


def object_is_lifted(
    env: ManagerBasedRLEnv, minimal_height: float, object_cfg: SceneEntityCfg = SceneEntityCfg("object")
) -> torch.Tensor:
    """Reward the agent for lifting the object above the minimal height."""
    object: RigidObject = env.scene[object_cfg.name]
    return torch.where(object.data.root_pos_w[:, 2] > minimal_height, 1.0, 0.0)


def object_ee_distance(
    env: ManagerBasedRLEnv,
    std: float,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """Reward the agent for reaching the object using tanh-kernel."""
    object: RigidObject = env.scene[object_cfg.name]
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    cube_pos_w = object.data.root_pos_w
    ee_w = ee_frame.data.target_pos_w[..., 0, :]
    object_ee_distance = torch.norm(cube_pos_w - ee_w, dim=1)
    return 1 - torch.tanh(object_ee_distance / std)


def object_goal_distance(
    env: ManagerBasedRLEnv,
    std: float,
    minimal_height: float,
    command_name: str,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Reward the agent for tracking the goal pose using tanh-kernel."""
    robot: RigidObject = env.scene[robot_cfg.name]
    object: RigidObject = env.scene[object_cfg.name]
    command = env.command_manager.get_command(command_name)
    des_pos_b = command[:, :3]
    des_pos_w, _ = combine_frame_transforms(robot.data.root_state_w[:, :3], robot.data.root_state_w[:, 3:7], des_pos_b)
    distance = torch.norm(des_pos_w - object.data.root_pos_w[:, :3], dim=1)
    return (object.data.root_pos_w[:, 2] > minimal_height) * (1 - torch.tanh(distance / std))


def get_robot_link_positions(robot):
    """获取机器人链节的世界坐标."""
    if hasattr(robot.data, "link_state_w"):
        return robot.data.link_state_w[:, :3]
    elif hasattr(robot.data, "root_state_w"):
        return robot.data.root_state_w[:, :3]
    elif hasattr(robot, "get_world_poses"):
        link_positions, _ = robot.get_world_poses()
        return link_positions
    print("Warning: Robot data does not contain link state or root state. Returning empty tensor.")
    return torch.empty(0, 3)  # 返回形状为 (0, 3) 的空张量


def proximity_penalty(
    env: ManagerBasedRLEnv,
    obstacle_cfg: SceneEntityCfg,
    robot_cfg: SceneEntityCfg,
    minimal_distance: float = 0.25,
    penalty_weight: float = -0.5,
    dynamic: bool = False,
) -> torch.Tensor:
    """Calculates penalty for proximity to obstacles."""
    robot = env.scene[robot_cfg.name]
    obstacle = env.scene[obstacle_cfg.name]

    # 动态更新障碍物位置
    if dynamic:
        if not hasattr(env, "current_time_step") or not hasattr(env, "total_time_steps"):
            raise AttributeError("Environment lacks required attributes 'current_time_step' or 'total_time_steps'.")
        update_obstacle_position(obstacle, env.current_time_step, env.total_time_steps)

    # 获取机器人链节位置
    robot.write_data_to_sim()
    link_positions = get_robot_link_positions(robot)

    if link_positions.numel() == 0:
        print("Warning: link_positions tensor is empty. Returning zero penalty.")
        return torch.tensor(0.0, device=env.device)

    # 获取障碍物位置
    if not hasattr(obstacle, "get_world_poses"):
        print("Warning: Obstacle does not support get_world_poses.")
        return torch.tensor(0.0, device=env.device)

    obstacle_positions, _ = obstacle.get_world_poses()
    if obstacle_positions.numel() == 0:
        print("Warning: obstacle_positions tensor is empty. Returning zero penalty.")
        return torch.tensor(0.0, device=env.device)

    obstacle_position = obstacle_positions[0]
    link_positions = link_positions.to(obstacle_position.device)

    # 计算链节到障碍物的距离
    distances = torch.norm(link_positions - obstacle_position, dim=1)
    penalties = penalty_weight * torch.exp(-distances / minimal_distance)  # 平滑距离惩罚

    return penalties.sum()


def update_obstacle_position(obstacle, time_step, total_time):
    """Update obstacle position dynamically."""
    max_distance = 0.5  # 最大移动距离
    speed = 0.1  # 移动速度
    new_x_position = max_distance * torch.sin(2 * math.pi * speed * time_step / total_time)

    if not hasattr(obstacle, "set_world_poses"):
        print("Warning: Obstacle does not support dynamic position updates.")
        return

    obstacle.set_world_poses(position=torch.tensor([new_x_position, 0.0, 0.25], device=obstacle.device))
    obstacle.write_data_to_sim()  # 确保更新同步到仿真







