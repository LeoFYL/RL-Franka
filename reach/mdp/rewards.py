# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from omni.isaac.lab.assets import RigidObject
from omni.isaac.lab.managers import SceneEntityCfg
from omni.isaac.lab.utils.math import combine_frame_transforms
from omni.isaac.lab.sensors import FrameTransformer
from omni.isaac.lab.assets import Articulation, RigidObject

if TYPE_CHECKING:
    from omni.isaac.lab.envs import ManagerBasedRLEnv


def object_is_lifted(env: ManagerBasedRLEnv, minimal_height: float, object_cfg: SceneEntityCfg) -> torch.Tensor:
    # Get the object from the scene and its world position
    obj = env.scene[object_cfg.name]
    object_pos_w, _ = obj.get_world_poses()

    # Check if the object's height is above the minimal threshold
    return object_pos_w[:, 2] > minimal_height


def object_reached_goal(
    env: ManagerBasedRLEnv,
    command_name: str = "object_pose",
    threshold: float = 0.02,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    # extract the used quantities (to enable type-hinting)
    robot: RigidObject = env.scene[robot_cfg.name]
    object: RigidObject = env.scene[object_cfg.name]
    command = env.command_manager.get_command(command_name)
    # compute the desired position in the world frame
    des_pos_b = command[:, :3]
    
    des_pos_w, _ = combine_frame_transforms(robot.data.root_state_w[:, :3], robot.data.root_state_w[:, 3:7], des_pos_b)
    # distance of the end-effector to the object: (num_envs,)
    distance = torch.norm(des_pos_w - object.data.root_pos_w[:, :3], dim=1)

    # rewarded if the object is lifted above the threshold
    return distance < threshold


def action_limitations(
    env: ManagerBasedRLEnv,
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    robot: RigidObject = env.scene[robot_cfg.name]
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    ee_tool = ee_frame.data.target_pos_w[..., 0, :]
    x = ee_tool[:,0] - robot.data.root_state_w[:, 0]
    y = ee_tool[:,1] - robot.data.root_state_w[:, 1]

    return (ee_tool[:,2] < 0.2) | (ee_tool[:,2] > 2.0) | (y > 0.5) | (y < -0.2) | (x < 0.1)


def trajectory_completed_or_deviated(
    env: ManagerBasedRLEnv, center: torch.Tensor, radius: float, threshold: float
) -> torch.Tensor:
    asset: RigidObject = env.scene["robot"]
    curr_pos_w = asset.data.root_pos_w[:, :2]  # Only consider XY coordinates
    time_step = env.current_time_step

    # Calculate desired position on the circular path
    theta = (time_step / env.total_time_steps) * 2 * math.pi
    des_pos_w = center + radius * torch.tensor([torch.cos(theta), torch.sin(theta)], device=curr_pos_w.device)

    distance = torch.norm(curr_pos_w - des_pos_w, dim=1)
    completed = time_step >= env.total_time_steps
    deviated = distance > threshold

    return completed | deviated

def position_command_error(env, command_name: str, asset_cfg):
    # position command error
    asset = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    des_pos = command[:, :3]
    curr_pos = asset.data.root_state[:, :3]
    return torch.norm(curr_pos - des_pos, dim=1)

def object_grasped(
    env: ManagerBasedRLEnv,
    gripper_cfg: SceneEntityCfg = SceneEntityCfg("gripper"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    threshold: float = 0.01
) -> torch.Tensor:
    gripper_frame: FrameTransformer = env.scene[gripper_cfg.name]
    object: RigidObject = env.scene[object_cfg.name]

    gripper_pos = gripper_frame.data.root_pos_w[..., 0, :]
    object_pos, _ = object.get_world_poses()

    dist = torch.norm(gripper_pos - object_pos, dim=1)

    return dist < threshold  # Terminate if the gripper is very close to the object


def object_placed(
    env: ManagerBasedRLEnv,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    target_position: torch.Tensor = torch.tensor([0.55, -0.2, 0.35]),
    threshold: float = 0.02
) -> torch.Tensor:
    """Check if the object is placed at the target position."""
    obj = env.scene[object_cfg.name]
    object_pos_w, _ = obj.get_world_poses()
    distance = torch.norm(object_pos_w[:, :2] - target_position[:2], dim=1)  # XY distance
    return distance < threshold  # Check if within threshold distance
