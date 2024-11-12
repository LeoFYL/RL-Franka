# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Common functions that can be used to activate certain terminations for the lift and place task.

The functions can be passed to the :class:`omni.isaac.lab.managers.TerminationTermCfg` object to enable
termination introduced by the function.
"""

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


def object_reached_goal(
    env: ManagerBasedRLEnv,
    command_name: str = "object_pose",
    threshold: float = 0.02,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Termination condition for the object reaching the goal position.

    Args:
        env: The environment.
        command_name: The name of the command that is used to control the object.
        threshold: The threshold for the object to reach the goal position. Defaults to 0.02.
        robot_cfg: The robot configuration. Defaults to SceneEntityCfg("robot").
        object_cfg: The object configuration. Defaults to SceneEntityCfg("object").
    """
    # extract the used quantities (to enable type-hinting)
    robot: Articulation = env.scene[robot_cfg.name]
    obj: RigidObject = env.scene[object_cfg.name]
    command = env.command_manager.get_command(command_name)

    # compute the desired position in the world frame
    des_pos_b = command[:, :3]
    des_pos_w, _ = combine_frame_transforms(robot.data.root_state_w[:, :3], robot.data.root_state_w[:, 3:7], des_pos_b)

    # distance of the end-effector to the object: (num_envs,)
    distance = torch.norm(des_pos_w - obj.data.root_pos_w[:, :3], dim=1)

    # rewarded if the object is within the threshold distance to the goal
    return distance < threshold


def action_limitations(
    env: ManagerBasedRLEnv,
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Terminate the episode if the end-effector exceeds certain positional limits.

    Args:
        env: The environment.
        ee_frame_cfg: Configuration for the end-effector frame.
        robot_cfg: The robot configuration.
    """
    robot: Articulation = env.scene[robot_cfg.name]
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    ee_tool_pos = ee_frame.data.target_pos_w[..., 0, :]

    x_offset = ee_tool_pos[:, 0] - robot.data.root_state_w[:, 0]
    y_offset = ee_tool_pos[:, 1] - robot.data.root_state_w[:, 1]

    # Terminate if EE z-position is outside range [0.2, 2.0] or y offset is beyond limits
    return (ee_tool_pos[:, 2] < 0.2) | (ee_tool_pos[:, 2] > 2.0) | (y_offset > 0.5) | (y_offset < -0.2) | (x_offset < 0.1)


def trajectory_completed_or_deviated(
    env: ManagerBasedRLEnv, center: torch.Tensor, radius: float, threshold: float
) -> torch.Tensor:
    """Terminate if the robot completes the trajectory or deviates too much from it."""
    asset: Articulation = env.scene["robot"]
    curr_pos_w = asset.data.root_pos_w[:, :2]  # Only consider XY coordinates
    time_step = env.current_time_step

    # Calculate desired position on the circular path
    theta = (time_step / env.total_time_steps) * 2 * torch.pi
    des_pos_w = center + radius * torch.tensor([torch.cos(theta), torch.sin(theta)], device=curr_pos_w.device)

    distance = torch.norm(curr_pos_w - des_pos_w, dim=1)
    completed = time_step >= env.total_time_steps
    deviated = distance > threshold

    return completed | deviated


def object_grasped(env, gripper_cfg, object_cfg, threshold=0.05):
    # Get the gripper and object from the scene
    gripper = env.scene[gripper_cfg.name]
    object_ = env.scene[object_cfg.name]

    # Retrieve the world position of the gripper and object
    gripper_pos, _ = gripper.get_world_poses()
    object_pos, _ = object_.get_world_poses()

    # Calculate the distance between the gripper and the object
    distance = torch.norm(gripper_pos - object_pos, dim=-1)

    # Check if the object is within the grasping threshold
    return distance < threshold



def object_placed(env, object_cfg, place_position, precision=0.05):
    
    obj_pos, _ = env.scene[object_cfg.name].get_world_poses()
    
    
    place_position_tensor = torch.tensor(place_position, device=obj_pos.device)

    distance_to_place = torch.norm(obj_pos[:, :2] - place_position_tensor[:2], dim=1)
    
    return distance_to_place <= precision
