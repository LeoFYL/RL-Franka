# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from omni.isaac.lab.assets import RigidObject
from omni.isaac.lab.managers import SceneEntityCfg
from omni.isaac.lab.utils.math import subtract_frame_transforms

if TYPE_CHECKING:
    from omni.isaac.lab.envs import ManagerBasedEnv, ManagerBasedRLEnv


def object_position_in_robot_root_frame(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """The position of the object in the robot's root frame."""
    robot: RigidObject = env.scene[robot_cfg.name]
    obj = env.scene[object_cfg.name]
    object_pos_w, _ = obj.get_world_poses()

    # Get object position relative to robot frame
    object_pos_b, _ = subtract_frame_transforms(
        robot.data.root_state_w[:, :3], robot.data.root_state_w[:, 3:7], object_pos_w
    )

    # Adaptive range checking for graspable area
    in_range = (0.3 < object_pos_b[:, 0]) & (object_pos_b[:, 0] < 0.7) & (-0.2 < object_pos_b[:, 1]) & (object_pos_b[:, 1] < 0.2)

    # Always return a tensor, even if the object is out of range
    if in_range.all():
        return object_pos_b
    else:
        return torch.zeros_like(object_pos_b)  # Return a tensor of zeros if out of range


def last_action_clamped(env: ManagerBasedEnv, action_name: str | None = None) -> torch.Tensor:
    """The last input action to the environment.

    The name of the action term for which the action is required. If None, the
    entire action tensor is returned.
    """
    if action_name is None:
        return env.action_manager.action
    else:
        return env.action_manager.get_term(action_name).raw_actions.clamp(-1, 1)


# Grasp retry logic in the main loop
def grasp_retry_logic(env: ManagerBasedRLEnv):
    for attempt in range(3):
        if verify_grasp(env):
            break
        else:
            # Retry grasp logic
            env.action_manager.get_term("gripper_action").apply_closing_action()


def verify_grasp(env: ManagerBasedRLEnv, object_cfg: SceneEntityCfg = SceneEntityCfg("object")) -> bool:
    obj = env.scene[object_cfg.name]
    object_pos_w, _ = obj.get_world_poses()
    
    gripper_pos = env.scene["robot"].data.root_state_w[:, :3]
    distance = torch.norm(gripper_pos - object_pos_w, dim=-1)
    
    # Consider grasp successful if distance is below a threshold
    return distance < 0.05

def gripper_object_alignment(env: ManagerBasedRLEnv, object_cfg: SceneEntityCfg, gripper_cfg: SceneEntityCfg) -> torch.Tensor:
    # Ensure alignment by checking both position and relative orientation of the gripper and object.
    object_pos = env.scene[object_cfg.name].data.root_pos_w
    gripper_pos = env.scene[gripper_cfg.name].data.root_pos_w

    dist = torch.norm(gripper_pos - object_pos, dim=1)
    return 1 - torch.tanh(dist / 0.05)
