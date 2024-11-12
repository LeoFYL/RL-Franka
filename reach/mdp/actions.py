# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from collections.abc import Sequence
import omni.isaac.lab.utils.math as math_utils
#from omni.isaac.core.utils.kinematics import compute_pseudo_inverse
import carb

import omni.isaac.lab.utils.string as string_utils
from omni.isaac.lab.assets.articulation import Articulation
from omni.isaac.lab.managers.action_manager import ActionTerm
from omni.isaac.lab.managers import SceneEntityCfg

from omni.isaac.lab.envs import ManagerBasedEnv

from . import actions_cfg

from omni.isaac.core.articulations import ArticulationView

def compute_pseudo_inverse(jacobian, damping=0.01):
        """Compute the pseudo-inverse of a given matrix using the damped least squares method."""
        jjt = torch.mm(jacobian, jacobian.T)
        damping_matrix = damping * torch.eye(jjt.shape[0])
        pseudo_inverse = torch.mm(jacobian.T, torch.inverse(jjt + damping_matrix))
        return pseudo_inverse

class JointAction(ActionTerm):
    cfg: actions_cfg.JointActionCfg
    _asset: Articulation
    _scale: torch.Tensor | float
    _offset: torch.Tensor | float

    def __init__(self, cfg: actions_cfg.JointActionCfg, env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> None:
        super().__init__(cfg, env)
        self._asset: Articulation = env.scene[asset_cfg.name]
        self._joint_ids, self._joint_names = self._asset.find_joints(self.cfg.joint_names)
        self._num_joints = len(self._joint_ids)

        carb.log_info(
            f"Resolved joint names for the action term {self.__class__.__name__}:"
            f" {self._joint_names} [{self._joint_ids}]"
        )

        if self._num_joints == self._asset.num_joints:
            self._joint_ids = slice(None)

        self._raw_actions = torch.zeros(self.num_envs, self.action_dim, device=self.device)
        self._processed_actions = torch.zeros_like(self.raw_actions)

        if isinstance(cfg.scale, (float, int)):
            self._scale = float(cfg.scale)
        elif isinstance(cfg.scale, dict):
            self._scale = torch.ones(self.num_envs, self.action_dim, device=self.device)
            index_list, _, value_list = string_utils.resolve_matching_names_values(self.cfg.scale, self._joint_names)
            self._scale[:, index_list] = torch.tensor(value_list, device=self.device)
        else:
            raise ValueError(f"Unsupported scale type: {type(cfg.scale)}. Supported types are float and dict.")

        if isinstance(cfg.offset, (float, int)):
            self._offset = float(cfg.offset)
        elif isinstance(cfg.offset, dict):
            self._offset = torch.zeros_like(self._raw_actions)
            index_list, _, value_list = string_utils.resolve_matching_names_values(self.cfg.offset, self._joint_names)
            self._offset[:, index_list] = torch.tensor(value_list, device=self.device)
        else:
            raise ValueError(f"Unsupported offset type: {type(cfg.offset)}. Supported types are float and dict.")

    @property
    def action_dim(self) -> int:
        return self._num_joints

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        return self._processed_actions

    def process_actions(self, actions: torch.Tensor):
        self._raw_actions[:] = actions.clamp(-1.0, 1.0)
        scaled_actions = self._raw_actions * self._scale + self._offset
        self._processed_actions[:] = scaled_actions[:]

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        self._raw_actions[env_ids] = 0.0


class JointPositionAction(JointAction):
    cfg: actions_cfg.JointPositionActionCfg

    def __init__(self, cfg: actions_cfg.JointPositionActionCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        if cfg.use_default_offset:
            self._offset = self._asset.data.default_joint_pos[:, self._joint_ids].clone()

    def apply_actions(self):
        self.processed_actions.clamp(
            self._asset.data.soft_joint_pos_limits[:, self._joint_ids, 0],
            self._asset.data.soft_joint_pos_limits[:, self._joint_ids, 1],
        )
        self._asset.set_joint_position_target(self.processed_actions, joint_ids=self._joint_ids)


class RelativeJointPositionAction(JointAction):
    cfg: actions_cfg.RelativeJointPositionActionCfg

    def __init__(self, cfg: actions_cfg.RelativeJointPositionActionCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        if cfg.use_zero_offset:
            self._offset = 0.0

    def apply_actions(self):
        current_actions = self.processed_actions + self._asset.data.joint_pos[:, self._joint_ids]
        current_actions = current_actions.clamp(
            self._asset.data.soft_joint_pos_limits[:, self._joint_ids, 0],
            self._asset.data.soft_joint_pos_limits[:, self._joint_ids, 1],
        )
        self._asset.set_joint_position_target(current_actions, joint_ids=self._joint_ids)


class JointVelocityAction(JointAction):
    cfg: actions_cfg.JointVelocityActionCfg

    def __init__(self, cfg: actions_cfg.JointVelocityActionCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        if cfg.use_default_offset:
            self._offset = self._asset.data.default_joint_vel[:, self._joint_ids].clone()

    def apply_actions(self):
        self._asset.set_joint_velocity_target(self.processed_actions, joint_ids=self._joint_ids)


class GripperAction(JointAction):
    """Gripper action term that applies the processed actions to the gripper joints."""

    cfg: actions_cfg.GripperActionCfg
    """The configuration of the action term."""

    def __init__(self, cfg: actions_cfg.GripperActionCfg, env: ManagerBasedEnv):
        # initialize the action term
        super().__init__(cfg, env)
        # Retrieve IDs of the left and right finger joints
        self._joint_ids, _ = self._asset.find_joints(self.cfg.joint_names)
        if len(self._joint_ids) != 2:
            raise ValueError("GripperAction expects exactly two joint names for controlling the gripper.")
        
    def apply_actions(self):
        # Ensure processed actions are clamped and properly cast to tensor
        self._processed_actions = self.processed_actions.clamp(-0.04, 0.04)

        # Ensure joint_ids is correctly passed as a tensor or list of indices
        joint_ids_tensor = torch.tensor(self._joint_ids, dtype=torch.long, device=self.device)

        # Set the joint positions for the gripper
        self._asset.set_joint_position_target(self._processed_actions, joint_ids=joint_ids_tensor)



class GripperPlaceAction(JointAction):
    """Action term for placing the grasped object at a new position."""

    cfg: actions_cfg.GripperPlaceActionCfg

    def __init__(self, cfg: actions_cfg.GripperPlaceActionCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        
        # Create ArticulationView for the robot
        prim_path = "/World/envs/env_.*/Robot"  
        self._asset = ArticulationView(prim_paths_expr=prim_path, name="robot_articulation")

        # Register prim paths directly to the `_prim_paths` attribute
        self._asset._prim_paths.append(prim_path)

    @property
    def action_dim(self) -> int:
        return 3  

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        return self._processed_actions

    def process_actions(self, actions: torch.Tensor):
        self._raw_actions[:] = actions.clamp(-1.0, 1.0)
        scaled_actions = self._raw_actions * self._scale + self._offset
        self._processed_actions[:] = scaled_actions[:]

    def apply_actions(self):
        # Get the jacobians from the articulation
        jacobians = None
        attempts = 3

        # Attempt to retrieve the jacobian with retries
        for _ in range(attempts):
            jacobians = self._asset.get_jacobians()
            if jacobians is not None:
                break

        if jacobians is None:
            raise ValueError("Jacobian could not be retrieved. Ensure the articulation is initialized correctly.")

        
        jacobian = jacobians[:, -1, :, :]  

        # Compute the pseudo-inverse using the valid Jacobian tensor
        pseudo_inverse = compute_pseudo_inverse(jacobian)

        # Use the pseudo-inverse to determine joint positions to achieve the target
        joint_positions = torch.mm(pseudo_inverse, self.processed_actions.unsqueeze(-1)).squeeze(-1)

        # Set the joint positions for the articulation
        self._asset.set_joint_position_target(joint_positions)

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        self._raw_actions[env_ids] = 0.0



class MoveToTargetPositionAction(ActionTerm):
    """Action class that moves the robotic arm to a target position after grasping an object."""

    cfg: actions_cfg.MoveToTargetPositionActionCfg
    """The configuration of the action term."""

    def __init__(self, cfg: actions_cfg.MoveToTargetPositionActionCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        self._asset: Articulation = env.scene[cfg.asset_name]

        # Set the target position for placing the object on the table
        self.target_position = torch.tensor([0.7, 0.0, 0.35], device=self.device)  

    def apply_actions(self):
        # Calculate the position difference and set the target position
        current_pos = self._asset.data.root_pos_w[:, :3]  # Get current position
        position_diff = self.target_position - current_pos
        processed_action = position_diff * self.cfg.scale
        self._asset.set_joint_position_target(processed_action, joint_ids=slice(None))


# class JointEffortAction(JointAction):
#     """Joint action term that applies the processed actions to the articulation's joints as effort commands."""

#     cfg: actions_cfg.JointEffortActionCfg
#     """The configuration of the action term."""

#     def __init__(self, cfg: actions_cfg.JointEffortActionCfg, env: ManagerBasedEnv):
#         super().__init__(cfg, env)

#     def apply_actions(self):
#         # set joint effort targets
#         self._asset.set_joint_effort_target(self.processed_actions, joint_ids=self._joint_ids)
