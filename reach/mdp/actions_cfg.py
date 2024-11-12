# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
from dataclasses import MISSING

from omni.isaac.lab.controllers import DifferentialIKControllerCfg
from omni.isaac.lab.managers.action_manager import ActionTerm, ActionTermCfg
from omni.isaac.lab.utils import configclass

from . import actions

##
# Joint actions.
##


@configclass
class JointActionCfg(ActionTermCfg):
    asset_name: str = "robot"  # Set the correct asset name
    joint_names: list[str] = ["panda_joint1", "panda_joint2", "panda_joint3", "panda_joint4", "panda_joint5", "panda_joint6", "panda_joint7"]  # Include all robot joints for full control
    scale: float | dict[str, float] = 1.0
    offset: float | dict[str, float] = 0.0

@configclass
class JointPositionActionCfg(JointActionCfg):
    """Configuration for the joint position action term.

    See :class:`JointPositionAction` for more details.
    """

    class_type: type[ActionTerm] = actions.JointPositionAction

    use_default_offset: bool = True
    """Whether to use default joint positions configured in the articulation asset as offset.
    Defaults to True.

    If True, this flag results in overwriting the values of :attr:`offset` to the default joint positions
    from the articulation asset.
    """


@configclass
class RelativeJointPositionActionCfg(JointActionCfg):
    """Configuration for the relative joint position action term.

    See :class:`RelativeJointPositionAction` for more details.
    """

    class_type: type[ActionTerm] = actions.RelativeJointPositionAction

    use_zero_offset: bool = True
    """Whether to ignore the offset defined in articulation asset. Defaults to True.

    If True, this flag results in overwriting the values of :attr:`offset` to zero.
    """


@configclass
class JointVelocityActionCfg(JointActionCfg):
    """Configuration for the joint velocity action term.

    See :class:`JointVelocityAction` for more details.
    """

    class_type: type[ActionTerm] = actions.JointVelocityAction

    use_default_offset: bool = True
    """Whether to use default joint velocities configured in the articulation asset as offset.
    Defaults to True.

    This overrides the settings from :attr:`offset` if set to True.
    """


@configclass
class GripperActionCfg(ActionTermCfg):
    class_type: type[ActionTerm] = actions.GripperAction
    asset_name: str = "robot"  # Set asset_name to match the robot
    joint_names: list[str] = ["panda_finger_joint1", "panda_finger_joint2"]  
    scale: float = 0.04
    offset: float = 0.0

    # Adding a more explicit range for gripper actions to increase success probability of grasping
    min_position: float = 0.0  # Minimum opening
    max_position: float = 0.04  # Maximum closing

@configclass
class GripperPlaceActionCfg(ActionTermCfg):
    """Configuration for placing the object using the gripper."""

    class_type: type[ActionTerm] = actions.GripperPlaceAction  
    asset_name: str = "robot"  # Set the correct asset name
    joint_names: list[str] = ["panda_finger_joint1", "panda_finger_joint2"]  # Control gripper joints for placing
    scale: float = 0.04
    offset: float = 0.0
    min_position: float = 0.0  # Minimum gripper opening for releasing the object
    max_position: float = 0.04  # Maximum gripper opening for releasing the object
    
# MoveToTargetPositionAction Configuration
@configclass
class MoveToTargetPositionActionCfg(ActionTermCfg):
    class_type: type[ActionTerm] = actions.MoveToTargetPositionAction
    asset_name: str = "robot"  # Set asset_name to match the robot
    scale: float = 1.0  # Control the movement speed of the arm
    target_position: tuple[float, float, float] = (0.7, 0.0, 0.35)  # Target position to place the object

# full sequence of actions required for the task
@configclass
class FullTaskActionCfg:
    joint_action: JointPositionActionCfg = JointPositionActionCfg()
    gripper_action: GripperActionCfg = GripperActionCfg()
    move_to_target_action: MoveToTargetPositionActionCfg = MoveToTargetPositionActionCfg()




    
# @configclass
# class JointEffortActionCfg(JointActionCfg):
#     """Configuration for the joint effort action term.

#     See :class:`JointEffortAction` for more details.
#     """

#     class_type: type[ActionTerm] = joint_actions.JointEffortAction


# ##
# # Joint actions rescaled to limits.
# ##


# @configclass
# class JointPositionToLimitsActionCfg(ActionTermCfg):
#     """Configuration for the bounded joint position action term.

#     See :class:`JointPositionWithinLimitsAction` for more details.
#     """

#     class_type: type[ActionTerm] = joint_actions_to_limits.JointPositionToLimitsAction

#     joint_names: list[str] = MISSING
#     """List of joint names or regex expressions that the action will be mapped to."""

#     scale: float | dict[str, float] = 1.0
#     """Scale factor for the action (float or dict of regex expressions). Defaults to 1.0."""

#     rescale_to_limits: bool = True
#     """Whether to rescale the action to the joint limits. Defaults to True.

#     If True, the input actions are rescaled to the joint limits, i.e., the action value in
#     the range [-1, 1] corresponds to the joint lower and upper limits respectively.

#     Note:
#         This operation is performed after applying the scale factor.
#     """


# @configclass
# class EMAJointPositionToLimitsActionCfg(JointPositionToLimitsActionCfg):
#     """Configuration for the exponential moving average (EMA) joint position action term.

#     See :class:`EMAJointPositionToLimitsAction` for more details.
#     """

#     class_type: type[ActionTerm] = joint_actions_to_limits.EMAJointPositionToLimitsAction

#     alpha: float | dict[str, float] = 1.0
#     """The weight for the moving average (float or dict of regex expressions). Defaults to 1.0.

#     If set to 1.0, the processed action is applied directly without any moving average window.
#     """


# ##
# # Gripper actions.
# ##


# @configclass
# class BinaryJointActionCfg(ActionTermCfg):
#     """Configuration for the base binary joint action term.

#     See :class:`BinaryJointAction` for more details.
#     """

#     joint_names: list[str] = MISSING
#     """List of joint names or regex expressions that the action will be mapped to."""
#     open_command_expr: dict[str, float] = MISSING
#     """The joint command to move to *open* configuration."""
#     close_command_expr: dict[str, float] = MISSING
#     """The joint command to move to *close* configuration."""


# @configclass
# class BinaryJointPositionActionCfg(BinaryJointActionCfg):
#     """Configuration for the binary joint position action term.

#     See :class:`BinaryJointPositionAction` for more details.
#     """

#     class_type: type[ActionTerm] = binary_joint_actions.BinaryJointPositionAction


# @configclass
# class BinaryJointVelocityActionCfg(BinaryJointActionCfg):
#     """Configuration for the binary joint velocity action term.

#     See :class:`BinaryJointVelocityAction` for more details.
#     """

#     class_type: type[ActionTerm] = binary_joint_actions.BinaryJointVelocityAction


# ##
# # Non-holonomic actions.
# ##


# @configclass
# class NonHolonomicActionCfg(ActionTermCfg):
#     """Configuration for the non-holonomic action term with dummy joints at the base.

#     See :class:`NonHolonomicAction` for more details.
#     """

#     class_type: type[ActionTerm] = non_holonomic_actions.NonHolonomicAction

#     body_name: str = MISSING
#     """Name of the body which has the dummy mechanism connected to."""
#     x_joint_name: str = MISSING
#     """The dummy joint name in the x direction."""
#     y_joint_name: str = MISSING
#     """The dummy joint name in the y direction."""
#     yaw_joint_name: str = MISSING
#     """The dummy joint name in the yaw direction."""
#     scale: tuple[float, float] = (1.0, 1.0)
#     """Scale factor for the action. Defaults to (1.0, 1.0)."""
#     offset: tuple[float, float] = (0.0, 0.0)
#     """Offset factor for the action. Defaults to (0.0, 0.0)."""


# ##
# # Task-space Actions.
# ##


# @configclass
# class DifferentialInverseKinematicsActionCfg(ActionTermCfg):
#     """Configuration for inverse differential kinematics action term.

#     See :class:`DifferentialInverseKinematicsAction` for more details.
#     """

#     @configclass
#     class OffsetCfg:
#         """The offset pose from parent frame to child frame.

#         On many robots, end-effector frames are fictitious frames that do not have a corresponding
#         rigid body. In such cases, it is easier to define this transform w.r.t. their parent rigid body.
#         For instance, for the Franka Emika arm, the end-effector is defined at an offset to the the
#         "panda_hand" frame.
#         """

#         pos: tuple[float, float, float] = (0.0, 0.0, 0.0)
#         """Translation w.r.t. the parent frame. Defaults to (0.0, 0.0, 0.0)."""
#         rot: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)
#         """Quaternion rotation ``(w, x, y, z)`` w.r.t. the parent frame. Defaults to (1.0, 0.0, 0.0, 0.0)."""

#     class_type: type[ActionTerm] = task_space_actions.DifferentialInverseKinematicsAction

#     joint_names: list[str] = MISSING
#     """List of joint names or regex expressions that the action will be mapped to."""
#     body_name: str = MISSING
#     """Name of the body or frame for which IK is performed."""
#     body_offset: OffsetCfg | None = None
#     """Offset of target frame w.r.t. to the body frame. Defaults to None, in which case no offset is applied."""
#     scale: float | tuple[float, ...] = 1.0
#     """Scale factor for the action. Defaults to 1.0."""
#     controller: DifferentialIKControllerCfg = MISSING
#     """The configuration for the differential IK controller."""