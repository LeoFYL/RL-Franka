# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from dataclasses import MISSING

import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.assets import ArticulationCfg, AssetBaseCfg
from omni.isaac.lab.envs import ManagerBasedRLEnvCfg
from omni.isaac.lab.managers.action_manager import ActionTerm, ActionTermCfg

from omni.isaac.lab.managers import CurriculumTermCfg as CurrTerm
from omni.isaac.lab.managers import EventTermCfg as EventTerm
from omni.isaac.lab.managers import ObservationGroupCfg as ObsGroup
from omni.isaac.lab.managers import ObservationTermCfg as ObsTerm
from omni.isaac.lab.managers import RewardTermCfg as RewTerm
from omni.isaac.lab.managers import SceneEntityCfg
from omni.isaac.lab.managers import TerminationTermCfg as DoneTerm
from omni.isaac.lab.scene import InteractiveSceneCfg
from omni.isaac.lab.utils import configclass
from omni.isaac.lab.utils.assets import ISAAC_NUCLEUS_DIR
from omni.isaac.lab.utils.noise import AdditiveUniformNoiseCfg as Unoise

import isaacLab.manipulation.tasks.Robot_arm.reach.mdp as mdp

from isaacLab.manipulation.tasks.Robot_arm.reach.mdp.actions import GripperAction, GripperPlaceAction
from isaacLab.manipulation.tasks.Robot_arm.reach.mdp.actions_cfg import GripperActionCfg, GripperPlaceActionCfg
from isaacLab.manipulation.tasks.Robot_arm.reach.mdp.actions_cfg import JointPositionActionCfg


from omni.isaac.core.prims.xform_prim_view import XFormPrimView
from omni.isaac.core.utils.prims import get_prim_at_path, create_prim


# Check if the prim already exists, if not create it
prim_path = "/World/envs/env_0/TargetPlacePosition"
if not create_prim(prim_path):
    create_prim(prim_path)

prim_path_gripper = "/World/envs/env_0/Gripper"
if not create_prim(prim_path_gripper):
    create_prim(prim_path_gripper)


##
# Scene definition
##

@configclass
class ReachSceneCfg(InteractiveSceneCfg):
    """Configuration for the scene with a robotic arm."""

    def __post_init__(self):
        # Create necessary prims or verify their existence
        prim_path_robot = "/World/envs/env_0/Robot"
        if get_prim_at_path(prim_path_robot) is None:
            create_prim(prim_path_robot)

        super().__post_init__()

    # Use PlaceholderCfg when no asset needs to be spawned
    target_place_position = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/TargetPlacePosition",
        spawn=None, 
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.85, 0, 0), rot=(1.0, 0.0, 0.0, 0.0)),
    )

    # World configuration
    ground = AssetBaseCfg(
        prim_path="/World/ground",
        spawn=sim_utils.GroundPlaneCfg(),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, -1.05)),
    )

    table = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Table",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Mounts/SeattleLabTable/table_instanceable.usd",
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.55, 0.0, 0.0), rot=(0.70711, 0.0, 0.0, 0.70711)),
    )

    # Robot configuration

    gripper = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Gripper",
        spawn=None,  # the gripper is a component of the main robot articulation
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.0), 
            rot=(1.0, 0.0, 0.0, 0.0)  # position and rotation 
        ),
    )

    robot: ArticulationCfg = ArticulationCfg(
        class_type=MISSING,
        prim_path="/World/envs/env_.*/Robot",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Robots/FrankaEmika/panda_instanceable.usd"
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.0),
            rot=(1.0, 0.0, 0.0, 0.0)
        )
    )
   

    # Object to grasp
    object = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Cube",
        spawn=sim_utils.UsdFileCfg(
            usd_path="omniverse://localhost/NVIDIA/Assets/Isaac/4.1/Isaac/Props/Blocks/yellow_block.usd",
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.55, 0.0, 0.0), rot=(1.0, 0.0, 0.0, 0.0)),
    )

    # Light configuration
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=2500.0),
    )


##
# MDP settings
##

@configclass
class CommandsCfg:
    """Command terms for the MDP."""

    ee_pose = mdp.UniformPoseCommandCfg(
        asset_name="robot",
        body_name="panda_hand",  # Specify the end-effector name
        resampling_time_range=(4.0, 4.0),
        debug_vis=True,
        ranges=mdp.UniformPoseCommandCfg.Ranges(
            pos_x=(0.35, 0.65),
            pos_y=(-0.2, 0.2),
            pos_z=(0.15, 0.5),
            roll=(0.0, 0.0),
            pitch=(0.0, 0.0),
            yaw=(-3.14, 3.14),
        ),
    )

    place_pose = mdp.UniformPoseCommandCfg(
        asset_name="robot",
        body_name="panda_hand",
        resampling_time_range=(4.0, 4.0),
        debug_vis=True,
        ranges=mdp.UniformPoseCommandCfg.Ranges(
            pos_x=(0.45, 0.55),
            pos_y=(-0.2, 0.2),
            pos_z=(0.15, 0.5),
            roll=(0.0, 0.0),
            pitch=(0.0, 0.0),
            yaw=(-3.14, 3.14),
        ),
    )


@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    arm_action: JointPositionActionCfg = JointPositionActionCfg()
    gripper_action: GripperActionCfg = GripperActionCfg()  # Added gripper control
    place_action: GripperPlaceActionCfg = GripperPlaceActionCfg()  

@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        # Existing observation terms
        joint_pos = ObsTerm(func=mdp.joint_pos_rel, noise=Unoise(n_min=-0.01, n_max=0.01))
        joint_vel = ObsTerm(func=mdp.joint_vel_rel, noise=Unoise(n_min=-0.01, n_max=0.01))
        pose_command = ObsTerm(func=mdp.generated_commands, params={"command_name": "ee_pose"})
        actions = ObsTerm(func=mdp.last_action)

        # Add observation for object position relative to the gripper
        object_pos_rel = ObsTerm(func=mdp.object_position_in_robot_root_frame, params={"object_cfg": SceneEntityCfg("object")})

        # Add observation for gripper state
        gripper_state = ObsTerm(func=mdp.last_action_clamped, params={"action_name": "gripper_action"})

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    # observation groups
    policy: PolicyCfg = PolicyCfg()


@configclass
class EventCfg:
    """Configuration for events."""

    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_scale,
        mode="reset",
        params={
            "position_range": (0.5, 1.5),
            "velocity_range": (0.0, 0.0),
        },
    )

# Instance of ReachSceneCfg used across all classes
scene_instance = ReachSceneCfg()

@configclass
class RewardsCfg:
    """Reward terms for the MDP."""

    # task terms
    end_effector_position_tracking = RewTerm(
        func=mdp.position_command_error,
        weight=-0.2,
        params={"asset_cfg": SceneEntityCfg("robot", body_names=MISSING), "command_name": "ee_pose"},
    )
    end_effector_position_tracking_fine_grained = RewTerm(
        func=mdp.position_command_error,
        weight=0.1,
        params={"asset_cfg": SceneEntityCfg("robot", body_names=MISSING), "command_name": "ee_pose"},
    )

    end_effector_orientation_tracking = RewTerm(
        func=mdp.position_command_error,
        weight=-0.1,
        params={"asset_cfg": SceneEntityCfg("robot", body_names=MISSING), "command_name": "ee_pose"},
    )

    # action penalty
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-0.0001)
    joint_vel = RewTerm(
        func=mdp.joint_vel_l2,
        weight=-0.0001,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )

    object_grasped = RewTerm(
        func=mdp.object_grasped,
        weight=1.0,
        params={"object_cfg": SceneEntityCfg("object"), "gripper_cfg": SceneEntityCfg("robot")},
    )
    
    # Update the parameter name in TerminationsCfg and RewardsCfg
    object_placed = RewTerm(
        func=mdp.object_placed,
        weight=1.5,
        params={
            "object_cfg": SceneEntityCfg("object"),
            "place_position": (0.85, 0, 0),  # Make sure to pass only the expected mandatory parameters.
            "precision": 0.05  # Optional parameter.
        }
    )


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    object_is_lifted = DoneTerm(
        func=mdp.object_grasped,
        params={
            "gripper_cfg": SceneEntityCfg(name="gripper"),
            "object_cfg": SceneEntityCfg(name="object"),
            "threshold": 0.1  # Optional parameter
        }
    )

    object_is_placed = DoneTerm(
        func=mdp.object_placed,
        params={
            "object_cfg": SceneEntityCfg(name="object"),
            "place_position": (0.85, 0, 0),  # Mandatory parameter.
            "precision": 0.05  # Optional parameter.
        }
    )




@configclass
class CurriculumCfg:
    """Curriculum terms for the MDP."""

    action_rate = CurrTerm(
        func=mdp.modify_reward_weight, params={"term_name": "action_rate", "weight": -0.005, "num_steps": 4500}
    )

    joint_vel = CurrTerm(
        func=mdp.modify_reward_weight, params={"term_name": "joint_vel", "weight": -0.001, "num_steps": 4500}
    )


##
# Environment configuration
##

@configclass
class ReachEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the reach end-effector pose tracking environment."""

    # Scene settings
    scene: ReachSceneCfg = ReachSceneCfg(num_envs=4096, env_spacing=2.5)
    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    # MDP settings
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

    def __post_init__(self):
        """Post initialization."""
        # general settings
        self.decimation = 2
        self.episode_length_s = 12.0
        self.viewer.eye = (3.5, 3.5, 3.5)
        # simulation settings
        self.sim.dt = 1.0 / 60.0

