# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Custom Lift Environment for RL training with dynamic obstacles."""

from omni.isaac.lab.envs.manager_based_rl_env import ManagerBasedRLEnv
from omni.isaac.lab.managers import SceneEntityCfg

from omni.isaac.lab_tasks.manager_based.manipulation.lift.mdp.dynamic_obstacle_manager import DynamicObstacleManager

from omni.isaac.lab_tasks.manager_based.manipulation.lift.lift_env_cfg import LiftEnvCfg



class CustomLiftEnv(ManagerBasedRLEnv):
    """Custom lifting environment with dynamic obstacles."""

    def __init__(self, cfg: LiftEnvCfg):
        super().__init__(cfg)
        self.current_time_step = 0  # Initialize time step

        # Initialize dynamic obstacle manager
        self.dynamic_obstacle_manager = DynamicObstacleManager(
            obstacle_cfg=SceneEntityCfg("obstacle"), env=self
        )

    def step(self, actions):
        """Override step to include dynamic obstacle updates and time step management."""
        # Update the dynamic obstacle positions
        if self.cfg.dynamic_obstacles:  # Ensure the dynamic obstacles setting is enabled
            self.dynamic_obstacle_manager.update(self.current_time_step, self.cfg.total_time_steps)

        # Increment the current time step
        self.current_time_step += 1
        if self.current_time_step >= self.cfg.total_time_steps:
            self.current_time_step = 0  # Reset time step after one episode

        # Call the parent step function
        return super().step(actions)

    def reset(self):
        """Override reset to ensure time step and obstacles are reset."""
        self.current_time_step = 0  # Reset time step
        self.dynamic_obstacle_manager.reset()  # Reset dynamic obstacles if needed
        return super().reset()