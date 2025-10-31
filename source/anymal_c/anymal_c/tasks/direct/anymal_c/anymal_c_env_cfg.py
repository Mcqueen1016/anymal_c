# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from anymal_c.robots.anymal import ANYMAL_C_CONFIG

from isaaclab.assets import ArticulationCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass

@configclass
class IsaacLabTutorialEnvCfg(DirectRLEnvCfg):
    # env
    decimation = 2
    episode_length_s = 20.0

    dof_names = [
        "LF_HAA", "LF_HFE", "LF_KFE",
        "RF_HAA", "RF_HFE", "RF_KFE",
        "LH_HAA", "LH_HFE", "LH_KFE",
        "RH_HAA", "RH_HFE", "RH_KFE",
    ]

    # - spaces definition
    action_space = 12
    observation_space = 3
    state_space = 0
    # simulation
    sim: SimulationCfg = SimulationCfg(dt=1 / 120, render_interval=decimation)
    # robot(s)
    robot_cfg: ArticulationCfg = ANYMAL_C_CONFIG.replace(prim_path="/World/envs/env_.*/AnymalC")
    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=64, env_spacing=2.0, replicate_physics=True)
    