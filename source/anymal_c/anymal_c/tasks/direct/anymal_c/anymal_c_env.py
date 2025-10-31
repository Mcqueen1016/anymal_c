# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import math
import torch
from collections.abc import Sequence

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.envs import DirectRLEnv
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from .anymal_c_env_cfg import IsaacLabTutorialEnvCfg

from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
import isaaclab.utils.math as math_utils

def define_markers() -> VisualizationMarkers:
    """Define markers with various different shapes."""
    marker_cfg = VisualizationMarkersCfg(
        prim_path="/Visuals/myMarkers",
        markers={
                "forward": sim_utils.UsdFileCfg(
                    usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/UIElements/arrow_x.usd",
                    scale=(0.25, 0.25, 0.5),
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 1.0)),
                ),
                "command": sim_utils.UsdFileCfg(
                    usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/UIElements/arrow_x.usd",
                    scale=(0.25, 0.25, 0.5),
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0)),
                ),
        },
    )
    return VisualizationMarkers(cfg=marker_cfg)

class IsaacLabTutorialEnv(DirectRLEnv):
    cfg: IsaacLabTutorialEnvCfg

    def __init__(self, cfg: IsaacLabTutorialEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        self.dof_idx, _ = self.robot.find_joints(self.cfg.dof_names)
        print(self.dof_idx)

    def _setup_scene(self):
        # 建立並先註冊到 scene，再 clone（確保會被複製）
        self.robot = Articulation(self.cfg.robot_cfg)
        self.scene.articulations["robot"] = self.robot

        # 共享地面
        spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg())

        # 先建立 markers，再 clone
        self.visualization_markers = define_markers()

        # 複製環境
        self.scene.clone_environments(copy_from_source=False)

        # 共享光源
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

        # 張量設備與初始化
        device = self.device
        num_envs = self.cfg.scene.num_envs
        self.up_dir = torch.tensor([0.0, 0.0, 1.0], device=device)

        # 隨機指令，限制在水平面並正規化
        self.commands = torch.randn((num_envs, 3), device=device)
        self.commands[:, -1] = 0.0
        norms = torch.linalg.norm(self.commands[:, :2], dim=1, keepdim=True).clamp_min(1e-6)
        self.commands[:, :2] = self.commands[:, :2] / norms

        # 用 atan2 計算 yaw
        self.yaws = torch.atan2(self.commands[:, 1], self.commands[:, 0]).unsqueeze(1)

        # markers 緩衝
        self.marker_locations = torch.zeros((num_envs, 3), device=device)
        self.marker_offset = torch.zeros((num_envs, 3), device=device)
        self.marker_offset[:, -1] = 0.75
        self.forward_marker_orientations = torch.zeros((num_envs, 4), device=device)
        self.command_marker_orientations = torch.zeros((num_envs, 4), device=device)

    def _visualize_markers(self):
        # get marker locations and orientations
        self.marker_locations = self.robot.data.root_pos_w
        self.forward_marker_orientations = self.robot.data.root_quat_w
        self.command_marker_orientations = math_utils.quat_from_angle_axis(self.yaws, self.up_dir).squeeze()

        # offset markers so they are above the robot
        loc = self.marker_locations + self.marker_offset
        loc = torch.vstack((loc, loc))
        rots = torch.vstack((self.forward_marker_orientations, self.command_marker_orientations))

        # render the markers
        all_envs = torch.arange(self.cfg.scene.num_envs, device=self.device)
        indices = torch.hstack((torch.zeros_like(all_envs), torch.ones_like(all_envs)))
        self.visualization_markers.visualize(loc, rots, marker_indices=indices)

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self.actions = actions.clone()
        self._visualize_markers()

    def _apply_action(self) -> None:
        # 12 維速度控制（之後可改成 position/torque）
        self.robot.set_joint_velocity_target(self.actions, joint_ids=self.dof_idx)

    def _get_observations(self) -> dict:
        self.velocity = self.robot.data.root_com_vel_w
        self.forwards = math_utils.quat_apply(self.robot.data.root_link_quat_w, self.robot.data.FORWARD_VEC_B)

        dot = torch.sum(self.forwards * self.commands, dim=-1, keepdim=True)
        cross = torch.cross(self.forwards, self.commands, dim=-1)[:, -1].reshape(-1, 1)
        forward_speed = self.robot.data.root_com_lin_vel_b[:, 0].reshape(-1, 1)
        obs = torch.hstack((dot, cross, forward_speed))
        return {"policy": obs}

    def _get_rewards(self) -> torch.Tensor:
        forward_reward = self.robot.data.root_com_lin_vel_b[:, 0].reshape(-1, 1)
        alignment_reward = torch.sum(self.forwards * self.commands, dim=-1, keepdim=True)
        total_reward = forward_reward * torch.exp(alignment_reward)
        return total_reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        # return 1D tensors: shape [num_envs]
        time_out = (self.episode_length_buf >= self.max_episode_length - 1)
        terminated = torch.zeros_like(time_out, dtype=torch.bool)
        return terminated, time_out

    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None:
            env_ids = self.robot._ALL_INDICES
        super()._reset_idx(env_ids)

        device = self.device
        env_ids = torch.as_tensor(env_ids, device=device)

        # 產生新指令（水平面）並正規化
        self.commands[env_ids] = torch.randn((len(env_ids), 3), device=device)
        self.commands[env_ids, -1] = 0.0
        norms = torch.linalg.norm(self.commands[env_ids, :2], dim=1, keepdim=True).clamp_min(1e-6)
        self.commands[env_ids, :2] = self.commands[env_ids, :2] / norms

        # 計算新的 yaw
        self.yaws[env_ids] = torch.atan2(self.commands[env_ids, 1], self.commands[env_ids, 0]).unsqueeze(1)

        # 重設 root 狀態
        default_root_state = self.robot.data.default_root_state[env_ids]
        default_root_state[:, :3] += self.scene.env_origins[env_ids]
        self.robot.write_root_state_to_sim(default_root_state, env_ids)
        self._visualize_markers()

