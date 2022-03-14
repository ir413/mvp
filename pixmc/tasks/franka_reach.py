#!/usr/bin/env python3

"""FrankaCabinet task."""

import numpy as np
import os
import torch
import imageio
import random

from typing import Tuple
from torch import Tensor

from pixmc.utils.torch_jit_utils import *
from pixmc.tasks.base.base_task import BaseTask

from isaacgym import gymtorch
from isaacgym import gymapi


class FrankaReach(BaseTask):

    def __init__(self, cfg, sim_params, physics_engine, device_type, device_id, headless):
        self.cfg = cfg
        self.sim_params = sim_params
        self.physics_engine = physics_engine
        assert self.physics_engine == gymapi.SIM_PHYSX

        self.max_episode_length = self.cfg["env"]["episodeLength"]

        self.dof_vel_scale = self.cfg["env"]["dofVelocityScale"]
        self.action_scale = self.cfg["env"]["actionScale"]

        self.goal_dist_reward_scale = self.cfg["env"]["goalDistRewardScale"]
        self.goal_bonus_reward_scale = self.cfg["env"]["goalBonusRewardScale"]
        self.action_penalty_scale = self.cfg["env"]["actionPenaltyScale"]

        self.up_axis = "z"
        self.up_axis_idx = 2

        self.distX_offset = 0.04
        self.dt = 1 / 60.

        self.obs_type = self.cfg["env"]["obs_type"]
        assert self.obs_type in ["robot", "oracle", "pixels"]

        if self.obs_type == "robot":
            num_obs = 9 * 2
            self.compute_observations = self.compute_robot_obs
        elif self.obs_type == "oracle":
            num_obs = 27
            self.compute_observations = self.compute_oracle_obs
        else:
            self.cam_crop = self.cfg["env"]["cam"]["crop"]
            self.cam_w = self.cfg["env"]["cam"]["w"]
            self.cam_h = self.cfg["env"]["cam"]["h"]
            self.cam_fov = self.cfg["env"]["cam"]["fov"]
            self.cam_ss = self.cfg["env"]["cam"]["ss"]
            self.cam_loc_p = self.cfg["env"]["cam"]["loc_p"]
            self.cam_loc_r = self.cfg["env"]["cam"]["loc_r"]
            self.im_size = self.cfg["env"]["im_size"]
            num_obs = (3, self.im_size, self.im_size)
            self.compute_observations = self.compute_pixel_obs
            assert self.cam_crop in ["center", "left"]
            assert self.cam_h == self.im_size
            assert self.cam_w % 2 == 0

        self.cfg["env"]["numObservations"] = num_obs
        self.cfg["env"]["numStates"]= 9 * 2
        self.cfg["env"]["numActions"] = 9

        self.cfg["device_type"] = device_type
        self.cfg["device_id"] = device_id
        self.cfg["headless"] = headless

        super().__init__(cfg=self.cfg, enable_camera_sensors=(self.obs_type == "pixels"))

        actor_root_state_tensor = self.gym.acquire_actor_root_state_tensor(self.sim)
        dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        rigid_body_tensor = self.gym.acquire_rigid_body_state_tensor(self.sim)
        net_contact_forces = self.gym.acquire_net_contact_force_tensor(self.sim)

        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)

        # Default franka dof pos
        self.franka_default_dof_pos = to_torch(
            [1.157, -1.066, -0.155, -2.239, -1.841, 1.003, 0.469, 0.035, 0.035], device=self.device
        )

        # Dof state slices
        self.dof_state = gymtorch.wrap_tensor(dof_state_tensor)
        self.franka_dof_state = self.dof_state.view(self.num_envs, self.num_franka_dofs, 2)
        self.franka_dof_pos = self.franka_dof_state[..., 0]
        self.franka_dof_vel = self.franka_dof_state[..., 1]

        # (N, num_bodies, 13)
        self.rigid_body_states = gymtorch.wrap_tensor(rigid_body_tensor).view(self.num_envs, -1, 13)

        # (N, 3, 13)
        self.root_state_tensor = gymtorch.wrap_tensor(actor_root_state_tensor).view(self.num_envs, -1, 13)

        # (N, num_bodies, 3)
        self.contact_forces = gymtorch.wrap_tensor(net_contact_forces).view(self.num_envs, -1, 3)

        # Finger pos
        self.lfinger_pos = self.rigid_body_states[:, self.rigid_body_lfinger_ind, :3]
        self.rfinger_pos = self.rigid_body_states[:, self.rigid_body_rfinger_ind, :3]

        # Finger rot
        self.lfinger_rot = self.rigid_body_states[:, self.rigid_body_lfinger_ind, 3:7]
        self.rfinger_rot = self.rigid_body_states[:, self.rigid_body_rfinger_ind, 3:7]

        # Goal pos
        self.goal_pos = self.root_state_tensor[:, self.env_goal_ind, :3]

        # Dof targets
        self.dof_targets = torch.zeros((self.num_envs, self.num_franka_dofs), dtype=torch.float, device=self.device)

        # Global inds
        self.global_indices = torch.arange(
            self.num_envs * (1 + 1 + 1), dtype=torch.int32, device=self.device
        ).view(self.num_envs, -1)

        # Franka dof pos and vel scaled
        self.franka_dof_pos_scaled = torch.zeros_like(self.franka_dof_pos)
        self.franka_dof_vel_scaled = torch.zeros_like(self.franka_dof_vel)

        # Tracks hand pos and to goal vec # TODO: rename
        self.hand_pos = torch.zeros((self.num_envs, 3), dtype=torch.float, device=self.device)
        self.to_target = torch.zeros((self.num_envs, 3), dtype=torch.float, device=self.device)

        # Image mean and std
        if self.obs_type == "pixels":
            self.im_mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float, device=self.device).view(3, 1, 1)
            self.im_std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float, device=self.device).view(3, 1, 1)

        # Goal pos randomization
        self.goal_pos_init = torch.tensor(cfg["env"]["goal_pos_init"], dtype=torch.float, device=self.device)
        self.goal_pos_delta = torch.tensor(cfg["env"]["goal_pos_delta"], dtype=torch.float, device=self.device)

        # Success counts
        self.successes = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        self.extras["successes"] = self.successes

        self.reset(torch.arange(self.num_envs, device=self.device))

    def create_sim(self):
        self.sim_params.up_axis = gymapi.UP_AXIS_Z
        self.sim_params.gravity.x = 0
        self.sim_params.gravity.y = 0
        self.sim_params.gravity.z = -9.81
        self.sim = super().create_sim(
            self.device_id, self.graphics_device_id, self.physics_engine, self.sim_params)
        self._create_ground_plane()
        self._create_envs(self.num_envs, self.cfg["env"]['envSpacing'], int(np.sqrt(self.num_envs)))

    def _create_ground_plane(self):
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        self.gym.add_ground(self.sim, plane_params)

    def _create_envs(self, num_envs, spacing, num_per_row):
        lower = gymapi.Vec3(-spacing, -spacing, 0.0)
        upper = gymapi.Vec3(spacing, spacing, spacing)

        # Retrieve asset paths
        asset_root = self.cfg["env"]["asset"]["assetRoot"]
        franka_asset_file = self.cfg["env"]["asset"]["assetFileNameFranka"]

        # Load franka asset
        asset_options = gymapi.AssetOptions()
        asset_options.flip_visual_attachments = True
        asset_options.fix_base_link = True
        asset_options.collapse_fixed_joints = False
        asset_options.disable_gravity = True
        asset_options.thickness = 0.001
        asset_options.default_dof_drive_mode = gymapi.DOF_MODE_POS
        asset_options.use_mesh_materials = True
        franka_asset = self.gym.load_asset(self.sim, asset_root, franka_asset_file, asset_options)

        # Create table asset
        table_dims = gymapi.Vec3(0.6, 1.0, 0.4)
        asset_options = gymapi.AssetOptions()
        asset_options.fix_base_link = True
        table_asset = self.gym.create_box(self.sim, table_dims.x, table_dims.y, table_dims.z, asset_options)

        # Create goal asset
        sphere_r = 0.04
        asset_options = gymapi.AssetOptions()
        asset_options.disable_gravity = True
        goal_asset = self.gym.create_sphere(self.sim, sphere_r, asset_options)

        self.num_franka_bodies = self.gym.get_asset_rigid_body_count(franka_asset)
        self.num_franka_dofs = self.gym.get_asset_dof_count(franka_asset)

        print("num franka bodies: ", self.num_franka_bodies)
        print("num franka dofs: ", self.num_franka_dofs)

        # Franka dof gains (should probably be in the config)
        franka_dof_stiffness = [400, 400, 400, 400, 400, 400, 400, 1.0e6, 1.0e6]
        franka_dof_damping = [80, 80, 80, 80, 80, 80, 80, 1.0e2, 1.0e2]

        # Set franka dof props
        franka_dof_props = self.gym.get_asset_dof_properties(franka_asset)
        for i in range(self.num_franka_dofs):
            franka_dof_props["driveMode"][i] = gymapi.DOF_MODE_POS
            franka_dof_props["stiffness"][i] = franka_dof_stiffness[i]
            franka_dof_props["damping"][i] = franka_dof_damping[i]

        # Record franka dof limits
        self.franka_dof_lower_limits = torch.zeros(self.num_franka_dofs, device=self.device, dtype=torch.float)
        self.franka_dof_upper_limits = torch.zeros(self.num_franka_dofs, device=self.device, dtype=torch.float)
        for i in range(self.num_franka_dofs):
            self.franka_dof_lower_limits[i] = franka_dof_props["lower"][i].item()
            self.franka_dof_upper_limits[i] = franka_dof_props["upper"][i].item()

        # Set franka gripper dof props
        self.franka_dof_speed_scales = torch.ones_like(self.franka_dof_lower_limits)
        self.franka_dof_speed_scales[[7, 8]] = 0.1
        franka_dof_props['effort'][7] = 200
        franka_dof_props['effort'][8] = 200

        franka_start_pose = gymapi.Transform()
        franka_start_pose.p = gymapi.Vec3(1.0, 0.0, 0.0)
        franka_start_pose.r = gymapi.Quat(0.0, 0.0, 1.0, 0.0)

        table_start_pose = gymapi.Transform()
        table_start_pose.p = gymapi.Vec3(0.5, 0.0, 0.5 * table_dims.z)

        goal_start_pose = gymapi.Transform()
        goal_start_pose.p = gymapi.Vec3(0.55, 0.0, 0.7)

        # Compute aggregate size
        num_franka_bodies = self.gym.get_asset_rigid_body_count(franka_asset)
        num_franka_shapes = self.gym.get_asset_rigid_shape_count(franka_asset)
        num_table_bodies = self.gym.get_asset_rigid_body_count(table_asset)
        num_table_shapes = self.gym.get_asset_rigid_shape_count(table_asset)
        num_goal_bodies = self.gym.get_asset_rigid_body_count(goal_asset)
        num_goal_shapes = self.gym.get_asset_rigid_shape_count(goal_asset)
        max_agg_bodies = num_franka_bodies + num_table_bodies + num_goal_bodies
        max_agg_shapes = num_franka_shapes + num_table_shapes + num_goal_shapes

        self.frankas = []
        self.tables = []
        self.goals = []
        self.envs = []

        if self.obs_type == "pixels":
            self.cams = []
            self.cam_tensors = []

        for i in range(self.num_envs):
            # Create env instance
            env_ptr = self.gym.create_env(self.sim, lower, upper, num_per_row)

            # Aggregate actors
            self.gym.begin_aggregate(env_ptr, max_agg_bodies, max_agg_shapes, True)

            # Franka actor
            franka_actor = self.gym.create_actor(env_ptr, franka_asset, franka_start_pose, "franka", i, 1, 0)
            self.gym.set_actor_dof_properties(env_ptr, franka_actor, franka_dof_props)

            # Table actor
            table_actor = self.gym.create_actor(env_ptr, table_asset, table_start_pose, "table", i, 0, 0)

            # Goal actor
            goal_actor = self.gym.create_actor(env_ptr, goal_asset, goal_start_pose, "goal", self.num_envs + i, 0, 0)
            goal_color = gymapi.Vec3(0.466, 0.674, 0.188)
            self.gym.set_rigid_body_color(env_ptr, goal_actor, 0, gymapi.MESH_VISUAL_AND_COLLISION, goal_color)

            self.gym.end_aggregate(env_ptr)

            self.envs.append(env_ptr)
            self.frankas.append(franka_actor)
            self.tables.append(table_actor)
            self.goals.append(goal_actor)

            # Set up camera
            if self.obs_type == "pixels":
                # Camera
                cam_props = gymapi.CameraProperties()
                cam_props.width = self.cam_w
                cam_props.height = self.cam_h
                cam_props.horizontal_fov = self.cam_fov
                cam_props.supersampling_horizontal = self.cam_ss
                cam_props.supersampling_vertical = self.cam_ss
                cam_props.enable_tensors = True
                cam_handle = self.gym.create_camera_sensor(env_ptr, cam_props)
                rigid_body_hand_ind = self.gym.find_actor_rigid_body_handle(env_ptr, franka_actor, "panda_hand")
                local_t = gymapi.Transform()
                local_t.p = gymapi.Vec3(*self.cam_loc_p)
                xyz_angle_rad = [np.radians(a) for a in self.cam_loc_r]
                local_t.r = gymapi.Quat.from_euler_zyx(*xyz_angle_rad)
                self.gym.attach_camera_to_body(
                    cam_handle, env_ptr, rigid_body_hand_ind,
                    local_t, gymapi.FOLLOW_TRANSFORM
                )
                self.cams.append(cam_handle)
                # Camera tensor
                cam_tensor = self.gym.get_camera_image_gpu_tensor(self.sim, env_ptr, cam_handle, gymapi.IMAGE_COLOR)
                cam_tensor_th = gymtorch.wrap_tensor(cam_tensor)
                self.cam_tensors.append(cam_tensor_th)

        self.rigid_body_hand_ind = self.gym.find_actor_rigid_body_handle(env_ptr, franka_actor, "panda_hand")
        self.rigid_body_lfinger_ind = self.gym.find_actor_rigid_body_handle(env_ptr, franka_actor, "panda_leftfinger")
        self.rigid_body_rfinger_ind = self.gym.find_actor_rigid_body_handle(env_ptr, franka_actor, "panda_rightfinger")

        self.env_franka_ind = self.gym.get_actor_index(env_ptr, franka_actor, gymapi.DOMAIN_ENV)
        self.env_table_ind = self.gym.get_actor_index(env_ptr, table_actor, gymapi.DOMAIN_ENV)
        self.env_goal_ind = self.gym.get_actor_index(env_ptr, goal_actor, gymapi.DOMAIN_ENV)

        franka_rigid_body_names = self.gym.get_actor_rigid_body_names( env_ptr, franka_actor)
        franka_arm_body_names = [name for name in franka_rigid_body_names if "link" in name]

        self.rigid_body_arm_inds = torch.zeros(len(franka_arm_body_names), dtype=torch.long, device=self.device)
        for i, n in enumerate(franka_arm_body_names):
            self.rigid_body_arm_inds[i] = self.gym.find_actor_rigid_body_handle(env_ptr, franka_actor, n)

        self.init_grasp_pose()

    def init_grasp_pose(self):
        self.local_finger_grasp_pos = torch.zeros((self.num_envs, 3), dtype=torch.float, device=self.device)
        self.local_finger_grasp_pos[:, 2] = 0.045
        self.local_finger_grasp_rot = torch.zeros((self.num_envs, 4), dtype=torch.float, device=self.device)
        self.local_finger_grasp_rot[:, 3] = 1.0

        self.lfinger_grasp_pos = torch.zeros_like(self.local_finger_grasp_pos)
        self.lfinger_grasp_rot = torch.zeros_like(self.local_finger_grasp_rot)
        self.lfinger_grasp_rot[..., 3] = 1.0

        self.rfinger_grasp_pos = torch.zeros_like(self.local_finger_grasp_pos)
        self.rfinger_grasp_rot = torch.zeros_like(self.local_finger_grasp_rot)
        self.rfinger_grasp_rot[..., 3] = 1.0

    def compute_reward(self, actions):
        self.rew_buf[:], self.reset_buf[:], self.successes[:] = compute_franka_reward(
            self.reset_buf, self.progress_buf, self.successes, self.actions, self.to_target,
            self.goal_dist_reward_scale, self.goal_bonus_reward_scale, self.action_penalty_scale,
            self.contact_forces, self.rigid_body_arm_inds, self.max_episode_length
        )

    def reset(self, env_ids):

        # Franka multi env ids
        franka_multi_env_ids_int32 = self.global_indices[env_ids, self.env_franka_ind].flatten()

        # Reset franka dofs
        dof_pos_noise = torch.rand((len(env_ids), self.num_franka_dofs), device=self.device)
        dof_pos = tensor_clamp(
            self.franka_default_dof_pos.unsqueeze(0) + 0.25 * (dof_pos_noise - 0.5),
            self.franka_dof_lower_limits, self.franka_dof_upper_limits
        )
        self.franka_dof_pos[env_ids, :] = dof_pos
        self.franka_dof_vel[env_ids, :] = 0.0
        self.dof_targets[env_ids, :] = dof_pos

        self.gym.set_dof_position_target_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self.dof_targets),
            gymtorch.unwrap_tensor(franka_multi_env_ids_int32),
            len(franka_multi_env_ids_int32)
        )
        self.gym.set_dof_state_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self.dof_state),
            gymtorch.unwrap_tensor(franka_multi_env_ids_int32),
            len(franka_multi_env_ids_int32)
        )

        # Goal multi env ids
        goal_multi_env_ids_int32 = self.global_indices[env_ids, self.env_goal_ind].flatten()

        # Reset goal pos
        delta_x = torch_rand_float(
            -self.goal_pos_delta[0], self.goal_pos_delta[0],
            (len(env_ids), 1), device=self.device
        ).squeeze(dim=1)
        delta_y = torch_rand_float(
            -self.goal_pos_delta[1], self.goal_pos_delta[1],
            (len(env_ids), 1), device=self.device
        ).squeeze(dim=1)
        delta_z = torch_rand_float(
            -self.goal_pos_delta[2], self.goal_pos_delta[2],
            (len(env_ids), 1), device=self.device
        ).squeeze(dim=1)

        self.root_state_tensor[env_ids, self.env_goal_ind, 0] = self.goal_pos_init[0] + delta_x
        self.root_state_tensor[env_ids, self.env_goal_ind, 1] = self.goal_pos_init[1] + delta_y
        self.root_state_tensor[env_ids, self.env_goal_ind, 2] = self.goal_pos_init[2] + delta_z

        self.gym.set_actor_root_state_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self.root_state_tensor),
            gymtorch.unwrap_tensor(goal_multi_env_ids_int32),
            len(goal_multi_env_ids_int32)
        )

        self.progress_buf[env_ids] = 0
        self.reset_buf[env_ids] = 0
        self.successes[env_ids] = 0

    def pre_physics_step(self, actions):
        self.actions = actions.clone().to(self.device)
        targets = self.dof_targets \
            + self.franka_dof_speed_scales * self.dt * self.actions * self.action_scale
        self.dof_targets[:] = tensor_clamp(
            targets, self.franka_dof_lower_limits, self.franka_dof_upper_limits
        )
        self.gym.set_dof_position_target_tensor(self.sim, gymtorch.unwrap_tensor(self.dof_targets))

    def compute_task_state(self):
        self.lfinger_grasp_rot[:], self.lfinger_grasp_pos[:] = tf_combine(
            self.lfinger_rot, self.lfinger_pos,
            self.local_finger_grasp_rot, self.local_finger_grasp_pos,
        )
        self.rfinger_grasp_rot[:], self.rfinger_grasp_pos[:] = tf_combine(
            self.rfinger_rot, self.rfinger_pos,
            self.local_finger_grasp_rot, self.local_finger_grasp_pos,
        )
        # TODO: rename to avoid confusion
        self.hand_pos[:] = 0.5 * (self.lfinger_grasp_pos + self.rfinger_grasp_pos)
        self.to_target[:] = self.goal_pos - self.hand_pos

    def compute_robot_state(self):
        self.franka_dof_pos_scaled[:] = \
            (2.0 * (self.franka_dof_pos - self.franka_dof_lower_limits) /
                (self.franka_dof_upper_limits - self.franka_dof_lower_limits) - 1.0)
        self.franka_dof_vel_scaled[:] = self.franka_dof_vel * self.dof_vel_scale

        self.states_buf[:, :self.num_franka_dofs] = self.franka_dof_pos_scaled
        self.states_buf[:, self.num_franka_dofs:] = self.franka_dof_vel_scaled

    def compute_robot_obs(self):
        self.obs_buf[:, :self.num_franka_dofs] = self.franka_dof_pos_scaled
        self.obs_buf[:, self.num_franka_dofs:] = self.franka_dof_vel_scaled

    def compute_oracle_obs(self):
        self.obs_buf[:] = torch.cat([
            self.franka_dof_pos_scaled, self.franka_dof_vel_scaled,
            self.hand_pos, self.goal_pos, self.to_target
        ], dim=-1)

    def compute_pixel_obs(self):
        self.gym.render_all_camera_sensors(self.sim)
        self.gym.start_access_image_tensors(self.sim)
        for i in range(self.num_envs):
            crop_l = (self.cam_w - self.im_size) // 2 if self.cam_crop == "center" else 0
            crop_r = crop_l + self.im_size
            self.obs_buf[i] = self.cam_tensors[i][:, crop_l:crop_r, :3].permute(2, 0, 1).float() / 255.
            self.obs_buf[i] = (self.obs_buf[i] - self.im_mean) / self.im_std
        self.gym.end_access_image_tensors(self.sim)

    def post_physics_step(self):
        self.progress_buf += 1

        env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(env_ids) > 0:
            self.reset(env_ids)

        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)

        self.compute_task_state()
        self.compute_robot_state()
        self.compute_observations()
        self.compute_reward(self.actions)


@torch.jit.script
def compute_franka_reward(
    reset_buf: Tensor, progress_buf: Tensor, successes: Tensor, actions: Tensor, to_target: Tensor,
    goal_dist_reward_scale: float, goal_bonus_reward_scale: float, action_penalty_scale: float,
    contact_forces: Tensor, arm_inds: Tensor, max_episode_length: int
) -> Tuple[Tensor, Tensor, Tensor]:

    # Hand to goal distance
    d = torch.norm(to_target, p=2, dim=-1)
    goal_dist_reward = 1.0 / (0.04 + d)

    # Bonus if hand is near goal
    goal_bonus_reward = torch.zeros_like(goal_dist_reward)
    goal_bonus_reward = torch.where(d <= 0.04, goal_bonus_reward + 0.5, goal_bonus_reward)

    # Regularization on the actions
    action_penalty = torch.sum(actions ** 2, dim=-1)

    # Total reward
    rewards = goal_dist_reward_scale * goal_dist_reward \
        + goal_bonus_reward_scale * goal_bonus_reward \
        - action_penalty_scale * action_penalty

    # Goal reached
    s = torch.where(successes < 10.0, torch.zeros_like(successes), successes)
    successes = torch.where(d <= 0.04, torch.ones_like(successes) + successes, s)

    # Arm collision
    arm_collision = torch.any(torch.norm(contact_forces[:, arm_inds, :], dim=2) > 1.0, dim=1)
    reset_buf = torch.where(arm_collision, torch.ones_like(reset_buf), reset_buf)

    # Max episode length exceeded
    reset_buf = torch.where(progress_buf >= max_episode_length - 1, torch.ones_like(reset_buf), reset_buf)

    binary_s = torch.where(successes >= 10, torch.ones_like(successes), torch.zeros_like(successes))
    successes = torch.where(reset_buf > 0, binary_s, successes)

    return rewards, reset_buf, successes
