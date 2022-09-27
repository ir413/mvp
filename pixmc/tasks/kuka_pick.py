#!/usr/bin/env python3

"""KukaPick task."""

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


class KukaPick(BaseTask):

    def __init__(self, cfg, sim_params, physics_engine, device_type, device_id, headless):
        self.cfg = cfg
        self.sim_params = sim_params
        self.physics_engine = physics_engine
        assert self.physics_engine == gymapi.SIM_PHYSX

        self.max_episode_length = self.cfg["env"]["episodeLength"]

        self.dof_vel_scale = self.cfg["env"]["dofVelocityScale"]
        self.action_scale = self.cfg["env"]["actionScale"]

        self.finger_dist_reward_scale = self.cfg["env"]["fingerDistRewardScale"]
        self.thumb_dist_reward_scale = self.cfg["env"]["thumbDistRewardScale"]
        self.lift_bonus_reward_scale = self.cfg["env"]["liftBonusRewardScale"]
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
            num_obs = 23 * 2
            self.compute_observations = self.compute_robot_obs
        elif self.obs_type == "oracle":
            num_obs = 77
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
        self.cfg["env"]["numStates"]= 23 * 2
        self.cfg["env"]["numActions"] = 23

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

        # Default dof pos
        self.kuka_default_dof_pos = to_torch([
            # Kuka arm
            0.0, -0.4, 0.0, -1.0, 0.0, 1.0, 0.0,
            # Index, middle, ring
            0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0,
            # Thumb
            0.0, 0.0, 0.0, 0.0
        ], device=self.device)

        # Dof state slices
        self.dof_state = gymtorch.wrap_tensor(dof_state_tensor)
        self.kuka_dof_state = self.dof_state.view(self.num_envs, self.num_kuka_dofs, 2)
        self.kuka_dof_pos = self.kuka_dof_state[..., 0]
        self.kuka_dof_vel = self.kuka_dof_state[..., 1]

        # (N, num_bodies, 13)
        self.rigid_body_states = gymtorch.wrap_tensor(rigid_body_tensor).view(self.num_envs, -1, 13)

        # (N, 3, 13)
        self.root_state_tensor = gymtorch.wrap_tensor(actor_root_state_tensor).view(self.num_envs, -1, 13)

        # (N, num_bodies, 3)
        self.contact_forces = gymtorch.wrap_tensor(net_contact_forces).view(self.num_envs, -1, 3)

        # Palm pos
        self.palm_pos = self.rigid_body_states[:, self.rigid_body_palm_ind, 0:3]

        # Finger pos
        self.index_pos = self.rigid_body_states[:, self.rigid_body_index_ind, 0:3]
        self.middle_pos = self.rigid_body_states[:, self.rigid_body_middle_ind, 0:3]
        self.ring_pos = self.rigid_body_states[:, self.rigid_body_ring_ind, 0:3]
        self.thumb_pos = self.rigid_body_states[:, self.rigid_body_thumb_ind, 0:3]

        # Finger rot
        self.index_rot = self.rigid_body_states[:, self.rigid_body_index_ind, 3:7]
        self.middle_rot = self.rigid_body_states[:, self.rigid_body_middle_ind, 3:7]
        self.ring_rot = self.rigid_body_states[:, self.rigid_body_ring_ind, 3:7]
        self.thumb_rot = self.rigid_body_states[:, self.rigid_body_thumb_ind, 3:7]

        # Object pos
        self.object_pos = self.root_state_tensor[:, self.env_object_ind, :3]

        # Dof targets
        self.dof_targets = torch.zeros((self.num_envs, self.num_kuka_dofs), dtype=torch.float, device=self.device)

        # Global inds
        self.global_indices = torch.arange(
            self.num_envs * (1 + 1 + 1), dtype=torch.int32, device=self.device
        ).view(self.num_envs, -1)

        # Kuka dof pos and vel scaled
        self.kuka_dof_pos_scaled = torch.zeros_like(self.kuka_dof_pos)
        self.kuka_dof_vel_scaled = torch.zeros_like(self.kuka_dof_vel)

        # Finger to object vecs
        self.index_to_object = torch.zeros((self.num_envs, 3), dtype=torch.float, device=self.device)
        self.middle_to_object = torch.zeros((self.num_envs, 3), dtype=torch.float, device=self.device)
        self.ring_to_object = torch.zeros((self.num_envs, 3), dtype=torch.float, device=self.device)
        self.thumb_to_object = torch.zeros((self.num_envs, 3), dtype=torch.float, device=self.device)

        # Goal height diff
        self.to_height = torch.zeros((self.num_envs, 1), dtype=torch.float, device=self.device)

        # Image mean and std
        if self.obs_type == "pixels":
            self.im_mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float, device=self.device).view(3, 1, 1)
            self.im_std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float, device=self.device).view(3, 1, 1)

        # Object pos randomization
        self.object_pos_init = torch.tensor(cfg["env"]["object_pos_init"], dtype=torch.float, device=self.device)
        self.object_pos_delta = torch.tensor(cfg["env"]["object_pos_delta"], dtype=torch.float, device=self.device)

        # Goal height
        self.goal_height = torch.tensor(cfg["env"]["goal_height"], dtype=torch.float, device=self.device)

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
        kuka_asset_file = self.cfg["env"]["asset"]["assetFileNameKuka"]

        # Load kuka asset
        asset_options = gymapi.AssetOptions()
        asset_options.flip_visual_attachments = False
        asset_options.fix_base_link = True
        asset_options.disable_gravity = True
        #asset_options.thickness = 0.001
        asset_options.default_dof_drive_mode = gymapi.DOF_MODE_POS
        #asset_options.use_mesh_materials = True
        kuka_asset = self.gym.load_asset(self.sim, asset_root, kuka_asset_file, asset_options)

        # Create table asset
        table_dims = gymapi.Vec3(0.6, 1.0, 0.4)
        asset_options = gymapi.AssetOptions()
        asset_options.fix_base_link = True
        table_asset = self.gym.create_box(self.sim, table_dims.x, table_dims.y, table_dims.z, asset_options)

        # Create object asset
        object_size = 0.06
        asset_options = gymapi.AssetOptions()
        object_asset = self.gym.create_box(self.sim, object_size, object_size, object_size, asset_options)

        self.num_kuka_bodies = self.gym.get_asset_rigid_body_count(kuka_asset)
        self.num_kuka_dofs = self.gym.get_asset_dof_count(kuka_asset)

        print("num kuka bodies: ", self.num_kuka_bodies)
        print("num kuka dofs: ", self.num_kuka_dofs)

        # PD gains
        kuka_dof_stiffness = [
            400, 400, 400, 400, 400, 400, 400,
            600, 600, 600, 1000,
            600, 600, 600, 1000,
            600, 600, 600, 1000,
            1000, 1000, 1000, 600
        ]
        kuka_dof_damping = [
            80, 80, 80, 80, 80, 80, 80,
            15, 20, 15, 15,
            15, 20, 15, 15,
            15, 20, 15, 15,
            30, 20, 20, 15
        ]

        # Maximum torque
        kuka_dof_effort = [
            320, 320, 176, 176, 110, 40, 40,
            0.7, 0.7, 0.7, 0.7,
            0.7, 0.7, 0.7, 0.7,
            0.7, 0.7, 0.7, 0.7,
            0.7, 0.7, 0.7, 0.7,
        ]

        # Set kuka dof props
        kuka_dof_props = self.gym.get_asset_dof_properties(kuka_asset)
        for i in range(self.num_kuka_dofs):
            kuka_dof_props["driveMode"][i] = gymapi.DOF_MODE_POS
            kuka_dof_props["stiffness"][i] = kuka_dof_stiffness[i]
            kuka_dof_props["damping"][i] = kuka_dof_damping[i]
            kuka_dof_props["effort"][i] = kuka_dof_effort[i]

        # Record kuka dof limits
        self.kuka_dof_lower_limits = torch.zeros(self.num_kuka_dofs, device=self.device, dtype=torch.float)
        self.kuka_dof_upper_limits = torch.zeros(self.num_kuka_dofs, device=self.device, dtype=torch.float)
        for i in range(self.num_kuka_dofs):
            self.kuka_dof_lower_limits[i] = kuka_dof_props["lower"][i].item()
            self.kuka_dof_upper_limits[i] = kuka_dof_props["upper"][i].item()

        self.kuka_dof_speed_scales = torch.ones_like(self.kuka_dof_lower_limits)

        kuka_start_pose = gymapi.Transform()
        kuka_start_pose.p = gymapi.Vec3(1.0, 0.0, 0.0)
        kuka_start_pose.r = gymapi.Quat(0.0, 0.0, 1.0, 0.0)

        table_start_pose = gymapi.Transform()
        table_start_pose.p = gymapi.Vec3(0.5, 0.0, 0.5 * table_dims.z)

        object_start_pose = gymapi.Transform()
        object_start_pose.p = gymapi.Vec3(0.5, 0.0, table_dims.z + 0.5 * object_size)
        self.object_z_init = object_start_pose.p.z

        # Compute aggregate size
        num_kuka_bodies = self.gym.get_asset_rigid_body_count(kuka_asset)
        num_kuka_shapes = self.gym.get_asset_rigid_shape_count(kuka_asset)
        num_table_bodies = self.gym.get_asset_rigid_body_count(table_asset)
        num_table_shapes = self.gym.get_asset_rigid_shape_count(table_asset)
        num_object_bodies = self.gym.get_asset_rigid_body_count(object_asset)
        num_object_shapes = self.gym.get_asset_rigid_shape_count(object_asset)
        max_agg_bodies = num_kuka_bodies + num_table_bodies + num_object_bodies
        max_agg_shapes = num_kuka_shapes + num_table_shapes + num_object_shapes

        self.kukas = []
        self.tables = []
        self.objects = []
        self.envs = []

        if self.obs_type == "pixels":
            self.cams = []
            self.cam_tensors = []

        for i in range(self.num_envs):
            # Create env instance
            env_ptr = self.gym.create_env(self.sim, lower, upper, num_per_row)

            # Aggregate actors
            self.gym.begin_aggregate(env_ptr, max_agg_bodies, max_agg_shapes, True)

            # Kuka actor
            kuka_actor = self.gym.create_actor(env_ptr, kuka_asset, kuka_start_pose, "kuka", i, 0, 0)
            self.gym.set_actor_dof_properties(env_ptr, kuka_actor, kuka_dof_props)
            self.gym.set_actor_scale(env_ptr, kuka_actor, 0.9)

            # Table actor
            table_actor = self.gym.create_actor(env_ptr, table_asset, table_start_pose, "table", i, 0, 0)

            # Object actor
            object_actor = self.gym.create_actor(env_ptr, object_asset, object_start_pose, "object", i, 0, 0)
            object_color = gymapi.Vec3(0.0, 0.447, 0.741)
            self.gym.set_rigid_body_color(env_ptr, object_actor, 0, gymapi.MESH_VISUAL_AND_COLLISION, object_color)

            self.gym.end_aggregate(env_ptr)

            self.envs.append(env_ptr)
            self.kukas.append(kuka_actor)
            self.tables.append(table_actor)
            self.objects.append(object_actor)

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
                rigid_body_mount_ind = self.gym.find_actor_rigid_body_handle(env_ptr, kuka_actor, "allegro_mount")
                local_t = gymapi.Transform()
                local_t.p = gymapi.Vec3(*self.cam_loc_p)
                xyz_angle_rad = [np.radians(a) for a in self.cam_loc_r]
                local_t.r = gymapi.Quat.from_euler_zyx(*xyz_angle_rad)
                self.gym.attach_camera_to_body(
                    cam_handle, env_ptr, rigid_body_mount_ind,
                    local_t, gymapi.FOLLOW_TRANSFORM
                )
                self.cams.append(cam_handle)
                # Camera tensor
                cam_tensor = self.gym.get_camera_image_gpu_tensor(self.sim, env_ptr, cam_handle, gymapi.IMAGE_COLOR)
                cam_tensor_th = gymtorch.wrap_tensor(cam_tensor)
                self.cam_tensors.append(cam_tensor_th)

        self.rigid_body_palm_ind = self.gym.find_actor_rigid_body_handle(env_ptr, kuka_actor, "link_palm")

        self.rigid_body_index_ind = self.gym.find_actor_rigid_body_handle(env_ptr, kuka_actor, "index_link_3")
        self.rigid_body_middle_ind = self.gym.find_actor_rigid_body_handle(env_ptr, kuka_actor, "middle_link_3")
        self.rigid_body_ring_ind = self.gym.find_actor_rigid_body_handle(env_ptr, kuka_actor, "ring_link_3")
        self.rigid_body_thumb_ind = self.gym.find_actor_rigid_body_handle(env_ptr, kuka_actor, "thumb_link_3")

        self.env_kuka_ind = self.gym.get_actor_index(env_ptr, kuka_actor, gymapi.DOMAIN_ENV)
        self.env_table_ind = self.gym.get_actor_index(env_ptr, table_actor, gymapi.DOMAIN_ENV)
        self.env_object_ind = self.gym.get_actor_index(env_ptr, object_actor, gymapi.DOMAIN_ENV)

        kuka_rigid_body_names = self.gym.get_actor_rigid_body_names( env_ptr, kuka_actor)
        kuka_arm_body_names = [name for name in kuka_rigid_body_names if "iiwa7" in name]

        self.rigid_body_arm_inds = torch.zeros(len(kuka_arm_body_names), dtype=torch.long, device=self.device)
        for i, n in enumerate(kuka_arm_body_names):
            self.rigid_body_arm_inds[i] = self.gym.find_actor_rigid_body_handle(env_ptr, kuka_actor, n)

        self.init_grasp_pose()

    def init_grasp_pose(self):
        self.local_finger_grasp_pos = torch.zeros((self.num_envs, 3), dtype=torch.float, device=self.device)
        self.local_finger_grasp_pos[:, 0] = 0.045
        self.local_finger_grasp_pos[:, 1] = 0.01
        self.local_finger_grasp_rot = torch.zeros((self.num_envs, 4), dtype=torch.float, device=self.device)
        self.local_finger_grasp_rot[:, 3] = 1.0

        self.index_grasp_pos = torch.zeros_like(self.local_finger_grasp_pos)
        self.index_grasp_rot = torch.zeros_like(self.local_finger_grasp_rot)
        self.index_grasp_rot[..., 3] = 1.0

        self.middle_grasp_pos = torch.zeros_like(self.local_finger_grasp_pos)
        self.middle_grasp_rot = torch.zeros_like(self.local_finger_grasp_rot)
        self.middle_grasp_rot[..., 3] = 1.0

        self.ring_grasp_pos = torch.zeros_like(self.local_finger_grasp_pos)
        self.ring_grasp_rot = torch.zeros_like(self.local_finger_grasp_rot)
        self.ring_grasp_rot[..., 3] = 1.0

        self.local_thumb_grasp_pos = torch.zeros((self.num_envs, 3), dtype=torch.float, device=self.device)
        self.local_thumb_grasp_pos[:, 0] = 0.06
        self.local_thumb_grasp_pos[:, 1] = 0.01
        self.local_thumb_grasp_rot = torch.zeros((self.num_envs, 4), dtype=torch.float, device=self.device)
        self.local_thumb_grasp_rot[:, 3] = 1.0

        self.thumb_grasp_pos = torch.zeros_like(self.local_thumb_grasp_pos)
        self.thumb_grasp_rot = torch.zeros_like(self.local_thumb_grasp_rot)
        self.thumb_grasp_rot[..., 3] = 1.0

    def compute_reward(self, actions):
        self.rew_buf[:], self.reset_buf[:], self.successes[:] = compute_kuka_reward(
            self.reset_buf, self.progress_buf, self.successes, self.actions,
            self.index_grasp_pos, self.middle_grasp_pos, self.ring_grasp_pos, self.thumb_grasp_pos,
            self.object_pos, self.to_height, self.object_z_init,
            self.finger_dist_reward_scale, self.thumb_dist_reward_scale, self.lift_bonus_reward_scale,
            self.goal_dist_reward_scale, self.goal_bonus_reward_scale, self.action_penalty_scale,
            self.contact_forces, self.rigid_body_arm_inds, self.max_episode_length
        )

    def reset(self, env_ids):

        # Kuka multi env ids
        kuka_multi_env_ids_int32 = self.global_indices[env_ids, self.env_kuka_ind].flatten()

        # Reset kuka dofs
        dof_pos_noise = torch.rand((len(env_ids), self.num_kuka_dofs), device=self.device)
        dof_pos = tensor_clamp(
            self.kuka_default_dof_pos.unsqueeze(0) + 0.25 * (dof_pos_noise - 0.5),
            self.kuka_dof_lower_limits, self.kuka_dof_upper_limits
        )
        self.kuka_dof_pos[env_ids, :] = dof_pos
        self.kuka_dof_vel[env_ids, :] = 0.0
        self.dof_targets[env_ids, :] = dof_pos

        self.gym.set_dof_position_target_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self.dof_targets),
            gymtorch.unwrap_tensor(kuka_multi_env_ids_int32),
            len(kuka_multi_env_ids_int32)
        )
        self.gym.set_dof_state_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self.dof_state),
            gymtorch.unwrap_tensor(kuka_multi_env_ids_int32),
            len(kuka_multi_env_ids_int32)
        )

        # Object multi env ids
        object_multi_env_ids_int32 = self.global_indices[env_ids, self.env_object_ind].flatten()

        # Reset object pos
        delta_x = torch_rand_float(
            -self.object_pos_delta[0], self.object_pos_delta[0],
            (len(env_ids), 1), device=self.device
        ).squeeze(dim=1)
        delta_y = torch_rand_float(
            -self.object_pos_delta[1], self.object_pos_delta[1],
            (len(env_ids), 1), device=self.device
        ).squeeze(dim=1)

        self.root_state_tensor[env_ids, self.env_object_ind, 0] = self.object_pos_init[0] + delta_x
        self.root_state_tensor[env_ids, self.env_object_ind, 1] = self.object_pos_init[1] + delta_y
        self.root_state_tensor[env_ids, self.env_object_ind, 2] = self.object_z_init
        self.root_state_tensor[env_ids, self.env_object_ind, 3:6] = 0.0
        self.root_state_tensor[env_ids, self.env_object_ind, 6] = 1.0
        self.root_state_tensor[env_ids, self.env_object_ind, 7:10] = 0.0
        self.root_state_tensor[env_ids, self.env_object_ind, 10:13] = 0.0

        self.gym.set_actor_root_state_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self.root_state_tensor),
            gymtorch.unwrap_tensor(object_multi_env_ids_int32),
            len(object_multi_env_ids_int32)
        )

        self.progress_buf[env_ids] = 0
        self.reset_buf[env_ids] = 0
        self.successes[env_ids] = 0

    def pre_physics_step(self, actions):
        self.actions = actions.clone().to(self.device)
        targets = self.dof_targets \
            + self.kuka_dof_speed_scales * self.dt * self.actions * self.action_scale
        self.dof_targets[:] = tensor_clamp(
            targets, self.kuka_dof_lower_limits, self.kuka_dof_upper_limits
        )
        self.gym.set_dof_position_target_tensor(self.sim, gymtorch.unwrap_tensor(self.dof_targets))

    def compute_task_state(self):
        self.index_grasp_rot[:], self.index_grasp_pos[:] = tf_combine(
            self.index_rot, self.index_pos,
            self.local_finger_grasp_rot, self.local_finger_grasp_pos
        )
        self.middle_grasp_rot[:], self.middle_grasp_pos[:] = tf_combine(
            self.middle_rot, self.middle_pos,
            self.local_finger_grasp_rot, self.local_finger_grasp_pos
        )
        self.ring_grasp_rot[:], self.ring_grasp_pos[:] = tf_combine(
            self.ring_rot, self.ring_pos,
            self.local_finger_grasp_rot, self.local_finger_grasp_pos
        )
        self.thumb_grasp_rot[:], self.thumb_grasp_pos[:] = tf_combine(
            self.thumb_rot, self.thumb_pos,
            self.local_thumb_grasp_rot, self.local_thumb_grasp_pos
        )
        self.index_to_object[:] = self.object_pos - self.index_grasp_pos
        self.middle_to_object[:] = self.object_pos - self.middle_grasp_pos
        self.ring_to_object[:] = self.object_pos - self.ring_grasp_pos
        self.thumb_to_object[:] = self.object_pos - self.thumb_grasp_pos
        self.to_height[:] = self.goal_height - self.object_pos[:, 2].unsqueeze(1)

    def compute_robot_state(self):
        self.kuka_dof_pos_scaled[:] = \
            (2.0 * (self.kuka_dof_pos - self.kuka_dof_lower_limits) /
                (self.kuka_dof_upper_limits - self.kuka_dof_lower_limits) - 1.0)
        self.kuka_dof_vel_scaled[:] = self.kuka_dof_vel * self.dof_vel_scale

        self.states_buf[:, :self.num_kuka_dofs] = self.kuka_dof_pos_scaled
        self.states_buf[:, self.num_kuka_dofs:] = self.kuka_dof_vel_scaled

    def compute_robot_obs(self):
        self.obs_buf[:, :self.num_kuka_dofs] = self.kuka_dof_pos_scaled
        self.obs_buf[:, self.num_kuka_dofs:] = self.kuka_dof_vel_scaled

    def compute_oracle_obs(self):
        self.obs_buf[:] = torch.cat([
            self.kuka_dof_pos_scaled, self.kuka_dof_vel_scaled,
            self.palm_pos,
            self.index_grasp_pos, self.middle_grasp_pos, self.ring_grasp_pos, self.thumb_grasp_pos,
            self.object_pos,
            self.index_to_object, self.middle_to_object, self.ring_to_object, self.thumb_to_object,
            self.to_height
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
def compute_kuka_reward(
    reset_buf: Tensor, progress_buf: Tensor, successes: Tensor, actions: Tensor,
    index_grasp_pos: Tensor, middle_grasp_pos: Tensor, ring_grasp_pos: Tensor, thumb_grasp_pos: Tensor,
    object_pos: Tensor, to_height: Tensor, object_z_init: float,
    finger_dist_reward_scale: float, thumb_dist_reward_scale: float, lift_bonus_reward_scale: float,
    goal_dist_reward_scale: float, goal_bonus_reward_scale: float, action_penalty_scale: float,
    contact_forces: Tensor, arm_inds: Tensor, max_episode_length: int
) -> Tuple[Tensor, Tensor, Tensor]:

    # Index to object distance
    io_d = torch.norm(object_pos - index_grasp_pos, p=2, dim=-1)
    io_d = torch.clamp(io_d, min=0.03)
    io_dist_reward = 1.0 / (0.04 + io_d)

    # Middle to object distance
    mo_d = torch.norm(object_pos - middle_grasp_pos, p=2, dim=-1)
    mo_d = torch.clamp(mo_d, min=0.03)
    mo_dist_reward = 1.0 / (0.04 + mo_d)

    # Ring to object distance
    ro_d = torch.norm(object_pos - ring_grasp_pos, p=2, dim=-1)
    ro_d = torch.clamp(ro_d, min=0.03)
    ro_dist_reward = 1.0 / (0.04 + ro_d)

    # Thumb to object distance
    to_d = torch.norm(object_pos - thumb_grasp_pos, p=2, dim=-1)
    to_d = torch.clamp(to_d, min=0.03)
    to_dist_reward = 1.0 / (0.04 + to_d)

    # Object above table
    object_above = (object_pos[:, 2] - object_z_init) > 0.02

    # Above the table bonus
    lift_bonus_reward = torch.zeros_like(io_dist_reward)
    lift_bonus_reward = torch.where(object_above, lift_bonus_reward + 0.5, lift_bonus_reward)

    # Object to goal height distance
    og_d = torch.norm(to_height, p=2, dim=-1)
    og_dist_reward = torch.zeros_like(io_dist_reward)
    og_dist_reward = torch.where(object_above, 1.0 / (0.04 + og_d), og_dist_reward)

    # Bonus if object is near goal height
    og_bonus_reward = torch.zeros_like(og_dist_reward)
    og_bonus_reward = torch.where(og_d <= 0.06, og_bonus_reward + 0.5, og_bonus_reward)

    # Regularization on the actions
    action_penalty = torch.sum(actions ** 2, dim=-1)

    # Total reward
    rewards = finger_dist_reward_scale * io_dist_reward \
        + finger_dist_reward_scale * mo_dist_reward \
        + finger_dist_reward_scale * ro_dist_reward \
        + thumb_dist_reward_scale * to_dist_reward \
        + lift_bonus_reward_scale * lift_bonus_reward \
        + goal_dist_reward_scale * og_dist_reward \
        + goal_bonus_reward_scale * og_bonus_reward \
        - action_penalty_scale * action_penalty

    # Goal reached
    goal_height = 0.8 - 0.4  # absolute goal height - table height
    s = torch.where(successes < 10.0, torch.zeros_like(successes), successes)
    successes = torch.where(og_d <= goal_height * 0.25, torch.ones_like(successes) + successes, s)

    # Object below table height
    object_below = (object_z_init - object_pos[:, 2]) > 0.06
    reset_buf = torch.where(object_below, torch.ones_like(reset_buf), reset_buf)

    # Arm collision
    arm_collision = torch.any(torch.norm(contact_forces[:, arm_inds, :], dim=2) > 1.0, dim=1)
    reset_buf = torch.where(arm_collision, torch.ones_like(reset_buf), reset_buf)

    # Max episode length exceeded
    reset_buf = torch.where(progress_buf >= max_episode_length - 1, torch.ones_like(reset_buf), reset_buf)

    binary_s = torch.where(successes >= 10, torch.ones_like(successes), torch.zeros_like(successes))
    successes = torch.where(reset_buf > 0, binary_s, successes)

    return rewards, reset_buf, successes
