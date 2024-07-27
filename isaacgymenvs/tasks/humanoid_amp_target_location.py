import numpy as np
import torch
from isaacgym import gymtorch
from isaacgym import gymapi
from .humanoid_amp_task import HumanoidAMPTask


class HumanoidAMPTargetLocation(HumanoidAMPTask):
    def __init__(self, cfg, rl_device, sim_device, graphics_device_id, headless, virtual_screen_capture, force_render):
        # target location will be sampled in a square
        self.tar_loc_size = cfg["env"]["tarLocSize"]
        super().__init__(cfg, rl_device, sim_device, graphics_device_id, headless, virtual_screen_capture, force_render)
        self._prev_root_pos = torch.zeros([self.num_envs, 3], device=self.device, dtype=torch.float)
        self._tar_loc = torch.zeros([self.num_envs, 2], device=self.device, dtype=torch.float)
        if (not self.headless):
            self._build_marker_state_tensors()
    
    def _create_envs(self, num_envs, spacing, num_per_row):
        # create flag asset to present target location
        if (not self.headless):
            self._flag_handles = []
            self._load_marker_asset()

        super()._create_envs(num_envs, spacing, num_per_row)
        return

    def _load_marker_asset(self):
        asset_root = "assets/mjcf/"
        asset_file = "heading_marker.urdf"

        asset_options = gymapi.AssetOptions()
        asset_options.angular_damping = 0.01
        asset_options.linear_damping = 0.01
        asset_options.max_angular_velocity = 100.0
        asset_options.density = 1.0
        asset_options.fix_base_link = True
        asset_options.default_dof_drive_mode = gymapi.DOF_MODE_NONE

        self._marker_asset = self.gym.load_asset(self.sim, asset_root, asset_file, asset_options)

        return
    
    def _build_env(self, env_id, env_ptr, humanoid_asset):
        super()._build_env(env_id, env_ptr, humanoid_asset)
        
        if (not self.headless):
            self._build_marker(env_id, env_ptr)

        return

    def _build_marker(self, env_id, env_ptr):
        col_group = env_id
        col_filter = 2
        segmentation_id = 0

        default_pose = gymapi.Transform()
        default_pose.p.x = 0.0 # origin position?
        default_pose.p.z = 0.2
        
        flag_handle = self.gym.create_actor(env_ptr, self._marker_asset, default_pose, "flag", col_group, col_filter, segmentation_id)
        self.gym.set_rigid_body_color(env_ptr, flag_handle, 0, gymapi.MESH_VISUAL, gymapi.Vec3(0.8, 0.0, 0.0))
        self._flag_handles.append(flag_handle)
        
        return
    
    def get_task_obs_size(self):
        obs_size = 0
        if (self._enable_task_obs):
            obs_size = 2 # x, y
        return obs_size

    def _build_marker_state_tensors(self):
        num_actors = self._root_states.shape[0] // self.num_envs

        self._marker_states = self._root_states.view(self.num_envs, num_actors, self._root_states.shape[-1])[..., -1, :]
        self._marker_pos = self._marker_states[..., :3]
        self._marker_rot = self._marker_states[..., 3:7]
        self._marker_actor_ids = self._humanoid_actor_ids + 1

        return
    
    def _reset_task(self, env_ids):
        n = len(env_ids)

        # sample target location
        target_location = torch.tensor(np.random.uniform(-self.tar_loc_size, self.tar_loc_size, size=(n, 2)), device=self.device,  dtype=torch.float32)

        self._tar_loc[env_ids] = target_location

        if (not self.headless):
            self._update_marker(env_ids)
    
    def pre_physics_step(self, actions):
        super().pre_physics_step(actions)
        self._prev_root_pos[:] = self._humanoid_root_states[..., 0:3]
        return

    def _update_task(self):
        return 
    
    def _draw_task(self):
        # don't need to draw task at every step
        return

    def _update_marker(self, env_ids=None):
        if (env_ids is None):
            self._marker_pos[..., :2] = self._tar_loc
            marker_ids = self._marker_actor_ids
        else:
            tar_loc = self._tar_loc[env_ids]
            tar_z = torch.tensor([0.2] * len(env_ids), device=self.device, dtype=torch.float).view(-1, 1)
            tar_pos = torch.cat([tar_loc, tar_z], dim=-1)
            self._marker_pos[env_ids] = tar_pos
            marker_ids = self._marker_actor_ids[env_ids]
        self.gym.set_actor_root_state_tensor_indexed(self.sim, gymtorch.unwrap_tensor(self._root_states),
                                                     gymtorch.unwrap_tensor(marker_ids), len(marker_ids))

    # =======post_physics_step=======
    def _compute_task_obs(self, env_ids=None):
        if (env_ids is None):
            root_states = self._humanoid_root_states
            tar_loc = self._tar_loc
        else:
            root_states = self._humanoid_root_states[env_ids]
            tar_loc = self._tar_loc[env_ids]
        
        obs = compute_location_observations(root_states, tar_loc)
        return obs

    def _compute_reward(self, actions):
        root_pos = self._humanoid_root_states[..., 0:3]
        root_rot = self._humanoid_root_states[..., 3:7]
        self.rew_buf[:] = compute_location_reward(root_pos, self._tar_loc, self._prev_root_pos, self.dt)
        print(self.rew_buf)
        return

@torch.jit.script
def compute_location_observations(root_states, tar_loc):
    # type: (Tensor, Tensor) -> Tensor

    obs = torch.cat([tar_loc], dim=-1)
    return obs

@torch.jit.script
def compute_location_reward(root_pos, tar_loc, prev_root_pos, dt):
    # type: (Tensor, Tensor, Tensor, float) -> Tensor

    dist_reward_w = 0.7
    vel_reward_w = 0.3
    tar_speed = 1.0
    # 2范数
    # distance term
    root_pos_xy = root_pos[..., :2]
    dist_norm = torch.norm(root_pos_xy - tar_loc, p=2, dim=-1)
    dist_reward = torch.exp(-0.5 * dist_norm)
    # TODO velocity term
    delta_root_pos = root_pos - prev_root_pos
    root_vel = delta_root_pos / dt
    tar_dir = tar_loc - root_pos_xy
    normalized_tar_dir = tar_dir / torch.norm(tar_dir, p=2, dim=-1, keepdim=True)
    # tar_vel = tar_speed * normalized_tar_dir
    # subtitute com velocity with root velocity
    tar_dir_speed = torch.sum(tar_dir * root_vel[..., :2], dim=-1)
    speed_error = tar_speed - tar_dir_speed
    speed_mask = speed_error <= 0
    speed_error[speed_mask] = 0
    vel_reward = torch.exp(-(speed_error ** 2))

    reward = dist_reward_w * dist_reward + vel_reward_w * vel_reward
    return reward
