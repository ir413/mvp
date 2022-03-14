#!/usr/bin/env python3

"""PPO."""

import os
import math
import time

from gym.spaces import Space

import statistics
from collections import deque

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from mvp.ppo import RolloutStorage


class PPO:

    def __init__(
        self,
        vec_env,
        actor_critic_class,
        num_transitions_per_env,
        num_learning_epochs,
        num_mini_batches,
        clip_param=0.2,
        gamma=0.998,
        lam=0.95,
        init_noise_std=1.0,
        value_loss_coef=1.0,
        entropy_coef=0.0,
        learning_rate=1e-3,
        max_grad_norm=0.5,
        use_clipped_value_loss=True,
        schedule="fixed",
        encoder_cfg=None,
        policy_cfg=None,
        device="cpu",
        sampler="sequential",
        log_dir="run",
        is_testing=False,
        print_log=True,
        apply_reset=False,
        num_gpus=1
    ):

        if not isinstance(vec_env.observation_space, Space):
            raise TypeError("vec_env.observation_space must be a gym Space")
        if not isinstance(vec_env.state_space, Space):
            raise TypeError("vec_env.state_space must be a gym Space")
        if not isinstance(vec_env.action_space, Space):
            raise TypeError("vec_env.action_space must be a gym Space")

        assert not (num_gpus > 1) or not is_testing

        self.observation_space = vec_env.observation_space
        self.action_space = vec_env.action_space
        self.state_space = vec_env.state_space

        self.device = device
        self.num_gpus = num_gpus

        self.schedule = schedule
        self.step_size_init = learning_rate
        self.step_size = learning_rate

        # PPO components
        self.vec_env = vec_env
        self.actor_critic = actor_critic_class(
            self.observation_space.shape,
            self.state_space.shape,
            self.action_space.shape,
            init_noise_std,
            encoder_cfg,
            policy_cfg
        )
        self.actor_critic.to(self.device)

        # Set up DDP for multi-gpu training
        if self.num_gpus > 1:
            self.actor_critic = torch.nn.parallel.DistributedDataParallel(
                module=self.actor_critic,
                device_ids=[self.device],
            )
            self.actor_critic.act = self.actor_critic.module.act
            self.actor_critic.log_std = self.actor_critic.module.log_std

        self.storage = RolloutStorage(
            self.vec_env.num_envs, num_transitions_per_env, self.observation_space.shape,
            self.state_space.shape, self.action_space.shape, self.device, sampler
        )
        self.optimizer = optim.Adam(self.actor_critic.parameters(), lr=learning_rate)

        # PPO parameters
        self.clip_param = clip_param
        self.num_learning_epochs = num_learning_epochs
        self.num_mini_batches = num_mini_batches
        self.num_transitions_per_env = num_transitions_per_env
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.gamma = gamma
        self.lam = lam
        self.max_grad_norm = max_grad_norm
        self.use_clipped_value_loss = use_clipped_value_loss

        # Logging fields
        self.log_dir = log_dir
        self.tot_timesteps = 0
        self.tot_time = 0
        self.is_testing = is_testing
        self.current_learning_iteration = 0

        # Single-gpu logging
        if self.num_gpus == 1:
            self.print_log = print_log
            self.writer = SummaryWriter(log_dir=self.log_dir, flush_secs=10) if not is_testing else None

        # Multi-gpu logging
        if self.num_gpus > 1:
            if torch.distributed.get_rank() == 0:
                self.print_log = print_log
                self.writer = SummaryWriter(log_dir=self.log_dir, flush_secs=10)
            else:
                self.print_log = False
                self.writer = None

        self.apply_reset = apply_reset

    def test(self, path):
        state_dict = torch.load(path)
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        self.actor_critic.load_state_dict(state_dict)
        self.actor_critic.eval()

    def load(self, path):
        self.actor_critic.load_state_dict(torch.load(path, map_location="cpu"))
        self.current_learning_iteration = int(path.split("_")[-1].split(".")[0])
        self.actor_critic.train()

    def save(self, path):
        torch.save(self.actor_critic.state_dict(), path)

    def run(self, num_learning_iterations, log_interval=1):
        current_obs = self.vec_env.reset()
        current_states = self.vec_env.get_state()

        if self.is_testing:
            maxlen = 200
            cur_reward_sum = torch.zeros(self.vec_env.num_envs, dtype=torch.float, device=self.device)
            cur_episode_length = torch.zeros(self.vec_env.num_envs, dtype=torch.float, device=self.device)

            reward_sum = []
            episode_length = []
            successes = []

            while len(reward_sum) <= maxlen:
                with torch.no_grad():
                    if self.apply_reset:
                        current_obs = self.vec_env.reset()
                        current_states = self.vec_env.get_state()
                    # Compute the action
                    actions = self.actor_critic.act_inference(current_obs, current_states)
                    # Step the vec_environment
                    next_obs, rews, dones, infos = self.vec_env.step(actions)
                    next_states = self.vec_env.get_state()
                    current_obs.copy_(next_obs)
                    current_states.copy_(next_states)

                    cur_reward_sum[:] += rews
                    cur_episode_length[:] += 1

                    new_ids = (dones > 0).nonzero(as_tuple=False)
                    reward_sum.extend(cur_reward_sum[new_ids][:, 0].cpu().numpy().tolist())
                    episode_length.extend(cur_episode_length[new_ids][:, 0].cpu().numpy().tolist())
                    successes.extend(infos["successes"][new_ids][:, 0].cpu().numpy().tolist())
                    cur_reward_sum[new_ids] = 0
                    cur_episode_length[new_ids] = 0

                    if len(new_ids) > 0:
                        print("-" * 80)
                        print("Num episodes: {}".format(len(reward_sum)))
                        print("Mean return: {:.2f}".format(statistics.mean(reward_sum)))
                        print("Mean ep len: {:.2f}".format(statistics.mean(episode_length)))
                        print("Mean success: {:.2f}".format(statistics.mean(successes) * 100))

        else:
            maxlen = 200
            rewbuffer = deque(maxlen=maxlen)
            lenbuffer = deque(maxlen=maxlen)
            successbuffer = deque(maxlen=maxlen)
            cur_reward_sum = torch.zeros(self.vec_env.num_envs, dtype=torch.float, device=self.device)
            cur_episode_length = torch.zeros(self.vec_env.num_envs, dtype=torch.float, device=self.device)

            reward_sum = []
            episode_length = []
            successes = []

            for it in range(self.current_learning_iteration, num_learning_iterations):
                start = time.time()

                # Rollout
                for _ in range(self.num_transitions_per_env):
                    if self.apply_reset:
                        current_obs = self.vec_env.reset()
                        current_states = self.vec_env.get_state()
                    # Compute the action
                    actions, actions_log_prob, values, mu, sigma, current_obs_feats = \
                        self.actor_critic.act(current_obs, current_states)
                    # Step the vec_environment
                    next_obs, rews, dones, infos = self.vec_env.step(actions)
                    next_states = self.vec_env.get_state()
                    # Record the transition
                    obs_in = current_obs_feats if current_obs_feats is not None else current_obs
                    self.storage.add_transitions(
                        obs_in, current_states, actions, rews, dones, values, actions_log_prob, mu, sigma
                    )
                    current_obs.copy_(next_obs)
                    current_states.copy_(next_states)

                    if self.print_log:
                        cur_reward_sum[:] += rews
                        cur_episode_length[:] += 1

                        new_ids = (dones > 0).nonzero(as_tuple=False)
                        reward_sum.extend(cur_reward_sum[new_ids][:, 0].cpu().numpy().tolist())
                        episode_length.extend(cur_episode_length[new_ids][:, 0].cpu().numpy().tolist())
                        successes.extend(infos["successes"][new_ids][:, 0].cpu().numpy().tolist())
                        cur_reward_sum[new_ids] = 0
                        cur_episode_length[new_ids] = 0

                if self.print_log:
                    rewbuffer.extend(reward_sum)
                    lenbuffer.extend(episode_length)
                    successbuffer.extend(successes)

                _, _, last_values, _, _, _ = self.actor_critic.act(current_obs, current_states)
                stop = time.time()
                collection_time = stop - start

                mean_trajectory_length, mean_reward = self.storage.get_statistics()

                # Learning step
                start = stop
                self.storage.compute_returns(last_values, self.gamma, self.lam)
                mean_value_loss, mean_surrogate_loss = self.update(it, num_learning_iterations)
                self.storage.clear()
                stop = time.time()
                learn_time = stop - start

                if self.print_log:
                    rewbuffer_len = len(rewbuffer)
                    rewbuffer_mean = statistics.mean(rewbuffer) if rewbuffer_len > 0 else 0
                    lenbuffer_mean = statistics.mean(lenbuffer) if rewbuffer_len > 0 else 0
                    successbuffer_mean = statistics.mean(successbuffer) if rewbuffer_len > 0 else 0
                    self.log(
                        it, num_learning_iterations, collection_time, learn_time, mean_value_loss, mean_surrogate_loss,
                        mean_trajectory_length, mean_reward, rewbuffer_mean, lenbuffer_mean, successbuffer_mean
                    )
                if self.print_log and it % log_interval == 0:
                    self.save(os.path.join(self.log_dir, 'model_{}.pt'.format(it)))

                # Use an explicit sync point since we are not syncing stats yet
                if self.num_gpus > 1:
                    torch.distributed.barrier()

            if self.print_log:
                self.save(os.path.join(self.log_dir, 'model_{}.pt'.format(num_learning_iterations)))
                self.writer.close()

    def log(
        self, it, num_learning_iterations, collection_time, learn_time, mean_value_loss, mean_surrogate_loss,
        mean_trajectory_length, mean_reward, rewbuffer_mean, lenbuffer_mean, successbuffer_mean, width=80, pad=35
    ):

        num_steps_per_iter = self.num_transitions_per_env * self.vec_env.num_envs * self.num_gpus
        self.tot_timesteps += num_steps_per_iter

        self.tot_time += collection_time + learn_time
        iteration_time = collection_time + learn_time

        mean_std = self.actor_critic.log_std.exp().mean()
        mean_success_rate = successbuffer_mean * 100

        self.writer.add_scalar('Policy-iter/value_function', mean_value_loss, it)
        self.writer.add_scalar('Policy-simstep/value_function', mean_value_loss, self.tot_timesteps)
        self.writer.add_scalar('Policy-iter/surrogate', mean_surrogate_loss, it)
        self.writer.add_scalar('Policy-simstep/surrogate', mean_surrogate_loss, self.tot_timesteps)

        self.writer.add_scalar('Policy-iter/mean_noise_std', mean_std.item(), it)
        self.writer.add_scalar('Policy-simstep/mean_noise_std', mean_std.item(), self.tot_timesteps)
        self.writer.add_scalar('Policy-iter/lr', self.step_size, it)
        self.writer.add_scalar('Policy-simstep/lr', self.step_size, self.tot_timesteps)

        self.writer.add_scalar('Train-iter/mean_traj_reward', rewbuffer_mean, it)
        self.writer.add_scalar('Train-iter/mean_traj_length', lenbuffer_mean, it)
        self.writer.add_scalar('Train-iter/mean_success_rate', mean_success_rate, it)
        self.writer.add_scalar("Train-simstep/mean_traj_reward", rewbuffer_mean, self.tot_timesteps)
        self.writer.add_scalar("Train-simstep/mean_traj_length", lenbuffer_mean, self.tot_timesteps)
        self.writer.add_scalar("Train-simstep/mean_success_rate", mean_success_rate, self.tot_timesteps)

        self.writer.add_scalar('Train-iter/mean_step_reward', mean_reward, it)
        self.writer.add_scalar('Train-simstep/mean_step_reward', mean_reward, self.tot_timesteps)

        fps = int(num_steps_per_iter / (collection_time + learn_time))

        str = f" \033[1m Learning iteration {it}/{num_learning_iterations} \033[0m "

        log_string = (f"""{'#' * width}\n"""
                      f"""{str.center(width, ' ')}\n\n"""
                      f"""{'Computation:':>{pad}} {fps:.0f} steps/s (collection: {
                          collection_time:.3f}s, learning {learn_time:.3f}s)\n"""
                      f"""{'Value function loss:':>{pad}} {mean_value_loss:.4f}\n"""
                      f"""{'Surrogate loss:':>{pad}} {mean_surrogate_loss:.4f}\n"""
                      f"""{'Mean action noise std:':>{pad}} {mean_std.item():.2f}\n"""
                      f"""{'Mean reward:':>{pad}} {rewbuffer_mean:.2f}\n"""
                      f"""{'Mean episode length:':>{pad}} {lenbuffer_mean:.2f}\n"""
                      f"""{'Mean success rate:':>{pad}} {mean_success_rate:.2f}\n"""
                      f"""{'Mean reward/step:':>{pad}} {mean_reward:.2f}\n"""
                      f"""{'Mean episode length/episode:':>{pad}} {mean_trajectory_length:.2f}\n""")

        log_string += (f"""{'-' * width}\n"""
                       f"""{'Total timesteps:':>{pad}} {self.tot_timesteps}\n"""
                       f"""{'Iteration time:':>{pad}} {iteration_time:.2f}s\n"""
                       f"""{'Total time:':>{pad}} {self.tot_time:.2f}s\n"""
                       f"""{'ETA:':>{pad}} {self.tot_time / (it + 1) * (
                               num_learning_iterations - it):.1f}s\n""")
        print(log_string)

    def update(self, cur_iter, max_iter):
        mean_value_loss = 0
        mean_surrogate_loss = 0

        batch = self.storage.mini_batch_generator(self.num_mini_batches)
        for epoch in range(self.num_learning_epochs):

            if self.schedule == "cos":
                self.step_size = self.adjust_learning_rate_cos(
                    self.optimizer, epoch, self.num_learning_epochs, cur_iter, max_iter
                )

            for indices in batch:
                obs_batch = self.storage.observations.view(-1, *self.storage.observations.size()[2:])[indices]
                if self.vec_env.num_states > 0:
                    states_batch = self.storage.states.view(-1, *self.storage.states.size()[2:])[indices]
                else:
                    states_batch = None
                actions_batch = self.storage.actions.view(-1, self.storage.actions.size(-1))[indices]
                target_values_batch = self.storage.values.view(-1, 1)[indices]
                returns_batch = self.storage.returns.view(-1, 1)[indices]
                old_actions_log_prob_batch = self.storage.actions_log_prob.view(-1, 1)[indices]
                advantages_batch = self.storage.advantages.view(-1, 1)[indices]
                old_mu_batch = self.storage.mu.view(-1, self.storage.actions.size(-1))[indices]
                old_sigma_batch = self.storage.sigma.view(-1, self.storage.actions.size(-1))[indices]

                actions_log_prob_batch, entropy_batch, value_batch, mu_batch, sigma_batch = \
                    self.actor_critic(obs_batch, states_batch, actions_batch)

                # Surrogate loss
                ratio = torch.exp(actions_log_prob_batch - torch.squeeze(old_actions_log_prob_batch))
                surrogate = -torch.squeeze(advantages_batch) * ratio
                surrogate_clipped = -torch.squeeze(advantages_batch) * torch.clamp(ratio, 1.0 - self.clip_param,
                                                                                   1.0 + self.clip_param)
                surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()

                # Value function loss
                if self.use_clipped_value_loss:
                    value_clipped = target_values_batch + (value_batch - target_values_batch).clamp(-self.clip_param,
                                                                                                    self.clip_param)
                    value_losses = (value_batch - returns_batch).pow(2)
                    value_losses_clipped = (value_clipped - returns_batch).pow(2)
                    value_loss = torch.max(value_losses, value_losses_clipped).mean()
                else:
                    value_loss = (returns_batch - value_batch).pow(2).mean()

                loss = surrogate_loss + self.value_loss_coef * value_loss - self.entropy_coef * entropy_batch.mean()

                # Gradient step
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.max_grad_norm)
                self.optimizer.step()

                mean_value_loss += value_loss.item()
                mean_surrogate_loss += surrogate_loss.item()

        num_updates = self.num_learning_epochs * self.num_mini_batches
        mean_value_loss /= num_updates
        mean_surrogate_loss /= num_updates

        return mean_value_loss, mean_surrogate_loss

    def adjust_learning_rate_cos(self, optimizer, epoch, max_epoch, iter, max_iter):
        lr = self.step_size_init * 0.5 * (1. + math.cos(math.pi * (iter + epoch / max_epoch) / max_iter))
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        return lr
