# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from rsl_rl.modules import ActorCritic
from rsl_rl.storage import RolloutStorage

class PPO:
    actor_critic: ActorCritic
    def __init__(self,
                 actor_critic,
                 num_learning_epochs=1,
                 num_mini_batches=1,
                 clip_param=0.2,
                 gamma=0.998,
                 lam=0.95,
                 value_loss_coef=1.0,
                 entropy_coef=0.0,
                 learning_rate=1e-3,
                 max_grad_norm=1.0,
                 use_clipped_value_loss=True,
                 schedule="fixed",
                 desired_kl=0.01,
                 device='cpu',
                 #new hjb parameters
                 hjb_coef: float = 0.1,
                 ):
        print("hjb coef:", hjb_coef)
        self.device = device

        self.desired_kl = desired_kl
        self.schedule = schedule
        self.learning_rate = learning_rate

        # PPO components
        self.actor_critic = actor_critic
        self.actor_critic.to(self.device)
        self.storage = None # initialized later
        self.optimizer = optim.Adam(self.actor_critic.parameters(), lr=learning_rate)
        self.transition = RolloutStorage.Transition()

        # PPO parameters
        self.clip_param = clip_param
        self.num_learning_epochs = num_learning_epochs
        self.num_mini_batches = num_mini_batches
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.gamma = gamma
        self.lam = lam
        self.max_grad_norm = max_grad_norm
        self.use_clipped_value_loss = use_clipped_value_loss
        self.dt = 0.02

        # HJB parameters
        self.hjb_coef = hjb_coef
        self.rho = -torch.log(torch.tensor(gamma)) / self.dt

    def init_storage(self, num_envs, num_transitions_per_env, actor_obs_shape, critic_obs_shape, action_shape):
        self.storage = RolloutStorage(num_envs, num_transitions_per_env, actor_obs_shape, critic_obs_shape, action_shape, self.device)

    def test_mode(self):
        self.actor_critic.test()
    
    def train_mode(self):
        self.actor_critic.train()

    def act(self, obs, critic_obs):
        if self.actor_critic.is_recurrent:
            self.transition.hidden_states = self.actor_critic.get_hidden_states()
        # Compute the actions and values
        self.transition.actions = self.actor_critic.act(obs).detach()
        self.transition.values = self.actor_critic.evaluate(critic_obs).detach()
        self.transition.actions_log_prob = self.actor_critic.get_actions_log_prob(self.transition.actions).detach()
        self.transition.action_mean = self.actor_critic.action_mean.detach()
        self.transition.action_sigma = self.actor_critic.action_std.detach()
        # need to record obs and critic_obs before env.step()
        self.transition.observations = obs
        self.transition.critic_observations = critic_obs
        return self.transition.actions
    
    def process_env_step(self, rewards, dones, infos):
        self.transition.rewards = rewards.clone()
        self.transition.dones = dones
        # Bootstrapping on time outs
        if 'time_outs' in infos:
            self.transition.rewards += self.gamma * torch.squeeze(self.transition.values * infos['time_outs'].unsqueeze(1).to(self.device), 1)

        # Record the transition
        self.storage.add_transitions(self.transition)
        self.transition.clear()
        self.actor_critic.reset(dones)
    
    def compute_returns(self, last_critic_obs):
        last_values= self.actor_critic.evaluate(last_critic_obs).detach()
        self.storage.compute_returns(last_values, self.gamma, self.lam)

    def compute_srb_dynamics(self, obs, approximated_dynamics):
        lin_vel_scales = 1
        ang_vel_scales = 1
        commands_scales = 1
        base_lin_vel = obs[:, :3] / lin_vel_scales
        base_ang_vel = obs[:, 3:6] / ang_vel_scales
        projected_gravity = obs[:, 6:9]
        commands = obs[:, 9:12] / commands_scales
        num_envs = obs.shape[0]
        actions = obs[:, -12:].clone().view(num_envs, 4, 3)

        # base weight = m_base + m_motor * 8
        base_weight = 1 # from config

        # compute base linear velocity
        assert base_lin_vel.shape == (num_envs, 3), "Base linear velocity shape mismatch"
        contact_indicator = torch.zeros((num_envs, 4, 3)) # inference from contact force in legged_gym 
        base_lin_vel_dot = -torch.cross(base_ang_vel, base_lin_vel, dim=1)
        total_contact_force = actions * contact_indicator
        total_contact_force = torch.sum(total_contact_force, dim=1) 
        base_lin_vel_dot += total_contact_force / base_weight

        # compute base angular velocity
        # inertia needs to update Ixx, Iyy, Izz according to leg configuration
        inertia = torch.zeros((num_envs, 3, 3), device=self.device) # from config
        quat = torch.zeros((num_envs, 4), device=self.device) # from legged_gym
        Rwb = self.quaternion_to_matrix(quat)
        Rwb_T = torch.transpose(Rwb, 1, 2)
        Rwb_T_expanded = Rwb_T.unsqueeze(1).repeat(1, 4, 1, 1) # (num_envs, 4, 3, 3)


    def quaternion_to_matrix(q):
        # q: (..., 4) with (w, x, y, z) ordering
        w, x, y, z = q.unbind(-1)
        B = q.shape[:-1]

        ww = w * w
        xx = x * x
        yy = y * y
        zz = z * z

        wx = w * x
        wy = w * y
        wz = w * z

        xy = x * y
        xz = x * z
        yz = y * z

        R = torch.empty(*B, 3, 3, device=q.device, dtype=q.dtype)
        R[..., 0, 0] = ww + xx - yy - zz
        R[..., 0, 1] = 2 * (xy - wz)
        R[..., 0, 2] = 2 * (xz + wy)

        R[..., 1, 0] = 2 * (xy + wz)
        R[..., 1, 1] = ww - xx + yy - zz
        R[..., 1, 2] = 2 * (yz - wx)

        R[..., 2, 0] = 2 * (xz - wy)
        R[..., 2, 1] = 2 * (yz + wx)
        R[..., 2, 2] = ww - xx - yy + zz
        return R


    def update(self):
        mean_value_loss = 0
        mean_surrogate_loss = 0
        mean_hjb_loss = 0.0

        if self.actor_critic.is_recurrent:
            generator = self.storage.reccurent_mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)
        else:
            generator = self.storage.mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)
        for (obs_batch, critic_obs_batch, actions_batch, target_values_batch, advantages_batch, returns_batch, old_actions_log_prob_batch, \
            old_mu_batch, old_sigma_batch, hid_states_batch, masks_batch, rewards_batch, dynamics_batch) in generator:

                self.actor_critic.act(obs_batch, masks=masks_batch, hidden_states=hid_states_batch[0])
                actions_log_prob_batch = self.actor_critic.get_actions_log_prob(actions_batch)
                value_batch = self.actor_critic.evaluate(critic_obs_batch, masks=masks_batch, hidden_states=hid_states_batch[1])
                mu_batch = self.actor_critic.action_mean
                sigma_batch = self.actor_critic.action_std
                entropy_batch = self.actor_critic.entropy

                # KL
                if self.desired_kl != None and self.schedule == 'adaptive':
                    with torch.inference_mode():
                        kl = torch.sum(
                            torch.log(sigma_batch / old_sigma_batch + 1.e-5) + (torch.square(old_sigma_batch) + torch.square(old_mu_batch - mu_batch)) / (2.0 * torch.square(sigma_batch)) - 0.5, axis=-1)
                        kl_mean = torch.mean(kl)

                        if kl_mean > self.desired_kl * 2.0:
                            self.learning_rate = max(1e-5, self.learning_rate / 1.5)
                        elif kl_mean < self.desired_kl / 2.0 and kl_mean > 0.0:
                            self.learning_rate = min(1e-2, self.learning_rate * 1.5)
                        
                        for param_group in self.optimizer.param_groups:
                            param_group['lr'] = self.learning_rate


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

                # HJB loss
                real_dynamic = self.compute_srb_dynamics(obs_batch, dynamics_batch)

                hjb_loss = torch.tensor(0.0, device=self.device)
                if self.hjb_coef > 0.0 and dynamics_batch is not None:
                    # Enable gradient through critic_obs_batch
                    critic_obs_batch.requires_grad_(True)
                    values_grad = self.actor_critic.evaluate(critic_obs_batch, masks=masks_batch,
                                                            hidden_states=hid_states_batch[1])
                    # ∂V/∂x
                    value_derivative = torch.autograd.grad(values_grad, critic_obs_batch,
                                                        grad_outputs=torch.ones_like(values_grad),
                                                        create_graph=True, retain_graph=True)[0]
                    critic_obs_batch.requires_grad_(False)
                    # V_x · f
                    B, obs_dim = value_derivative.shape
                    value_derivative_dot_f = torch.bmm(value_derivative.view(B, 1, obs_dim),
                                                    dynamics_batch.view(B, obs_dim, 1)).view(-1)
                    # ρ V vs r + V_x f
                    target = self.rho.to(self.device) * values_grad.view(-1)
                    rhs = value_derivative_dot_f + rewards_batch.view(-1)
                    hjb_loss = F.mse_loss(target, rhs)
                mean_hjb_loss += hjb_loss.item()
                
                

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
        mean_hjb_loss /= num_updates
        #print("HJB loss:", mean_hjb_loss)
        mean_surrogate_loss /= num_updates
        self.storage.clear()

        return mean_value_loss, mean_surrogate_loss, mean_hjb_loss
