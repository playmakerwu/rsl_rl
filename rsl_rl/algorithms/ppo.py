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
        self.actor_optimizer = optim.Adam(self.actor_critic.actor_parameters(), lr=learning_rate)
        self.critic_optimizer = optim.Adam(self.actor_critic.critic_parameters(), lr=learning_rate)
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
        self.rho = -torch.log(torch.tensor(gamma))

        # observation
        self.prev_obs = None
        self.prev_dones = None

    def init_storage(self, num_envs, num_transitions_per_env, actor_obs_shape, critic_obs_shape, action_shape, srb_shape):
        self.storage = RolloutStorage(num_envs, num_transitions_per_env, actor_obs_shape, critic_obs_shape, action_shape, srb_shape, self.device)

    def test_mode(self):
        self.actor_critic.test()
    
    def train_mode(self):
        self.actor_critic.train()

    def act(self, obs, critic_obs):
        if self.actor_critic.is_recurrent:
            self.transition.hidden_states = self.actor_critic.get_hidden_states()
        # Compute the actions and values

        if self.prev_obs is not None:
            fd = (obs - self.prev_obs) / self.dt           
            if self.prev_dones is not None:
                fd[self.prev_dones] = 0.0
            self.transition.dynamics = fd.detach()

        self.transition.actions = self.actor_critic.act(obs).detach()
        self.transition.values = self.actor_critic.evaluate(critic_obs).detach()
        self.transition.actions_log_prob = self.actor_critic.get_actions_log_prob(self.transition.actions).detach()
        self.transition.action_mean = self.actor_critic.action_mean.detach()
        self.transition.action_sigma = self.actor_critic.action_std.detach()
        # need to record obs and critic_obs before env.step()
        self.transition.observations = obs
        self.transition.critic_observations = critic_obs
        self.prev_obs = obs.detach()
        return self.transition.actions
    
    def process_env_step(self, rewards, dones, infos, srb_dynamics=None):
        self.transition.rewards = rewards.clone()
        self.transition.dones = dones
        # Bootstrapping on time outs
        if 'time_outs' in infos:
            self.transition.rewards += self.gamma * torch.squeeze(self.transition.values * infos['time_outs'].unsqueeze(1).to(self.device), 1)
        if srb_dynamics is not None:
            self.transition.srb_dynamics = srb_dynamics.clone()

        # Record the transition
        self.prev_dones = dones.clone()
        self.storage.add_transitions(self.transition)
        self.transition.clear()
        self.actor_critic.reset(dones)

    
    def compute_returns(self, last_critic_obs):
        last_values= self.actor_critic.evaluate(last_critic_obs).detach()
        self.storage.compute_returns(last_values, self.gamma, self.lam)

    def update(self):
        mean_value_loss = 0
        mean_surrogate_loss = 0
        mean_hjb_loss = 0.0
        
        E_actor = self.num_learning_epochs
        E_critic = self.num_learning_epochs * 2

        if self.actor_critic.is_recurrent:
            generator = self.storage.reccurent_mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)
        else:
            generator_actor = self.storage.mini_batch_generator(self.num_mini_batches, E_actor)
            generator_critic = self.storage.mini_batch_generator(self.num_mini_batches, E_critic)
            
        '''
        for (obs_batch, critic_obs_batch, actions_batch, target_values_batch, advantages_batch, returns_batch, old_actions_log_prob_batch, \
            old_mu_batch, old_sigma_batch, hid_states_batch, masks_batch, rewards_batch, dynamics_batch, srb_dynamics_batch) in generator:
            # Normalize advantages
                #print("fd_rsl", srb_dynamics_batch[5][:6])
                #print("dynamics_batch:", dynamics_batch[5][:9])
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

                hjb_loss = torch.tensor(0.0, device=self.device)
                if self.hjb_coef >= 0.0 and srb_dynamics_batch is not None:
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
                                                    srb_dynamics_batch.view(B, obs_dim, 1)).view(-1)
                    # ρ V vs r + V_x f
                    target = self.rho.to(self.device) * values_grad.view(-1)
                    rhs = value_derivative_dot_f + rewards_batch.view(-1)
                    hjb_loss = F.mse_loss(target, rhs)
                mean_hjb_loss += hjb_loss.item()
                
                

                loss = surrogate_loss + self.value_loss_coef * value_loss - self.entropy_coef * entropy_batch.mean() + self.hjb_coef * hjb_loss

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
        '''
        actor_updates = 0
        for (obs_batch, critic_obs_batch, actions_batch, target_values_batch, advantages_batch, returns_batch,
            old_actions_log_prob_batch, old_mu_batch, old_sigma_batch, hid_states_batch, masks_batch,
            rewards_batch, dynamics_batch, srb_dynamics_batch) in generator_actor:

            # 前向（只需策略相关量）
            self.actor_critic.act(obs_batch, masks=masks_batch, hidden_states=hid_states_batch[0])
            actions_log_prob_batch = self.actor_critic.get_actions_log_prob(actions_batch)
            mu_batch   = self.actor_critic.action_mean
            sigma_batch= self.actor_critic.action_std
            entropy_batch = self.actor_critic.entropy

            # 自适应 KL 学习率（若启用，建议只调 actor 的 lr）
            if self.desired_kl is not None and self.schedule == 'adaptive':
                with torch.inference_mode():
                    # 数值更稳的写法可用 clamp（略）
                    kl = torch.sum(
                        torch.log(sigma_batch / (old_sigma_batch + 1e-8)) +
                        (old_sigma_batch.pow(2) + (old_mu_batch - mu_batch).pow(2)) / (2.0 * sigma_batch.pow(2)) - 0.5,
                        dim=-1
                    )
                    kl_mean = kl.mean()
                # 按你的逻辑调 lr（只动 actor_optimizer）
                if kl_mean > self.desired_kl * 2.0:
                    new_lr = max(1e-5, self.actor_optimizer.param_groups[0]['lr'] / 1.5)
                elif kl_mean < self.desired_kl / 2.0 and kl_mean > 0.0:
                    new_lr = min(1e-2, self.actor_optimizer.param_groups[0]['lr'] * 1.5)
                else:
                    new_lr = self.actor_optimizer.param_groups[0]['lr']
                for g in self.actor_optimizer.param_groups:
                    g['lr'] = new_lr

            # PPO 策略损失（裁剪）
            ratio = torch.exp(actions_log_prob_batch - old_actions_log_prob_batch.squeeze(-1))
            # 若 advantages 未标准化，建议：(advantages_batch - mean) / (std + 1e-8)
            surrogate = -advantages_batch.squeeze(-1) * ratio
            surrogate_clipped = -advantages_batch.squeeze(-1) * torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param)
            surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()

            # 熵正则
            loss_actor = surrogate_loss - self.entropy_coef * entropy_batch.mean()

            # 只更新 actor 参数
            self.actor_optimizer.zero_grad()
            loss_actor.backward()
            nn.utils.clip_grad_norm_(self.actor_critic.actor_parameters(), self.max_grad_norm)
            self.actor_optimizer.step()

            mean_surrogate_loss += surrogate_loss.item()
            actor_updates += 1

        # ========= 第二段：只更新 Critic（2E 次） =========
        critic_updates = 0
        for (obs_batch, critic_obs_batch, actions_batch, target_values_batch, advantages_batch, returns_batch,
            old_actions_log_prob_batch, old_mu_batch, old_sigma_batch, hid_states_batch, masks_batch,
            rewards_batch, dynamics_batch, srb_dynamics_batch) in generator_critic:

            # 价值前向（允许梯度）
            value_batch = self.actor_critic.evaluate(critic_obs_batch, masks=masks_batch, hidden_states=hid_states_batch[1])

            # 价值损失（可裁剪）
            if self.use_clipped_value_loss:
                value_clipped = target_values_batch + (value_batch - target_values_batch).clamp(-self.clip_param, self.clip_param)
                value_losses = (value_batch - returns_batch).pow(2)
                value_losses_clipped = (value_clipped - returns_batch).pow(2)
                value_loss = torch.max(value_losses, value_losses_clipped).mean()
            else:
                value_loss = (returns_batch - value_batch).pow(2).mean()

            # HJB 残差（可选）
            hjb_loss = torch.tensor(0.0, device=self.device)
            if self.hjb_coef >= 0.0 and srb_dynamics_batch is not None:
                critic_obs_batch.requires_grad_(True)
                values_grad = self.actor_critic.evaluate(critic_obs_batch, masks=masks_batch, hidden_states=hid_states_batch[1])
                value_derivative = torch.autograd.grad(
                    values_grad, critic_obs_batch,
                    grad_outputs=torch.ones_like(values_grad),
                    create_graph=True, retain_graph=True
                )[0]
                critic_obs_batch.requires_grad_(False)

                B, obs_dim = value_derivative.shape
                value_derivative_dot_f = torch.bmm(
                    value_derivative.view(B, 1, obs_dim),
                    srb_dynamics_batch.view(B, obs_dim, 1)
                ).view(-1)

                # ★ 若 r 为“每步奖励”，建议 rhs 用 rewards_batch.view(-1)/self.dt
                # ★ 且 ρ 用 -log(gamma)/self.dt
                target = (self.rho.to(self.device)) * values_grad.view(-1)
                rhs    = value_derivative_dot_f + rewards_batch.view(-1)
                hjb_loss = F.mse_loss(target, rhs)

            loss_critic = self.value_loss_coef * value_loss + self.hjb_coef * hjb_loss

            # 只更新 critic 参数
            self.critic_optimizer.zero_grad()
            loss_critic.backward()
            nn.utils.clip_grad_norm_(self.actor_critic.critic_parameters(), self.max_grad_norm)
            self.critic_optimizer.step()

            mean_value_loss += value_loss.item()
            mean_hjb_loss   += hjb_loss.item()
            critic_updates  += 1

        # 统计 & 清空
        mean_surrogate_loss /= max(actor_updates, 1)
        mean_value_loss     /= max(critic_updates, 1)
        mean_hjb_loss       /= max(critic_updates, 1)

        self.storage.clear()
        return mean_value_loss, mean_surrogate_loss, mean_hjb_loss
