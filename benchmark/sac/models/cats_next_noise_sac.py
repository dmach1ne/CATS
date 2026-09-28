import os
import sys
from typing import Any, Optional, Union

import numpy as np
import torch as th
from torch.nn import functional as F

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from custom_sac import CustomSAC
from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.noise import ActionNoise
from stable_baselines3.common.type_aliases import GymEnv, Schedule
from stable_baselines3.common.utils import polyak_update
from stable_baselines3.sac.policies import SACPolicy


class CATSNextNoiseSAC(CustomSAC):
    def __init__(
        self,
        policy: Union[str, type[SACPolicy]],
        env: Union[GymEnv, str],
        learning_rate: Union[float, Schedule] = 3e-4,
        buffer_size: int = 1_000_000,
        learning_starts: int = 100,
        batch_size: int = 256,
        tau: float = 0.005,
        gamma: float = 0.99,
        train_freq: Union[int, tuple[int, str]] = 1,
        gradient_steps: int = 1,
        action_noise: Optional[ActionNoise] = None,
        replay_buffer_class: Optional[type[ReplayBuffer]] = None,
        replay_buffer_kwargs: Optional[dict[str, Any]] = None,
        optimize_memory_usage: bool = False,
        ent_coef: Union[str, float] = "auto",
        target_update_interval: int = 1,
        target_entropy: Union[str, float] = "auto",
        use_sde: bool = False,
        sde_sample_freq: int = -1,
        use_sde_at_warmup: bool = False,
        stats_window_size: int = 100,
        tensorboard_log: Optional[str] = None,
        policy_kwargs: Optional[dict[str, Any]] = None,
        verbose: int = 0,
        seed: Optional[int] = None,
        device: Union[th.device, str] = "auto",
        _init_setup_model: bool = True,
        cats_alpha: float = 30.0,
        next_noise_sigma: float = 0.1,
    ):
        self.cats_alpha = cats_alpha
        self.next_noise_sigma = next_noise_sigma
        super().__init__(
            policy=policy,
            env=env,
            learning_rate=learning_rate,
            buffer_size=buffer_size,
            learning_starts=learning_starts,
            batch_size=batch_size,
            tau=tau,
            gamma=gamma,
            train_freq=train_freq,
            gradient_steps=gradient_steps,
            action_noise=action_noise,
            replay_buffer_class=replay_buffer_class,
            replay_buffer_kwargs=replay_buffer_kwargs,
            optimize_memory_usage=optimize_memory_usage,
            ent_coef=ent_coef,
            target_update_interval=target_update_interval,
            target_entropy=target_entropy,
            use_sde=use_sde,
            sde_sample_freq=sde_sample_freq,
            use_sde_at_warmup=use_sde_at_warmup,
            stats_window_size=stats_window_size,
            tensorboard_log=tensorboard_log,
            policy_kwargs=policy_kwargs,
            verbose=verbose,
            seed=seed,
            device=device,
            _init_setup_model=_init_setup_model,
        )

    def train(self, gradient_steps: int, batch_size: int = 64) -> None:
        self.policy.set_training_mode(True)
        optimizers = [self.actor.optimizer, self.critic.optimizer]
        if self.ent_coef_optimizer is not None:
            optimizers += [self.ent_coef_optimizer]

        self._update_learning_rate(optimizers)

        ent_coef_losses, ent_coefs = [], []
        actor_losses, critic_losses = [], []
        mean_target_q_values = []
        loss_t_values = []
        lambda_values = []
        obs_cos_sim_values = []
        obs_weight_mean_values = []

        for gradient_step in range(gradient_steps):
            replay_data = self.replay_buffer.sample(batch_size, env=self._vec_normalize_env)

            if self.use_sde:
                self.actor.reset_noise()

            actions_pi, log_prob = self.actor.action_log_prob(replay_data.observations)
            log_prob = log_prob.reshape(-1, 1)

            ent_coef_loss = None
            if self.ent_coef_optimizer is not None and self.log_ent_coef is not None:
                ent_coef = th.exp(self.log_ent_coef.detach())
                ent_coef_loss = -(self.log_ent_coef * (log_prob + self.target_entropy).detach()).mean()
                ent_coef_losses.append(ent_coef_loss.item())
            else:
                ent_coef = self.ent_coef_tensor

            ent_coefs.append(ent_coef.item())

            if ent_coef_loss is not None and self.ent_coef_optimizer is not None:
                self.ent_coef_optimizer.zero_grad()
                ent_coef_loss.backward()
                self.ent_coef_optimizer.step()

            with th.no_grad():
                next_actions, next_log_prob = self.actor.action_log_prob(replay_data.next_observations)
                next_q_values = th.cat(self.critic_target(replay_data.next_observations, next_actions), dim=1)
                next_q_values, _ = th.min(next_q_values, dim=1, keepdim=True)
                next_q_values = next_q_values - ent_coef * next_log_prob.reshape(-1, 1)
                target_q_values = replay_data.rewards + (1 - replay_data.dones) * self.gamma * next_q_values
                mean_target_q_values.append(target_q_values.mean().item())

            current_q_values = self.critic(replay_data.observations, replay_data.actions)
            critic_loss = 0.5 * sum(F.mse_loss(current_q, target_q_values) for current_q in current_q_values)
            critic_losses.append(critic_loss.item())

            self.critic.optimizer.zero_grad()
            critic_loss.backward()
            self.critic.optimizer.step()

            cats_now_actions = self.policy._predict(replay_data.observations, deterministic=True).type(th.float32)
            noisy_next_observations = replay_data.next_observations + th.normal(
                mean=0.0,
                std=self.next_noise_sigma,
                size=replay_data.next_observations.size(),
                device=replay_data.next_observations.device,
            )
            cats_next_actions = self.policy._predict(noisy_next_observations, deterministic=True).type(th.float32)

            loss_t = 0.5 * F.mse_loss(cats_now_actions, cats_next_actions)
            loss_t_values.append(loss_t.item())

            progress = min(1.0, max(0.0, self.num_timesteps / max(1, self._total_timesteps)))
            cats_lambda = self.cats_alpha * progress
            lambda_values.append(cats_lambda)

            cats_loss = cats_lambda * loss_t

            q_values_pi = th.cat(self.critic(replay_data.observations, actions_pi), dim=1)
            min_qf_pi, _ = th.min(q_values_pi, dim=1, keepdim=True)
            actor_loss = (ent_coef * log_prob - min_qf_pi).mean() + cats_loss
            actor_losses.append(actor_loss.item())

            with th.no_grad():
                obs_current_actions = self.actor(replay_data.observations, deterministic=True)
                delta_a = replay_data.actions - obs_current_actions
                delta_a_norm_sq = delta_a.pow(2).mean(dim=1, keepdim=True)
                sigma_sq = ent_coef.item() ** 2 if hasattr(ent_coef, "item") else float(ent_coef) ** 2
                weight = th.exp(-delta_a_norm_sq / (2 * sigma_sq + 1e-6))
                obs_weight_mean_values.append(weight.mean().item())

            obs_actions_g = self.actor(replay_data.observations, deterministic=True)
            obs_next_actions_g = self.actor(noisy_next_observations, deterministic=True)
            obs_loss_t_g = 0.5 * F.mse_loss(obs_actions_g, obs_next_actions_g)

            obs_q_pi = th.cat(self.critic(replay_data.observations, obs_actions_g), dim=1)
            obs_min_q, _ = th.min(obs_q_pi, dim=1, keepdim=True)
            obs_q_loss = -obs_min_q.mean()

            mu = self.actor.mu
            if isinstance(mu, th.nn.Linear):
                last_params = [mu.weight, mu.bias]
            elif isinstance(mu, th.nn.Sequential):
                last_params = None
                for module in reversed(list(mu.modules())):
                    if isinstance(module, th.nn.Linear):
                        last_params = [module.weight, module.bias]
                        break
            else:
                last_params = None

            if last_params is not None:
                g_main = th.autograd.grad(obs_q_loss, last_params, retain_graph=True, allow_unused=True)
                g_penalty = th.autograd.grad(obs_loss_t_g, last_params, retain_graph=True, allow_unused=True)
                g_main_flat = th.cat([g.flatten() for g in g_main if g is not None])
                g_penalty_flat = th.cat([g.flatten() for g in g_penalty if g is not None])
                if g_main_flat.numel() > 0 and g_penalty_flat.numel() > 0:
                    cos_sim = F.cosine_similarity(g_main_flat.unsqueeze(0), g_penalty_flat.unsqueeze(0)).item()
                    obs_cos_sim_values.append(cos_sim)

            self.actor.optimizer.zero_grad()
            actor_loss.backward()
            self.actor.optimizer.step()

            if gradient_step % self.target_update_interval == 0:
                polyak_update(self.critic.parameters(), self.critic_target.parameters(), self.tau)
                polyak_update(self.batch_norm_stats, self.batch_norm_stats_target, 1.0)

        self._n_updates += gradient_steps

        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/ent_coef", np.mean(ent_coefs))
        self.logger.record("train/actor_loss", np.mean(actor_losses))
        self.logger.record("train/critic_loss", np.mean(critic_losses))
        if len(ent_coef_losses) > 0:
            self.logger.record("train/ent_coef_loss", np.mean(ent_coef_losses))
        if len(mean_target_q_values) > 0:
            self.logger.record("train/mean_target_q", np.mean(mean_target_q_values))
        if len(loss_t_values) > 0:
            self.logger.record("train/loss_t", np.mean(loss_t_values))
        if len(lambda_values) > 0:
            self.logger.record("train/cats_lambda", np.mean(lambda_values))
        if len(obs_cos_sim_values) > 0:
            self.logger.record("train/obs_cos_sim", np.mean(obs_cos_sim_values))
        if len(obs_weight_mean_values) > 0:
            self.logger.record("train/obs_weight_mean", np.mean(obs_weight_mean_values))
        self.logger.record("train/next_noise_sigma", self.next_noise_sigma)
