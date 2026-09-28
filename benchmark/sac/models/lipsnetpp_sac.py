"""LipsNet++ policy and SAC integration."""

from typing import Any, Optional, Union

import numpy as np
import torch as th
from gymnasium import spaces
from torch import nn
from torch.nn import functional as F

from afqr_common.lipsnetpp import LipsNetPlusPlus, LipsNetPlusPlusLoss
from stable_baselines3 import SAC
from stable_baselines3.common.policies import BaseFeaturesExtractor
from stable_baselines3.common.preprocessing import get_action_dim
from stable_baselines3.common.torch_layers import get_actor_critic_arch
from stable_baselines3.common.type_aliases import PyTorchObs, Schedule
from stable_baselines3.common.utils import polyak_update
from stable_baselines3.sac.policies import (
    LOG_STD_MAX,
    LOG_STD_MIN,
    Actor,
    SACPolicy,
)


class CurrentObservationExtractor(BaseFeaturesExtractor):
    """Expose only the current row of an observation history to a critic."""

    def __init__(self, observation_space: spaces.Box) -> None:
        if len(observation_space.shape) != 2:
            raise ValueError("expected (history_len, obs_dim) observations")
        super().__init__(observation_space, features_dim=observation_space.shape[1])

    def forward(self, observations: th.Tensor) -> th.Tensor:
        return observations[:, 0, :]


class LipsNetPlusPlusActor(Actor):
    """SAC actor whose deterministic action map is LipsNet++."""

    def __init__(
        self,
        *args: Any,
        history_len: int,
        lipsnetpp_lambda_k: float,
        lipsnetpp_lambda_h: float,
        lipsnetpp_kernel_scale: float = 0.02,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        if len(self.observation_space.shape) != 2:
            raise ValueError("LipsNet++ SAC expects history observations")
        if self.observation_space.shape[0] != history_len:
            raise ValueError("history_len does not match the observation space")

        self.history_len = int(history_len)
        self.obs_dim = int(self.observation_space.shape[1])
        self.lipsnetpp_lambda_k = float(lipsnetpp_lambda_k)
        self.lipsnetpp_lambda_h = float(lipsnetpp_lambda_h)
        self.lipsnetpp_kernel_scale = float(lipsnetpp_kernel_scale)
        action_dim = get_action_dim(self.action_space)

        self.latent_pi = nn.Identity()
        self.mu = LipsNetPlusPlus(
            obs_dim=self.obs_dim,
            history_len=self.history_len,
            action_dim=action_dim,
            hidden_sizes=self.net_arch,
            lambda_k=self.lipsnetpp_lambda_k,
            lambda_h=self.lipsnetpp_lambda_h,
            kernel_scale=self.lipsnetpp_kernel_scale,
            activation_fn=self.activation_fn,
            output_activation=nn.Identity,
            regularization_output_tanh=True,
        )

    @property
    def lipsnetpp(self) -> LipsNetPlusPlus:
        if not isinstance(self.mu, LipsNetPlusPlus):
            raise TypeError("LipsNet++ SAC actor is not initialized")
        return self.mu

    def _get_constructor_parameters(self) -> dict[str, Any]:
        data = super()._get_constructor_parameters()
        data.update(
            history_len=self.history_len,
            lipsnetpp_lambda_k=self.lipsnetpp_lambda_k,
            lipsnetpp_lambda_h=self.lipsnetpp_lambda_h,
            lipsnetpp_kernel_scale=self.lipsnetpp_kernel_scale,
        )
        return data

    def get_action_dist_params(
        self, obs: PyTorchObs
    ) -> tuple[th.Tensor, th.Tensor, dict[str, th.Tensor]]:
        if not isinstance(obs, th.Tensor):
            raise TypeError("LipsNet++ SAC requires tensor observations")
        prepared_observations = obs.float()
        mean_actions, actor_latent = self.lipsnetpp.forward_with_latent(
            prepared_observations
        )
        if self.use_sde:
            return mean_actions, self.log_std, dict(latent_sde=actor_latent)
        log_std = self.log_std(actor_latent)
        log_std = th.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)
        return mean_actions, log_std, {}

    def regularization_loss(
        self, obs: th.Tensor
    ) -> LipsNetPlusPlusLoss:
        return self.lipsnetpp.regularization_loss(obs.float())


class LipsNetPlusPlusSACPolicy(SACPolicy):
    """SAC policy with a history actor and current-state critics."""

    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Box,
        lr_schedule: Schedule,
        history_len: int,
        lipsnetpp_lambda_k: float,
        lipsnetpp_lambda_h: float,
        lipsnetpp_kernel_scale: float = 0.02,
        net_arch: Optional[Union[list[int], dict[str, list[int]]]] = None,
        **kwargs: Any,
    ) -> None:
        if not isinstance(observation_space, spaces.Box):
            raise TypeError("LipsNet++ SAC requires a Box observation space")
        if len(observation_space.shape) != 2:
            raise ValueError("LipsNet++ SAC expects history observations")

        self.history_len = int(history_len)
        self.lipsnetpp_lambda_k = float(lipsnetpp_lambda_k)
        self.lipsnetpp_lambda_h = float(lipsnetpp_lambda_h)
        self.lipsnetpp_kernel_scale = float(lipsnetpp_kernel_scale)
        kwargs.pop("features_extractor_class", None)
        kwargs.pop("features_extractor_kwargs", None)
        super().__init__(
            observation_space=observation_space,
            action_space=action_space,
            lr_schedule=lr_schedule,
            net_arch=net_arch,
            features_extractor_class=CurrentObservationExtractor,
            features_extractor_kwargs=None,
            **kwargs,
        )

    def make_actor(
        self, features_extractor: Optional[BaseFeaturesExtractor] = None
    ) -> LipsNetPlusPlusActor:
        actor_kwargs = self._update_features_extractor(
            self.actor_kwargs, features_extractor
        )
        actor_kwargs.update(
            history_len=self.history_len,
            lipsnetpp_lambda_k=self.lipsnetpp_lambda_k,
            lipsnetpp_lambda_h=self.lipsnetpp_lambda_h,
            lipsnetpp_kernel_scale=self.lipsnetpp_kernel_scale,
        )
        return LipsNetPlusPlusActor(**actor_kwargs).to(self.device)

    def _get_constructor_parameters(self) -> dict[str, Any]:
        data = super()._get_constructor_parameters()
        data.update(
            history_len=self.history_len,
            lipsnetpp_lambda_k=self.lipsnetpp_lambda_k,
            lipsnetpp_lambda_h=self.lipsnetpp_lambda_h,
            lipsnetpp_kernel_scale=self.lipsnetpp_kernel_scale,
        )
        return data


class LipsNetPlusPlusSAC(SAC):
    """SAC with the released LipsNet++ regularizers in the actor objective."""

    policy: LipsNetPlusPlusSACPolicy
    actor: LipsNetPlusPlusActor

    def train(self, gradient_steps: int, batch_size: int = 64) -> None:
        self.policy.set_training_mode(True)
        optimizers = [self.actor.optimizer, self.critic.optimizer]
        if self.ent_coef_optimizer is not None:
            optimizers.append(self.ent_coef_optimizer)
        self._update_learning_rate(optimizers)

        ent_coef_losses: list[float] = []
        ent_coefs: list[float] = []
        actor_losses: list[float] = []
        critic_losses: list[float] = []
        regularizer_losses: list[float] = []
        jacobian_losses: list[float] = []
        filter_losses: list[float] = []

        for gradient_step in range(gradient_steps):
            replay_data = self.replay_buffer.sample(
                batch_size, env=self._vec_normalize_env
            )
            if self.use_sde:
                self.actor.reset_noise()

            actions_pi, log_prob = self.actor.action_log_prob(
                replay_data.observations
            )
            log_prob = log_prob.reshape(-1, 1)

            ent_coef_loss = None
            if self.ent_coef_optimizer is not None and self.log_ent_coef is not None:
                ent_coef = th.exp(self.log_ent_coef.detach())
                ent_coef_loss = -(
                    self.log_ent_coef
                    * (log_prob + self.target_entropy).detach()
                ).mean()
                ent_coef_losses.append(ent_coef_loss.item())
            else:
                ent_coef = self.ent_coef_tensor
            ent_coefs.append(ent_coef.item())

            if ent_coef_loss is not None and self.ent_coef_optimizer is not None:
                self.ent_coef_optimizer.zero_grad()
                ent_coef_loss.backward()
                self.ent_coef_optimizer.step()

            with th.no_grad():
                next_actions, next_log_prob = self.actor.action_log_prob(
                    replay_data.next_observations
                )
                next_q_values = th.cat(
                    self.critic_target(
                        replay_data.next_observations, next_actions
                    ),
                    dim=1,
                )
                next_q_values, _ = th.min(next_q_values, dim=1, keepdim=True)
                next_q_values = next_q_values - ent_coef * next_log_prob.reshape(
                    -1, 1
                )
                target_q_values = replay_data.rewards + (
                    1 - replay_data.dones
                ) * self.gamma * next_q_values

            current_q_values = self.critic(
                replay_data.observations, replay_data.actions
            )
            critic_loss = 0.5 * sum(
                F.mse_loss(current_q, target_q_values)
                for current_q in current_q_values
            )
            critic_losses.append(critic_loss.item())
            self.critic.optimizer.zero_grad()
            critic_loss.backward()
            self.critic.optimizer.step()

            q_values_pi = th.cat(
                self.critic(replay_data.observations, actions_pi), dim=1
            )
            min_qf_pi, _ = th.min(q_values_pi, dim=1, keepdim=True)
            base_actor_loss = (ent_coef * log_prob - min_qf_pi).mean()
            regularization = self.actor.regularization_loss(
                replay_data.observations
            )
            actor_loss = base_actor_loss + regularization.total
            actor_losses.append(actor_loss.item())
            regularizer_losses.append(regularization.total.item())
            jacobian_losses.append(regularization.jacobian.item())
            filter_losses.append(regularization.filter.item())

            self.actor.optimizer.zero_grad()
            actor_loss.backward()
            self.actor.optimizer.step()

            if gradient_step % self.target_update_interval == 0:
                polyak_update(
                    self.critic.parameters(),
                    self.critic_target.parameters(),
                    self.tau,
                )
                polyak_update(
                    self.batch_norm_stats,
                    self.batch_norm_stats_target,
                    1.0,
                )

        self._n_updates += gradient_steps
        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/ent_coef", np.mean(ent_coefs))
        self.logger.record("train/actor_loss", np.mean(actor_losses))
        self.logger.record("train/critic_loss", np.mean(critic_losses))
        self.logger.record(
            "train/lipsnetpp_regularization_loss",
            np.mean(regularizer_losses),
        )
        self.logger.record(
            "train/lipsnetpp_jacobian_loss", np.mean(jacobian_losses)
        )
        self.logger.record(
            "train/lipsnetpp_filter_loss", np.mean(filter_losses)
        )
        if ent_coef_losses:
            self.logger.record("train/ent_coef_loss", np.mean(ent_coef_losses))
