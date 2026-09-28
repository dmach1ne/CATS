"""LipsNet++ policy and PPO integration."""

from typing import Any, Optional, Union

import numpy as np
import torch as th
from gymnasium import spaces
from torch import nn
from torch.nn import functional as F

from cats_common.lipsnetpp import LipsNetPlusPlus, LipsNetPlusPlusLoss
from stable_baselines3 import PPO
from stable_baselines3.common.distributions import (
    DiagGaussianDistribution,
    Distribution,
    StateDependentNoiseDistribution,
)
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.preprocessing import get_action_dim, preprocess_obs
from stable_baselines3.common.utils import explained_variance


def _split_architecture(
    net_arch: Optional[Union[list[int], dict[str, list[int]]]],
) -> tuple[list[int], list[int]]:
    if net_arch is None:
        return [64, 64], [64, 64]
    if isinstance(net_arch, dict):
        return list(net_arch.get("pi", [])), list(net_arch.get("vf", []))
    return list(net_arch), list(net_arch)


def _value_network(
    obs_dim: int,
    hidden_sizes: list[int],
    activation_fn: type[nn.Module],
    ortho_init: bool,
) -> nn.Sequential:
    layers: list[nn.Module] = []
    input_dim = obs_dim
    for hidden_dim in hidden_sizes:
        layers.extend([nn.Linear(input_dim, hidden_dim), activation_fn()])
        input_dim = hidden_dim
    layers.append(nn.Linear(input_dim, 1))
    network = nn.Sequential(*layers)

    if ortho_init:
        linears = [module for module in network if isinstance(module, nn.Linear)]
        for linear in linears[:-1]:
            nn.init.orthogonal_(linear.weight, gain=np.sqrt(2))
            nn.init.zeros_(linear.bias)
        nn.init.orthogonal_(linears[-1].weight, gain=1.0)
        nn.init.zeros_(linears[-1].bias)
    return network


class LipsNetPlusPlusPolicy(ActorCriticPolicy):
    """Actor-history PPO policy with a current-observation value network."""

    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        lr_schedule,
        history_len: int,
        lipsnetpp_lambda_k: float,
        lipsnetpp_lambda_h: float,
        lipsnetpp_kernel_scale: float = 0.02,
        **kwargs: Any,
    ) -> None:
        if not isinstance(observation_space, spaces.Box):
            raise TypeError("LipsNet++ PPO requires a Box observation space")
        if len(observation_space.shape) != 2:
            raise ValueError("LipsNet++ PPO expects (history_len, obs_dim) observations")
        if observation_space.shape[0] != history_len:
            raise ValueError("history_len does not match the wrapped observation space")
        if not isinstance(action_space, spaces.Box):
            raise TypeError("LipsNet++ PPO currently supports continuous actions only")

        self.history_len = int(history_len)
        self.obs_dim = int(observation_space.shape[1])
        self.lipsnetpp_lambda_k = float(lipsnetpp_lambda_k)
        self.lipsnetpp_lambda_h = float(lipsnetpp_lambda_h)
        self.lipsnetpp_kernel_scale = float(lipsnetpp_kernel_scale)
        super().__init__(
            observation_space=observation_space,
            action_space=action_space,
            lr_schedule=lr_schedule,
            **kwargs,
        )

        actor_arch, value_arch = _split_architecture(self.net_arch)
        if not actor_arch:
            raise ValueError("LipsNet++ PPO requires a non-empty actor architecture")

        self.action_net = LipsNetPlusPlus(
            obs_dim=self.obs_dim,
            history_len=self.history_len,
            action_dim=get_action_dim(action_space),
            hidden_sizes=actor_arch,
            lambda_k=self.lipsnetpp_lambda_k,
            lambda_h=self.lipsnetpp_lambda_h,
            kernel_scale=self.lipsnetpp_kernel_scale,
            activation_fn=self.activation_fn,
            output_activation=nn.Tanh,
        ).to(self.device)
        self.value_net = _value_network(
            self.obs_dim,
            value_arch,
            self.activation_fn,
            self.ortho_init,
        ).to(self.device)
        self.mlp_extractor = nn.Identity()
        self.optimizer = self.optimizer_class(
            self.parameters(),
            lr=lr_schedule(1),
            **self.optimizer_kwargs,
        )

    @property
    def lipsnetpp(self) -> LipsNetPlusPlus:
        if not isinstance(self.action_net, LipsNetPlusPlus):
            raise TypeError("LipsNet++ action network is not initialized")
        return self.action_net

    def _get_constructor_parameters(self) -> dict[str, Any]:
        data = super()._get_constructor_parameters()
        data.update(
            history_len=self.history_len,
            lipsnetpp_lambda_k=self.lipsnetpp_lambda_k,
            lipsnetpp_lambda_h=self.lipsnetpp_lambda_h,
            lipsnetpp_kernel_scale=self.lipsnetpp_kernel_scale,
        )
        return data

    def _prepared_observations(self, obs: th.Tensor) -> th.Tensor:
        prepared = preprocess_obs(
            obs,
            self.observation_space,
            normalize_images=self.normalize_images,
        )
        if not isinstance(prepared, th.Tensor):
            raise TypeError("LipsNet++ PPO requires tensor observations")
        return prepared

    def _distribution_value_and_filtered(
        self, obs: th.Tensor
    ) -> tuple[Distribution, th.Tensor, th.Tensor]:
        prepared = self._prepared_observations(obs)
        filtered_observation = self.lipsnetpp.filtered_current(prepared)
        actor_latent = self.lipsnetpp.hidden_net(filtered_observation)
        mean_actions = self.lipsnetpp.output_activation(
            self.lipsnetpp.output_layer(actor_latent)
        )

        if isinstance(self.action_dist, DiagGaussianDistribution):
            distribution = self.action_dist.proba_distribution(
                mean_actions, self.log_std
            )
        elif isinstance(self.action_dist, StateDependentNoiseDistribution):
            distribution = self.action_dist.proba_distribution(
                mean_actions,
                self.log_std,
                actor_latent,
            )
        else:
            raise TypeError("LipsNet++ PPO requires a Gaussian action distribution")

        values = self.value_net(prepared[:, 0, :])
        return distribution, values, filtered_observation

    def _distribution_and_value(
        self, obs: th.Tensor
    ) -> tuple[Distribution, th.Tensor]:
        distribution, values, _ = self._distribution_value_and_filtered(obs)
        return distribution, values

    def forward(
        self, obs: th.Tensor, deterministic: bool = False
    ) -> tuple[th.Tensor, th.Tensor, th.Tensor]:
        distribution, values = self._distribution_and_value(obs)
        actions = distribution.get_actions(deterministic=deterministic)
        log_prob = distribution.log_prob(actions)
        actions = actions.reshape((-1, *self.action_space.shape))
        return actions, values, log_prob

    def evaluate_actions(
        self, obs: th.Tensor, actions: th.Tensor
    ) -> tuple[th.Tensor, th.Tensor, Optional[th.Tensor]]:
        distribution, values = self._distribution_and_value(obs)
        return values, distribution.log_prob(actions), distribution.entropy()

    def evaluate_actions_with_filtered(
        self, obs: th.Tensor, actions: th.Tensor
    ) -> tuple[
        th.Tensor,
        th.Tensor,
        Optional[th.Tensor],
        th.Tensor,
    ]:
        distribution, values, filtered = self._distribution_value_and_filtered(obs)
        return (
            values,
            distribution.log_prob(actions),
            distribution.entropy(),
            filtered,
        )

    def get_distribution(self, obs: th.Tensor) -> Distribution:
        distribution, _ = self._distribution_and_value(obs)
        return distribution

    def predict_values(self, obs: th.Tensor) -> th.Tensor:
        prepared = self._prepared_observations(obs)
        return self.value_net(prepared[:, 0, :])

    def lipsnetpp_regularization_loss(
        self,
        obs: Optional[th.Tensor] = None,
        *,
        filtered_observation: Optional[th.Tensor] = None,
    ) -> LipsNetPlusPlusLoss:
        if filtered_observation is not None:
            return self.lipsnetpp.regularization_loss_from_filtered(
                filtered_observation
            )
        if obs is None:
            raise ValueError("obs or filtered_observation is required")
        return self.lipsnetpp.regularization_loss(
            self._prepared_observations(obs)
        )


class LipsNetPlusPlusPPO(PPO):
    """PPO with the released LipsNet++ regularizers in the actor objective."""

    policy: LipsNetPlusPlusPolicy

    def train(self) -> None:
        self.policy.set_training_mode(True)
        self._update_learning_rate(self.policy.optimizer)
        clip_range = self.clip_range(self._current_progress_remaining)
        if self.clip_range_vf is not None:
            clip_range_vf = self.clip_range_vf(self._current_progress_remaining)

        entropy_losses: list[float] = []
        pg_losses: list[float] = []
        value_losses: list[float] = []
        clip_fractions: list[float] = []
        regularizer_losses: list[float] = []
        jacobian_losses: list[float] = []
        filter_losses: list[float] = []

        continue_training = True
        for epoch in range(self.n_epochs):
            approx_kl_divs = []
            for rollout_data in self.rollout_buffer.get(self.batch_size):
                actions = rollout_data.actions
                (
                    values,
                    log_prob,
                    entropy,
                    filtered_observation,
                ) = self.policy.evaluate_actions_with_filtered(
                    rollout_data.observations, actions
                )
                values = values.flatten()

                advantages = rollout_data.advantages
                if self.normalize_advantage and len(advantages) > 1:
                    advantages = (advantages - advantages.mean()) / (
                        advantages.std() + 1e-8
                    )

                ratio = th.exp(log_prob - rollout_data.old_log_prob)
                policy_loss_1 = advantages * ratio
                policy_loss_2 = advantages * th.clamp(
                    ratio, 1 - clip_range, 1 + clip_range
                )
                policy_loss = -th.min(policy_loss_1, policy_loss_2).mean()
                pg_losses.append(policy_loss.item())
                clip_fractions.append(
                    th.mean((th.abs(ratio - 1) > clip_range).float()).item()
                )

                if self.clip_range_vf is None:
                    values_pred = values
                else:
                    values_pred = rollout_data.old_values + th.clamp(
                        values - rollout_data.old_values,
                        -clip_range_vf,
                        clip_range_vf,
                    )
                value_loss = F.mse_loss(rollout_data.returns, values_pred)
                value_losses.append(value_loss.item())

                if entropy is None:
                    entropy_loss = -th.mean(-log_prob)
                else:
                    entropy_loss = -th.mean(entropy)
                entropy_losses.append(entropy_loss.item())

                with th.no_grad():
                    log_ratio = log_prob - rollout_data.old_log_prob
                    approx_kl_div = th.mean(
                        (th.exp(log_ratio) - 1) - log_ratio
                    ).cpu().numpy()
                    approx_kl_divs.append(approx_kl_div)

                if (
                    self.target_kl is not None
                    and approx_kl_div > 1.5 * self.target_kl
                ):
                    continue_training = False
                    if self.verbose >= 1:
                        print(
                            f"Early stopping at step {epoch} due to reaching "
                            f"max kl: {approx_kl_div:.2f}"
                        )
                    break

                regularization = self.policy.lipsnetpp_regularization_loss(
                    filtered_observation=filtered_observation
                )
                loss = (
                    policy_loss
                    + self.ent_coef * entropy_loss
                    + self.vf_coef * value_loss
                    + regularization.total
                )
                regularizer_losses.append(regularization.total.item())
                jacobian_losses.append(regularization.jacobian.item())
                filter_losses.append(regularization.filter.item())

                self.policy.optimizer.zero_grad()
                loss.backward()
                th.nn.utils.clip_grad_norm_(
                    self.policy.parameters(), self.max_grad_norm
                )
                self.policy.optimizer.step()

            self._n_updates += 1
            if not continue_training:
                break

        explained_var = explained_variance(
            self.rollout_buffer.values.flatten(),
            self.rollout_buffer.returns.flatten(),
        )
        self.logger.record("train/entropy_loss", np.mean(entropy_losses))
        self.logger.record("train/policy_gradient_loss", np.mean(pg_losses))
        self.logger.record("train/value_loss", np.mean(value_losses))
        self.logger.record("train/approx_kl", np.mean(approx_kl_divs))
        self.logger.record("train/clip_fraction", np.mean(clip_fractions))
        self.logger.record("train/loss", loss.item())
        self.logger.record("train/explained_variance", explained_var)
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
        if hasattr(self.policy, "log_std"):
            self.logger.record(
                "train/std", th.exp(self.policy.log_std).mean().item()
            )
        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/clip_range", clip_range)
        if self.clip_range_vf is not None:
            self.logger.record("train/clip_range_vf", clip_range_vf)
