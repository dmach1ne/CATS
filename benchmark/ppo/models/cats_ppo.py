import numpy as np
import os
import torch as th
from torch.nn import functional as F
from gymnasium import spaces
from typing import Any, Optional, Union

from stable_baselines3 import PPO
from stable_baselines3.common.buffers import RolloutBuffer
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.utils import explained_variance

from models.custom_buffer import CustomRolloutBuffer
from models.collect_rollouts import collect_rollouts_with_next_obs
from models.policy_evaluation import evaluate_actions_with_context
from cats_common.cats_schedules import linear_ramp_up_coefficient



class CATSPPO(PPO):
    collect_rollouts = collect_rollouts_with_next_obs

    def __init__(
        self,
        policy: Union[str, type[ActorCriticPolicy]],
        env: Union[str, Any],
        learning_rate: Union[float, Any] = 3e-4,
        n_steps: int = 2048,
        batch_size: int = 64,
        n_epochs: int = 10,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_range: Union[float, Any] = 0.2,
        clip_range_vf: Optional[Union[float, Any]] = None,
        normalize_advantage: bool = True,
        ent_coef: float = 0.0,
        vf_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        use_sde: bool = False,
        sde_sample_freq: int = -1,
        rollout_buffer_class: Optional[type[RolloutBuffer]] = None,
        rollout_buffer_kwargs: Optional[dict[str, Any]] = None,
        target_kl: Optional[float] = None,
        stats_window_size: int = 100,
        tensorboard_log: Optional[str] = None,
        policy_kwargs: Optional[dict[str, Any]] = None,
        verbose: int = 0,
        seed: Optional[int] = None,
        device: Union[th.device, str] = "auto",
        _init_setup_model: bool = True,
        cats_alpha: float = 30.0,
    ):
        super().__init__(
            policy,
            env,
            learning_rate=learning_rate,
            n_steps=n_steps,
            batch_size=batch_size,
            n_epochs=n_epochs,
            gamma=gamma,
            gae_lambda=gae_lambda,
            clip_range=clip_range,
            clip_range_vf=clip_range_vf,
            normalize_advantage=normalize_advantage,
            ent_coef=ent_coef,
            vf_coef=vf_coef,
            max_grad_norm=max_grad_norm,
            use_sde=use_sde,
            sde_sample_freq=sde_sample_freq,
            rollout_buffer_class=CustomRolloutBuffer,
            rollout_buffer_kwargs=rollout_buffer_kwargs,
            target_kl=target_kl,
            stats_window_size=stats_window_size,
            tensorboard_log=tensorboard_log,
            policy_kwargs=policy_kwargs,
            verbose=verbose,
            seed=seed,
            device=device,
            _init_setup_model=_init_setup_model,
        )
        self.cats_alpha = cats_alpha

    def _cats_lambda(self) -> float:
        completed_fraction = min(
            1.0, max(0.0, 1.0 - self._current_progress_remaining)
        )
        return linear_ramp_up_coefficient(self.cats_alpha, completed_fraction)

    def train(self) -> None:
        self.policy.set_training_mode(True)
        self._update_learning_rate(self.policy.optimizer)
        clip_range = self.clip_range(self._current_progress_remaining)
        if self.clip_range_vf is not None:
            clip_range_vf = self.clip_range_vf(self._current_progress_remaining)

        record_auxiliary = os.environ.get(
            "CATS_DISABLE_AUXILIARY_METRICS", "0"
        ) != "1"
        entropy_losses = []
        pg_losses, value_losses = [], []
        clip_fractions = []
        cats_lambda = self._cats_lambda()
        if record_auxiliary:
            cats_losses = []
            loss_t_values = []

        continue_training = True
        for epoch in range(self.n_epochs):
            approx_kl_divs = []
            for rollout_data in self.rollout_buffer.get(self.batch_size):
                actions = rollout_data.actions
                if isinstance(self.action_space, spaces.Discrete):
                    actions = rollout_data.actions.long().flatten()

                (
                    values,
                    log_prob,
                    entropy,
                    pi_s,
                    _,
                    _,
                ) = evaluate_actions_with_context(
                    self.policy,
                    rollout_data.observations,
                    actions,
                    deterministic_action=True,
                )
                values = values.flatten()
                advantages = rollout_data.advantages
                if self.normalize_advantage and len(advantages) > 1:
                    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

                ratio = th.exp(log_prob - rollout_data.old_log_prob)
                policy_loss_1 = advantages * ratio
                policy_loss_2 = advantages * th.clamp(ratio, 1 - clip_range, 1 + clip_range)
                policy_loss = -th.min(policy_loss_1, policy_loss_2).mean()

                pg_losses.append(policy_loss.item())
                clip_fraction = th.mean((th.abs(ratio - 1) > clip_range).float()).item()
                clip_fractions.append(clip_fraction)

                if self.clip_range_vf is None:
                    values_pred = values
                else:
                    values_pred = rollout_data.old_values + th.clamp(
                        values - rollout_data.old_values, -clip_range_vf, clip_range_vf
                    )
                value_loss = F.mse_loss(rollout_data.returns, values_pred)
                value_losses.append(value_loss.item())

                if entropy is None:
                    entropy_loss = -th.mean(-log_prob)
                else:
                    entropy_loss = -th.mean(entropy)
                entropy_losses.append(entropy_loss.item())

                next_observations = rollout_data.next_observations
                pi_s_next = self.policy._predict(
                    next_observations, deterministic=True
                )
                action_diff = pi_s - pi_s_next

                loss_t = 0.5 * action_diff.square().mean()
                if record_auxiliary:
                    loss_t_values.append(loss_t.item())

                cats_loss = cats_lambda * loss_t
                if record_auxiliary:
                    cats_losses.append(cats_loss.item())

                loss = policy_loss + self.ent_coef * entropy_loss + self.vf_coef * value_loss + cats_loss

                with th.no_grad():
                    log_ratio = log_prob - rollout_data.old_log_prob
                    approx_kl_div = th.mean((th.exp(log_ratio) - 1) - log_ratio).cpu().numpy()
                    approx_kl_divs.append(approx_kl_div)

                if self.target_kl is not None and approx_kl_div > 1.5 * self.target_kl:
                    continue_training = False
                    if self.verbose >= 1:
                        print(f"Early stopping at step {epoch} due to reaching max kl: {approx_kl_div:.2f}")
                    break

                self.policy.optimizer.zero_grad()
                loss.backward()
                th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.policy.optimizer.step()

            self._n_updates += 1
            if not continue_training:
                break

        explained_var = explained_variance(self.rollout_buffer.values.flatten(), self.rollout_buffer.returns.flatten())

        self.logger.record("train/entropy_loss", np.mean(entropy_losses))
        self.logger.record("train/policy_gradient_loss", np.mean(pg_losses))
        self.logger.record("train/value_loss", np.mean(value_losses))
        self.logger.record("train/approx_kl", np.mean(approx_kl_divs))
        self.logger.record("train/clip_fraction", np.mean(clip_fractions))
        self.logger.record("train/loss", loss.item())
        self.logger.record("train/explained_variance", explained_var)
        if record_auxiliary:
            if loss_t_values:
                self.logger.record("train/loss_t", np.mean(loss_t_values))
            self.logger.record("train/cats_lambda", cats_lambda)
            if cats_losses:
                self.logger.record("train/cats_loss", np.mean(cats_losses))
        if hasattr(self.policy, "log_std"):
            self.logger.record("train/std", th.exp(self.policy.log_std).mean().item())
        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
