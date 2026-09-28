import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from stable_baselines3 import SAC
from custom_sac import CustomSAC
from typing import Any, ClassVar, Optional, TypeVar, Union

import numpy as np
import torch as th
from torch import nn
from gymnasium import spaces
from torch.nn import functional as F
import math

from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.noise import ActionNoise
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback, Schedule, PyTorchObs
from stable_baselines3.common.utils import get_parameters_by_name, polyak_update
from stable_baselines3.sac.policies import SACPolicy
from cats_common.cats_schedules import linear_ramp_up_coefficient


def _append_action_diff_poc_stats(stats: dict[str, list], action_diff: th.Tensor) -> None:
    with th.no_grad():
        abs_diff = action_diff.detach().abs()
        stats["mean_dim"].append(abs_diff.mean(dim=0).cpu().numpy())
        stats["p90_dim"].append(th.quantile(abs_diff, 0.90, dim=0).cpu().numpy())
        stats["p95_dim"].append(th.quantile(abs_diff, 0.95, dim=0).cpu().numpy())
        stats["max_dim"].append(abs_diff.max(dim=0).values.cpu().numpy())
        stats["p90_all"].append(th.quantile(abs_diff.reshape(-1), 0.90).item())


def _record_action_diff_poc_stats(logger, prefix: str, stats: dict[str, list]) -> None:
    if not stats["mean_dim"]:
        return
    mean_dim = np.mean(np.stack(stats["mean_dim"]), axis=0)
    p90_dim = np.mean(np.stack(stats["p90_dim"]), axis=0)
    p95_dim = np.mean(np.stack(stats["p95_dim"]), axis=0)
    max_dim = np.mean(np.stack(stats["max_dim"]), axis=0)
    for dim, value in enumerate(mean_dim):
        logger.record(f"train/{prefix}_abs_mean_dim{dim}", float(value))
        logger.record(f"train/{prefix}_abs_p90_dim{dim}", float(p90_dim[dim]))
        logger.record(f"train/{prefix}_abs_p95_dim{dim}", float(p95_dim[dim]))
        logger.record(f"train/{prefix}_abs_max_dim{dim}", float(max_dim[dim]))
    logger.record(f"train/{prefix}_abs_p90", float(np.mean(stats["p90_all"])))


def _safe_corr(x: th.Tensor, y: th.Tensor) -> float | None:
    x = x.detach().float().flatten()
    y = y.detach().float().flatten()
    if x.numel() < 2 or y.numel() < 2:
        return None
    x_centered = x - x.mean()
    y_centered = y - y.mean()
    denom = th.sqrt((x_centered.pow(2).mean() * y_centered.pow(2).mean()).clamp_min(1e-12))
    if denom.item() <= 1e-6:
        return None
    return ((x_centered * y_centered).mean() / denom).item()


def _overlap_rates(x: th.Tensor, y: th.Tensor) -> tuple[float, float] | None:
    x = x.detach().float().flatten()
    y = y.detach().float().flatten()
    if x.numel() == 0 or y.numel() == 0:
        return None
    x_high = x >= th.quantile(x, 0.90)
    y_high = y >= th.quantile(y, 0.90)
    if not x_high.any() or not y_high.any():
        return None
    return y_high[x_high].float().mean().item(), x_high[y_high].float().mean().item()


def _append_continuous_pair_poc_stats(
    stats: dict[str, list], action_diff_abs: th.Tensor, transition_staleness: th.Tensor
) -> None:
    with th.no_grad():
        action_largeness = action_diff_abs.detach().abs().mean(dim=1)
        staleness = transition_staleness.detach().float().flatten()
        corr = _safe_corr(action_largeness, staleness)
        overlap = _overlap_rates(action_largeness, staleness)
        if corr is not None:
            stats["corr"].append(corr)
        if overlap is not None:
            x_y_high_frac, y_x_high_frac = overlap
            stats["x_y_high_frac"].append(x_y_high_frac)
            stats["y_x_high_frac"].append(y_x_high_frac)

        corr_dim, x_y_high_frac_dim, y_x_high_frac_dim = [], [], []
        for dim in range(action_diff_abs.shape[1]):
            dim_largeness = action_diff_abs[:, dim]
            dim_corr = _safe_corr(dim_largeness, staleness)
            dim_overlap = _overlap_rates(dim_largeness, staleness)
            corr_dim.append(np.nan if dim_corr is None else dim_corr)
            if dim_overlap is None:
                x_y_high_frac_dim.append(np.nan)
                y_x_high_frac_dim.append(np.nan)
            else:
                x_y_high_frac, y_x_high_frac = dim_overlap
                x_y_high_frac_dim.append(x_y_high_frac)
                y_x_high_frac_dim.append(y_x_high_frac)
        stats["corr_dim"].append(np.array(corr_dim, dtype=np.float32))
        stats["x_y_high_frac_dim"].append(np.array(x_y_high_frac_dim, dtype=np.float32))
        stats["y_x_high_frac_dim"].append(np.array(y_x_high_frac_dim, dtype=np.float32))


def _record_continuous_pair_poc_stats(
    logger,
    stats: dict[str, list],
    corr_name: str,
    x_y_high_name: str,
    y_x_high_name: str,
) -> None:
    if stats["corr"]:
        logger.record(f"train/{corr_name}", float(np.mean(stats["corr"])))
    if stats["x_y_high_frac"]:
        logger.record(f"train/{x_y_high_name}", float(np.mean(stats["x_y_high_frac"])))
    if stats["y_x_high_frac"]:
        logger.record(f"train/{y_x_high_name}", float(np.mean(stats["y_x_high_frac"])))
    if stats["corr_dim"]:
        corr_dim = np.nanmean(np.stack(stats["corr_dim"]), axis=0)
        x_y_high_frac_dim = np.nanmean(np.stack(stats["x_y_high_frac_dim"]), axis=0)
        y_x_high_frac_dim = np.nanmean(np.stack(stats["y_x_high_frac_dim"]), axis=0)
        for dim, value in enumerate(corr_dim):
            if not np.isnan(value):
                logger.record(f"train/{corr_name}_dim{dim}", float(value))
            if not np.isnan(x_y_high_frac_dim[dim]):
                logger.record(f"train/{x_y_high_name}_dim{dim}", float(x_y_high_frac_dim[dim]))
            if not np.isnan(y_x_high_frac_dim[dim]):
                logger.record(f"train/{y_x_high_name}_dim{dim}", float(y_x_high_frac_dim[dim]))


def _append_loss_tail_stats(
    stats: dict[str, list], action_diff: th.Tensor, transition_staleness: th.Tensor
) -> None:
    with th.no_grad():
        abs_diff = action_diff.detach().abs()
        flat_abs_diff = abs_diff.reshape(-1)
        median = th.quantile(flat_abs_diff, 0.50).clamp_min(1e-8)
        stats["p95_over_p50"].append((th.quantile(flat_abs_diff, 0.95) / median).item())
        stats["p99_over_p50"].append((th.quantile(flat_abs_diff, 0.99) / median).item())

        mse_contrib = action_diff.detach().pow(2).mean(dim=1).flatten()
        total_mse = mse_contrib.sum().clamp_min(1e-12)
        n = mse_contrib.numel()
        top10_count = max(1, math.ceil(0.10 * n))
        top5_count = max(1, math.ceil(0.05 * n))
        top10_idx = th.topk(mse_contrib, top10_count).indices
        top5_idx = th.topk(mse_contrib, top5_count).indices
        stats["mse_top10_share"].append((mse_contrib[top10_idx].sum() / total_mse).item())
        stats["mse_top5_share"].append((mse_contrib[top5_idx].sum() / total_mse).item())

        staleness = transition_staleness.detach().float().flatten()
        high_stale = staleness >= th.quantile(staleness, 0.90)
        top10_mask = th.zeros_like(high_stale, dtype=th.bool)
        top10_mask[top10_idx] = True
        if top10_mask.any():
            stats["high_stale_in_top10_mse_frac"].append(high_stale[top10_mask].float().mean().item())
        if high_stale.any():
            stats["top10_mse_high_stale_frac"].append(top10_mask[high_stale].float().mean().item())


def _record_loss_tail_stats(logger, stats: dict[str, list]) -> None:
    if stats["mse_top10_share"]:
        logger.record("train/action_diff_mse_top10_share", float(np.mean(stats["mse_top10_share"])))
    if stats["mse_top5_share"]:
        logger.record("train/action_diff_mse_top5_share", float(np.mean(stats["mse_top5_share"])))
    if stats["p95_over_p50"]:
        logger.record("train/action_diff_abs_p95_over_p50", float(np.mean(stats["p95_over_p50"])))
    if stats["p99_over_p50"]:
        logger.record("train/action_diff_abs_p99_over_p50", float(np.mean(stats["p99_over_p50"])))
    if stats["high_stale_in_top10_mse_frac"]:
        logger.record(
            "train/high_behavior_action_diff_in_top10_mse_frac",
            float(np.mean(stats["high_stale_in_top10_mse_frac"])),
        )
    if stats["top10_mse_high_stale_frac"]:
        logger.record(
            "train/top10_mse_high_behavior_action_diff_frac",
            float(np.mean(stats["top10_mse_high_stale_frac"])),
        )


class CATSSAC(CustomSAC):
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
    ):
        self.cats_alpha = cats_alpha
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

    def _cats_lambda(self) -> float:
        completed_fraction = min(
            1.0, max(0.0, self.num_timesteps / max(1, self._total_timesteps))
        )
        return linear_ramp_up_coefficient(self.cats_alpha, completed_fraction)

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
        action_diff_stats = {key: [] for key in ["mean_dim", "p90_dim", "p95_dim", "max_dim", "p90_all"]}
        behavior_action_diff_stats = {key: [] for key in ["mean_dim", "p90_dim", "p95_dim", "max_dim", "p90_all"]}
        behavior_pair_stats = {
            key: [] for key in ["corr", "x_y_high_frac", "y_x_high_frac", "corr_dim", "x_y_high_frac_dim", "y_x_high_frac_dim"]
        }
        loss_tail_stats = {
            key: []
            for key in [
                "mse_top10_share",
                "mse_top5_share",
                "p95_over_p50",
                "p99_over_p50",
                "high_stale_in_top10_mse_frac",
                "top10_mse_high_stale_frac",
            ]
        }

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

            # CATS temporal loss
            cats_now_actions = self.policy._predict(replay_data.observations, deterministic=True).type(th.float32)
            cats_next_actions = self.policy._predict(replay_data.next_observations, deterministic=True).type(th.float32)
            action_diff = cats_now_actions - cats_next_actions
            
            loss_t = 0.5 * F.mse_loss(cats_now_actions, cats_next_actions)
            loss_t_values.append(loss_t.item())
            _append_action_diff_poc_stats(action_diff_stats, action_diff)

            # Linear Scheduling Calculation
            cats_lambda = self._cats_lambda()
            lambda_values.append(cats_lambda)
            
            reg_loss_t = cats_lambda * loss_t
            cats_loss = reg_loss_t

            q_values_pi = th.cat(self.critic(replay_data.observations, actions_pi), dim=1)
            min_qf_pi, _ = th.min(q_values_pi, dim=1, keepdim=True)
            actor_loss = (ent_coef * log_prob - min_qf_pi).mean() + cats_loss
            actor_losses.append(actor_loss.item())

            # === Observation-only logging (before optimizer step) ===
            with th.no_grad():
                obs_current_actions = self.actor(replay_data.observations, deterministic=True)
                behavior_action_diff = replay_data.actions - obs_current_actions
                _append_action_diff_poc_stats(behavior_action_diff_stats, behavior_action_diff)
                transition_staleness = behavior_action_diff.abs().mean(dim=1)
                _append_continuous_pair_poc_stats(behavior_pair_stats, action_diff.abs(), transition_staleness)
                _append_loss_tail_stats(loss_tail_stats, action_diff, transition_staleness)
                delta_a = replay_data.actions - obs_current_actions
                delta_a_norm_sq = delta_a.pow(2).mean(dim=1, keepdim=True)
                sigma_sq = ent_coef.item() ** 2 if hasattr(ent_coef, 'item') else float(ent_coef) ** 2
                weight = th.exp(-delta_a_norm_sq / (2 * sigma_sq + 1e-6))
                obs_weight_mean_values.append(weight.mean().item())

            obs_actions_g = self.actor(replay_data.observations, deterministic=True)
            obs_next_actions_g = self.actor(replay_data.next_observations, deterministic=True)
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
        _record_action_diff_poc_stats(self.logger, "action_diff", action_diff_stats)
        _record_action_diff_poc_stats(self.logger, "behavior_action_diff", behavior_action_diff_stats)
        _record_continuous_pair_poc_stats(
            self.logger,
            behavior_pair_stats,
            "action_diff_behavior_corr",
            "high_action_diff_behavior_high_frac",
            "high_behavior_action_diff_action_high_frac",
        )
        _record_loss_tail_stats(self.logger, loss_tail_stats)
        if len(obs_cos_sim_values) > 0:
            self.logger.record("train/obs_cos_sim", np.mean(obs_cos_sim_values))
        if len(obs_weight_mean_values) > 0:
            self.logger.record("train/obs_weight_mean", np.mean(obs_weight_mean_values))
