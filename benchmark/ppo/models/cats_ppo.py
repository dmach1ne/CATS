import numpy as np
import os
import torch as th
import math
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


def _binary_overlap_rates(x: th.Tensor, y_mask: th.Tensor) -> tuple[float, float] | None:
    x = x.detach().float().flatten()
    y_high = y_mask.detach().bool().flatten()
    if x.numel() == 0 or y_high.numel() == 0:
        return None
    x_high = x >= th.quantile(x, 0.90)
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


def _append_binary_pair_poc_stats(stats: dict[str, list], action_diff_abs: th.Tensor, transition_mask: th.Tensor) -> None:
    with th.no_grad():
        action_largeness = action_diff_abs.detach().abs().mean(dim=1)
        mask_float = transition_mask.detach().float().flatten()
        corr = _safe_corr(action_largeness, mask_float)
        overlap = _binary_overlap_rates(action_largeness, transition_mask)
        if corr is not None:
            stats["corr"].append(corr)
        if overlap is not None:
            x_y_high_frac, y_x_high_frac = overlap
            stats["x_y_high_frac"].append(x_y_high_frac)
            stats["y_x_high_frac"].append(y_x_high_frac)

        corr_dim, x_y_high_frac_dim, y_x_high_frac_dim = [], [], []
        for dim in range(action_diff_abs.shape[1]):
            dim_largeness = action_diff_abs[:, dim]
            dim_corr = _safe_corr(dim_largeness, mask_float)
            dim_overlap = _binary_overlap_rates(dim_largeness, transition_mask)
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


def _record_pair_poc_stats(
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
    stats: dict[str, list], action_diff: th.Tensor, ratio_shift: th.Tensor, clipped_mask: th.Tensor
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

        ratio_shift = ratio_shift.detach().float().flatten()
        high_ratio_shift = ratio_shift >= th.quantile(ratio_shift, 0.90)
        clipped_mask = clipped_mask.detach().bool().flatten()
        top10_mask = th.zeros_like(high_ratio_shift, dtype=th.bool)
        top10_mask[top10_idx] = True

        if top10_mask.any():
            stats["high_ratio_in_top10_mse_frac"].append(high_ratio_shift[top10_mask].float().mean().item())
            stats["clipped_in_top10_mse_frac"].append(clipped_mask[top10_mask].float().mean().item())
        if high_ratio_shift.any():
            stats["top10_mse_high_ratio_frac"].append(top10_mask[high_ratio_shift].float().mean().item())
        if clipped_mask.any():
            stats["top10_mse_clipped_frac"].append(top10_mask[clipped_mask].float().mean().item())


def _record_loss_tail_stats(logger, stats: dict[str, list]) -> None:
    if stats["mse_top10_share"]:
        logger.record("train/action_diff_mse_top10_share", float(np.mean(stats["mse_top10_share"])))
    if stats["mse_top5_share"]:
        logger.record("train/action_diff_mse_top5_share", float(np.mean(stats["mse_top5_share"])))
    if stats["p95_over_p50"]:
        logger.record("train/action_diff_abs_p95_over_p50", float(np.mean(stats["p95_over_p50"])))
    if stats["p99_over_p50"]:
        logger.record("train/action_diff_abs_p99_over_p50", float(np.mean(stats["p99_over_p50"])))
    if stats["high_ratio_in_top10_mse_frac"]:
        logger.record(
            "train/high_ratio_shift_in_top10_mse_frac",
            float(np.mean(stats["high_ratio_in_top10_mse_frac"])),
        )
    if stats["top10_mse_high_ratio_frac"]:
        logger.record(
            "train/top10_mse_high_ratio_shift_frac",
            float(np.mean(stats["top10_mse_high_ratio_frac"])),
        )
    if stats["clipped_in_top10_mse_frac"]:
        logger.record("train/clipped_in_top10_mse_frac", float(np.mean(stats["clipped_in_top10_mse_frac"])))
    if stats["top10_mse_clipped_frac"]:
        logger.record("train/top10_mse_clipped_frac", float(np.mean(stats["top10_mse_clipped_frac"])))


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

    def _caps_lambda(self) -> float:
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
        caps_lambda = self._caps_lambda()
        if record_auxiliary:
            caps_losses = []
            loss_t_values = []
            action_diff_stats = {
                key: []
                for key in ["mean_dim", "p90_dim", "p95_dim", "max_dim", "p90_all"]
            }
            ratio_shift_pair_stats = {
                key: []
                for key in [
                    "corr",
                    "x_y_high_frac",
                    "y_x_high_frac",
                    "corr_dim",
                    "x_y_high_frac_dim",
                    "y_x_high_frac_dim",
                ]
            }
            clipped_pair_stats = {
                key: []
                for key in [
                    "corr",
                    "x_y_high_frac",
                    "y_x_high_frac",
                    "corr_dim",
                    "x_y_high_frac_dim",
                    "y_x_high_frac_dim",
                ]
            }
            loss_tail_stats = {
                key: []
                for key in [
                    "mse_top10_share",
                    "mse_top5_share",
                    "p95_over_p50",
                    "p99_over_p50",
                    "high_ratio_in_top10_mse_frac",
                    "top10_mse_high_ratio_frac",
                    "clipped_in_top10_mse_frac",
                    "top10_mse_clipped_frac",
                ]
            }

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
                    _append_action_diff_poc_stats(action_diff_stats, action_diff)
                    with th.no_grad():
                        clipped_mask = (th.abs(ratio - 1) > clip_range).flatten()
                        ratio_shift = th.abs(ratio - 1).flatten()
                        _append_continuous_pair_poc_stats(
                            ratio_shift_pair_stats,
                            action_diff.abs(),
                            ratio_shift,
                        )
                        _append_binary_pair_poc_stats(
                            clipped_pair_stats,
                            action_diff.abs(),
                            clipped_mask,
                        )
                        _append_loss_tail_stats(
                            loss_tail_stats,
                            action_diff,
                            ratio_shift,
                            clipped_mask,
                        )

                caps_loss = caps_lambda * loss_t
                if record_auxiliary:
                    caps_losses.append(caps_loss.item())

                loss = policy_loss + self.ent_coef * entropy_loss + self.vf_coef * value_loss + caps_loss

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
            self.logger.record("train/cats_lambda", caps_lambda)
            _record_action_diff_poc_stats(self.logger, "action_diff", action_diff_stats)
            _record_pair_poc_stats(
                self.logger,
                ratio_shift_pair_stats,
                "action_diff_ratio_shift_corr",
                "high_action_diff_ratio_shift_high_frac",
                "high_ratio_shift_action_diff_high_frac",
            )
            _record_pair_poc_stats(
                self.logger,
                clipped_pair_stats,
                "action_diff_clip_corr",
                "high_action_diff_clipped_frac",
                "clipped_action_diff_high_frac",
            )
            _record_loss_tail_stats(self.logger, loss_tail_stats)
            if caps_losses:
                self.logger.record("train/caps_loss", np.mean(caps_losses))
        if hasattr(self.policy, "log_std"):
            self.logger.record("train/std", th.exp(self.policy.log_std).mean().item())
        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
