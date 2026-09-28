from __future__ import annotations

import math

import torch
from rl_games.algos_torch import a2c_continuous, torch_ext
from rl_games.common import common_losses
from rl_games.common.a2c_common import swap_and_flatten01

from .caps import perturb_goal_fixed_scale_observations, spatial_action_loss
from .cats import (
    resolve_action_layout,
    resolve_next_observations,
    temporal_action_loss,
    transition_indices,
)
from .configured_adaptive_scheduler import ConfiguredAdaptiveSchedulerMixin
from .finite_gradient_guard import FiniteGradientGuardMixin
from .visual_proprio_network import policy_mean


class CapsA2CAgent(
    ConfiguredAdaptiveSchedulerMixin,
    FiniteGradientGuardMixin,
    a2c_continuous.A2CAgent,
):
    """Stock continuous PPO with fixed temporal and spatial CAPS penalties."""

    def __init__(self, base_name, params):
        super().__init__(base_name, params)
        caps_cfg = self.config.get("caps", {})
        required = ("lambda_t", "lambda_s", "sigma")
        missing = [key for key in required if key not in caps_cfg]
        if missing:
            raise ValueError(f"CAPS requires explicit config values: {', '.join(missing)}")
        self.caps_lambda_t = float(caps_cfg["lambda_t"])
        self.caps_lambda_s = float(caps_cfg["lambda_s"])
        self.caps_sigma = float(caps_cfg["sigma"])
        self.caps_preserve_target_overlay = caps_cfg.get("preserve_target_overlay", False)
        if not isinstance(self.caps_preserve_target_overlay, bool):
            raise ValueError("CAPS preserve_target_overlay must be a boolean")
        self.caps_overlay_projection = None
        if self.caps_preserve_target_overlay:
            from er_isaac.tasks.manager_based.er_isaac.mdp.target_overlay_geometry import target_overlay_mask

            self._caps_overlay_mask = target_overlay_mask
            self.caps_overlay_projection = caps_cfg.get("overlay_projection")
            if not self.caps_overlay_projection:
                raise ValueError("CAPS overlay preservation requires overlay_projection")
            image_shape = self.model.a2c_network.image_shape
            projection_shape = (
                self.caps_overlay_projection.get("image_height"),
                self.caps_overlay_projection.get("image_width"),
            )
            if projection_shape != image_shape[:2]:
                raise ValueError("CAPS overlay projection must match the network image_shape")
            if self.normalize_input:
                raise ValueError("CAPS overlay preservation requires unnormalized raw observations")
        if not math.isfinite(self.caps_lambda_t) or self.caps_lambda_t < 0.0:
            raise ValueError("CAPS lambda_t must be finite and non-negative")
        if not math.isfinite(self.caps_lambda_s) or self.caps_lambda_s < 0.0:
            raise ValueError("CAPS lambda_s must be finite and non-negative")
        if not math.isfinite(self.caps_sigma) or self.caps_sigma <= 0.0:
            raise ValueError("CAPS sigma must be finite and positive")
        self.caps_arm_dims, self.caps_gripper_action_index = resolve_action_layout(
            caps_cfg, self.actions_num
        )
        if self.is_rnn:
            raise NotImplementedError("CAPS transition lookup supports non-recurrent policies only")
        self._caps_rollout_obs = None
        self._caps_final_obs = None
        self._last_caps_log_epoch = -1

    def play_steps(self):
        batch_dict = super().play_steps()
        rollout_dones = self.experience_buffer.tensor_dict["dones"].bool()
        horizon, num_envs = rollout_dones.shape[:2]

        transition_dones = torch.empty_like(rollout_dones)
        transition_dones[:-1] = rollout_dones[1:]
        transition_dones[-1] = self.dones.bool()
        next_indices, final_envs = transition_indices(horizon, num_envs, rollout_dones.device)

        batch_dict["caps_next_index"] = next_indices
        batch_dict["caps_final_env"] = final_envs
        batch_dict["caps_valid"] = swap_and_flatten01(~transition_dones)
        self._caps_rollout_obs = batch_dict["obses"]
        self._caps_final_obs = self.obs["obs"].detach()
        return batch_dict

    def prepare_dataset(self, batch_dict):
        super().prepare_dataset(batch_dict)
        values = self.dataset.values_dict
        values["caps_next_index"] = batch_dict["caps_next_index"]
        values["caps_final_env"] = batch_dict["caps_final_env"]
        values["caps_valid"] = batch_dict["caps_valid"]

    def calc_gradients(self, input_dict):
        value_preds_batch = input_dict["old_values"]
        old_action_log_probs_batch = input_dict["old_logp_actions"]
        advantage = input_dict["advantages"]
        old_mu_batch = input_dict["mu"]
        old_sigma_batch = input_dict["sigma"]
        return_batch = input_dict["returns"]
        actions_batch = input_dict["actions"]
        obs_batch = self._preproc_obs(input_dict["obs"])

        if self._caps_rollout_obs is None or self._caps_final_obs is None:
            raise RuntimeError("CAPS rollout observations were not initialized")
        next_obs_batch = resolve_next_observations(
            self._caps_rollout_obs,
            self._caps_final_obs,
            input_dict["caps_next_index"],
            input_dict["caps_final_env"],
        )
        next_obs_batch = self._preproc_obs(next_obs_batch)
        overlay_mask = None
        if self.caps_preserve_target_overlay:
            # Use each stored sample's goal, not the live environment's latest command.
            overlay_mask = self._caps_overlay_mask(obs_batch[:, -3:], **self.caps_overlay_projection)
        perturbed_obs_batch = perturb_goal_fixed_scale_observations(
            obs_batch,
            sigma=self.caps_sigma,
            action_dims=self.actions_num,
            overlay_mask=overlay_mask,
        )

        lr_mul = 1.0
        curr_e_clip = self.e_clip
        batch_dict = {"is_train": True, "prev_actions": actions_batch, "obs": obs_batch}

        rnn_masks = None
        with torch.cuda.amp.autocast(enabled=self.mixed_precision):
            res_dict = self.model(batch_dict)
            next_mu = policy_mean(self.model, next_obs_batch, actions_batch)
            perturbed_mu = policy_mean(self.model, perturbed_obs_batch, actions_batch)
            action_log_probs = res_dict["prev_neglogp"]
            values = res_dict["values"]
            entropy = res_dict["entropy"]
            mu = res_dict["mus"]
            sigma = res_dict["sigmas"]

            a_loss = self.actor_loss_func(
                old_action_log_probs_batch, action_log_probs, advantage, self.ppo, curr_e_clip
            )
            if self.has_value_loss:
                c_loss = common_losses.critic_loss(
                    self.model, value_preds_batch, values, curr_e_clip, return_batch, self.clip_value
                )
            else:
                c_loss = torch.zeros(1, device=self.ppo_device)
            if self.bound_loss_type == "regularisation":
                b_loss = self.reg_loss(mu)
            elif self.bound_loss_type == "bound":
                b_loss = self.bound_loss(mu)
            else:
                b_loss = torch.zeros(1, device=self.ppo_device)

            losses, _ = torch_ext.apply_masks(
                [a_loss.unsqueeze(1), c_loss, entropy.unsqueeze(1), b_loss.unsqueeze(1)], rnn_masks
            )
            a_loss, c_loss, entropy, b_loss = losses
            ppo_loss = (
                a_loss
                + 0.5 * c_loss * self.critic_coef
                - entropy * self.entropy_coef
                + b_loss * self.bounds_loss_coef
            )
            temporal_loss, temporal_per_dim = temporal_action_loss(
                mu, next_mu, input_dict["caps_valid"], arm_dims=self.caps_arm_dims
            )
            spatial_loss, spatial_per_dim = spatial_action_loss(
                mu, perturbed_mu, arm_dims=self.caps_arm_dims
            )
            caps_loss = self.caps_lambda_t * temporal_loss + self.caps_lambda_s * spatial_loss
            loss = ppo_loss + caps_loss

            aux_loss = self.model.get_aux_loss()
            self.aux_loss_dict = {}
            if aux_loss is not None:
                for key, value in aux_loss.items():
                    loss += value
                    self.aux_loss_dict.setdefault(key, []).append(value.detach())

            if self.multi_gpu:
                self.optimizer.zero_grad()
            else:
                for param in self.model.parameters():
                    param.grad = None

        self.scaler.scale(loss).backward()
        self.trancate_gradients_and_step()

        with torch.no_grad():
            kl_dist = torch_ext.policy_kl(
                mu.detach(), sigma.detach(), old_mu_batch, old_sigma_batch, True
            )

        self.diagnostics.mini_batch(
            self,
            {
                "values": value_preds_batch,
                "returns": return_batch,
                "new_neglogp": action_log_probs,
                "old_neglogp": old_action_log_probs_batch,
                "masks": rnn_masks,
            },
            curr_e_clip,
            0,
        )

        if self.writer is not None and self._last_caps_log_epoch != self.epoch_num:
            self.writer.add_scalar("caps/lambda_t", self.caps_lambda_t, self.frame)
            self.writer.add_scalar("caps/lambda_s", self.caps_lambda_s, self.frame)
            self.writer.add_scalar("caps/sigma", self.caps_sigma, self.frame)
            self.writer.add_scalar("caps/preserve_target_overlay", int(self.caps_preserve_target_overlay), self.frame)
            self.writer.add_scalar("caps/temporal_loss", temporal_loss.detach(), self.frame)
            self.writer.add_scalar("caps/spatial_loss", spatial_loss.detach(), self.frame)
            self.writer.add_scalar("caps/weighted_loss", caps_loss.detach(), self.frame)
            for dim, (temporal_dim, spatial_dim) in enumerate(
                zip(temporal_per_dim.detach(), spatial_per_dim.detach())
            ):
                self.writer.add_scalar(f"caps/arm_dim_{dim}_temporal_loss", temporal_dim, self.frame)
                self.writer.add_scalar(f"caps/arm_dim_{dim}_spatial_loss", spatial_dim, self.frame)
            self._last_caps_log_epoch = self.epoch_num

        self.train_result = (
            a_loss,
            c_loss,
            entropy,
            kl_dist,
            self.last_lr,
            lr_mul,
            mu.detach(),
            sigma.detach(),
            b_loss,
        )
