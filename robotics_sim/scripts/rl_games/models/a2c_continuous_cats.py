from __future__ import annotations

import torch
from rl_games.algos_torch import a2c_continuous, torch_ext
from rl_games.common import common_losses
from rl_games.common.a2c_common import swap_and_flatten01

from .cats import (
    linear_ramp,
    resolve_next_observations,
    resolve_action_layout,
    temporal_action_loss,
    transition_indices,
)
from .configured_adaptive_scheduler import ConfiguredAdaptiveSchedulerMixin
from .finite_gradient_guard import FiniteGradientGuardMixin
from .visual_proprio_network import policy_mean


class CatsA2CAgent(
    ConfiguredAdaptiveSchedulerMixin,
    FiniteGradientGuardMixin,
    a2c_continuous.A2CAgent,
):
    """Stock continuous PPO with a transition-conditioned policy-mean penalty."""

    def __init__(self, base_name, params):
        super().__init__(base_name, params)
        cats_cfg = self.config.get("cats", {})
        self.cats_alpha_max = float(cats_cfg.get("alpha_max", 1.0))
        self.cats_arm_dims, self.cats_gripper_action_index = resolve_action_layout(
            cats_cfg, self.actions_num
        )
        if self.is_rnn:
            raise NotImplementedError("CATS transition lookup currently supports non-recurrent policies only")
        self._cats_rollout_obs = None
        self._cats_final_obs = None
        self._cats_gripper_switch_rate = None
        self._last_cats_log_epoch = -1

    def play_steps(self):
        batch_dict = super().play_steps()
        rollout_dones = self.experience_buffer.tensor_dict["dones"].bool()
        horizon, num_envs = rollout_dones.shape[:2]

        transition_dones = torch.empty_like(rollout_dones)
        transition_dones[:-1] = rollout_dones[1:]
        transition_dones[-1] = self.dones.bool()
        next_indices, final_envs = transition_indices(horizon, num_envs, rollout_dones.device)

        batch_dict["cats_next_index"] = next_indices
        batch_dict["cats_final_env"] = final_envs
        batch_dict["cats_valid"] = swap_and_flatten01(~transition_dones)
        self._cats_rollout_obs = batch_dict["obses"]
        self._cats_final_obs = self.obs["obs"].detach()
        return batch_dict

    def prepare_dataset(self, batch_dict):
        super().prepare_dataset(batch_dict)
        values = self.dataset.values_dict
        values["cats_next_index"] = batch_dict["cats_next_index"]
        values["cats_final_env"] = batch_dict["cats_final_env"]
        values["cats_valid"] = batch_dict["cats_valid"]
        actions = batch_dict["actions"].reshape(
            self.num_actors * self.num_agents, self.horizon_length, -1
        )
        valid = batch_dict["cats_valid"].reshape(
            self.num_actors * self.num_agents, self.horizon_length
        )
        if self.cats_gripper_action_index is None:
            self._cats_gripper_switch_rate = None
        else:
            switch_pairs = valid[:, :-1]
            gripper_index = self.cats_gripper_action_index
            switches = (actions[:, 1:, gripper_index] > 0.0) != (
                actions[:, :-1, gripper_index] > 0.0
            )
            self._cats_gripper_switch_rate = (
                float(switches[switch_pairs].float().mean()) if switch_pairs.any() else 0.0
            )

    def _current_cats_alpha(self) -> float:
        total_steps = max((self.max_epochs - 1) * self.batch_size, 1)
        return linear_ramp(self.frame, total_steps, self.cats_alpha_max)

    def calc_gradients(self, input_dict):
        value_preds_batch = input_dict["old_values"]
        old_action_log_probs_batch = input_dict["old_logp_actions"]
        advantage = input_dict["advantages"]
        old_mu_batch = input_dict["mu"]
        old_sigma_batch = input_dict["sigma"]
        return_batch = input_dict["returns"]
        actions_batch = input_dict["actions"]
        obs_batch = self._preproc_obs(input_dict["obs"])

        if self._cats_rollout_obs is None or self._cats_final_obs is None:
            raise RuntimeError("CATS rollout observations were not initialized")
        next_obs_batch = resolve_next_observations(
            self._cats_rollout_obs,
            self._cats_final_obs,
            input_dict["cats_next_index"],
            input_dict["cats_final_env"],
        )
        next_obs_batch = self._preproc_obs(next_obs_batch)

        lr_mul = 1.0
        curr_e_clip = self.e_clip
        batch_dict = {"is_train": True, "prev_actions": actions_batch, "obs": obs_batch}

        rnn_masks = None
        with torch.cuda.amp.autocast(enabled=self.mixed_precision):
            res_dict = self.model(batch_dict)
            next_mu = policy_mean(self.model, next_obs_batch, actions_batch)
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
            cats_loss, cats_per_dim = temporal_action_loss(
                mu,
                next_mu,
                input_dict["cats_valid"],
                arm_dims=self.cats_arm_dims,
            )
            cats_alpha = self._current_cats_alpha()
            loss = ppo_loss + cats_alpha * cats_loss
            valid_transitions = input_dict["cats_valid"].bool()
            if valid_transitions.any():
                arm_abs_diff = (
                    mu[valid_transitions, : self.cats_arm_dims]
                    - next_mu[valid_transitions, : self.cats_arm_dims]
                ).abs().mean(dim=0)
            else:
                arm_abs_diff = mu[:1, : self.cats_arm_dims].sum(dim=0) * 0.0

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

        if self.writer is not None and self._last_cats_log_epoch != self.epoch_num:
            self.writer.add_scalar("cats/alpha", cats_alpha, self.frame)
            self.writer.add_scalar("cats/unweighted_loss", cats_loss.detach(), self.frame)
            self.writer.add_scalar("cats/weighted_loss", cats_alpha * cats_loss.detach(), self.frame)
            self.writer.add_scalar("cats/arm_abs_diff", arm_abs_diff.detach().mean(), self.frame)
            if self._cats_gripper_switch_rate is not None:
                self.writer.add_scalar(
                    "cats/gripper_switch_rate", self._cats_gripper_switch_rate, self.frame
                )
            for dim, (dim_loss, dim_diff) in enumerate(
                zip(cats_per_dim.detach(), arm_abs_diff.detach())
            ):
                self.writer.add_scalar(f"cats/arm_dim_{dim}_loss", dim_loss, self.frame)
                self.writer.add_scalar(f"cats/arm_dim_{dim}_abs_diff", dim_diff, self.frame)
            self._last_cats_log_epoch = self.epoch_num

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
