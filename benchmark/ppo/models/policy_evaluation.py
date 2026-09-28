from __future__ import annotations

from typing import Optional

import torch as th
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.type_aliases import PyTorchObs


def evaluate_actions_with_context(
    policy: ActorCriticPolicy,
    observations: PyTorchObs,
    actions: th.Tensor,
    *,
    deterministic_action: bool = False,
) -> tuple[
    th.Tensor,
    th.Tensor,
    Optional[th.Tensor],
    Optional[th.Tensor],
    object,
    th.Tensor,
]:
    """Mirror SB3 evaluate_actions while exposing reusable actor context."""
    features = policy.extract_features(observations)
    if policy.share_features_extractor:
        latent_pi, latent_vf = policy.mlp_extractor(features)
    else:
        pi_features, vf_features = features
        latent_pi = policy.mlp_extractor.forward_actor(pi_features)
        latent_vf = policy.mlp_extractor.forward_critic(vf_features)

    distribution = policy._get_action_dist_from_latent(latent_pi)
    log_prob = distribution.log_prob(actions)
    values = policy.value_net(latent_vf)
    entropy = distribution.entropy()
    deterministic = (
        distribution.get_actions(deterministic=True)
        if deterministic_action
        else None
    )
    return values, log_prob, entropy, deterministic, distribution, latent_pi
