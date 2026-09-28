from typing import NamedTuple
import torch as th
import numpy as np
from gymnasium import spaces
from stable_baselines3.common.buffers import RolloutBuffer

class CustomRolloutBufferSamples(NamedTuple):
    observations: th.Tensor
    actions: th.Tensor
    old_values: th.Tensor
    old_log_prob: th.Tensor
    advantages: th.Tensor
    returns: th.Tensor
    next_observations: th.Tensor
    prev_observations: th.Tensor

class CustomRolloutBuffer(RolloutBuffer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.next_observations = None
        self.prev_observations = None

    def reset(self) -> None:
        super().reset()
        self.next_observations = np.zeros((self.buffer_size, self.n_envs, *self.obs_shape), dtype=np.float32)
        self.prev_observations = None

    def add(self, obs, action, reward, episode_start, value, log_prob, next_obs=None):
        if next_obs is not None:
            if isinstance(self.observation_space, spaces.Discrete):
                next_obs = next_obs.reshape((self.n_envs, *self.obs_shape))
            self.next_observations[self.pos] = np.array(next_obs)
        super().add(obs, action, reward, episode_start, value, log_prob)

    def compute_returns_and_advantage(self, last_values: th.Tensor, dones: np.ndarray) -> None:
        super().compute_returns_and_advantage(last_values, dones)

        # next_observations fallback: if not stored via add(), reconstruct from buffer
        if not np.any(self.next_observations):
            next_obs = np.zeros_like(self.observations)
            next_obs[:-1] = self.observations[1:]
            next_obs[-1] = self.observations[-1]
            for t in range(1, self.buffer_size):
                mask = self.episode_starts[t].astype(bool)
                if mask.any():
                    next_obs[t - 1, mask] = self.observations[t - 1, mask]
            self.next_observations = next_obs

        # prev_observations: reconstruct from adjacent buffer positions
        prev_obs = np.zeros_like(self.observations)
        prev_obs[1:] = self.observations[:-1]
        prev_obs[0] = self.observations[0]

        for t in range(1, self.buffer_size):
            mask = self.episode_starts[t].astype(bool)
            if mask.any():
                prev_obs[t, mask] = self.observations[t, mask]

        self.prev_observations = prev_obs

    def get(self, batch_size=None):
        assert self.full, ""
        indices = np.random.permutation(self.buffer_size * self.n_envs)

        if not self.generator_ready:
            _tensor_names = [
                "observations",
                "actions",
                "values",
                "log_probs",
                "advantages",
                "returns",
                "next_observations",
                "prev_observations",
            ]
            for tensor in _tensor_names:
                self.__dict__[tensor] = self.swap_and_flatten(self.__dict__[tensor])
            self.generator_ready = True

        if batch_size is None:
            batch_size = self.buffer_size * self.n_envs

        start_idx = 0
        while start_idx < self.buffer_size * self.n_envs:
            yield self._get_samples(indices[start_idx : start_idx + batch_size])
            start_idx += batch_size

    def _get_samples(self, batch_inds, env=None) -> CustomRolloutBufferSamples:
        data = (
            self.observations[batch_inds],
            self.actions[batch_inds],
            self.values[batch_inds].flatten(),
            self.log_probs[batch_inds].flatten(),
            self.advantages[batch_inds].flatten(),
            self.returns[batch_inds].flatten(),
            self.next_observations[batch_inds],
            self.prev_observations[batch_inds],
        )
        return CustomRolloutBufferSamples(*tuple(map(self.to_torch, data)))
