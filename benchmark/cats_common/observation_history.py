"""Vector-environment observation history for LipsNet++."""

from typing import Optional

import numpy as np
from gymnasium import spaces
from stable_baselines3.common.vec_env import VecEnv, VecEnvWrapper
from stable_baselines3.common.vec_env.base_vec_env import (
    VecEnvObs,
    VecEnvStepReturn,
)


class ObservationHistoryVecEnv(VecEnvWrapper):
    """Prepend the current observation to a zero-padded N-step history."""

    def __init__(self, venv: VecEnv, history_len: int) -> None:
        if history_len <= 0:
            raise ValueError("history_len must be positive")
        if not isinstance(venv.observation_space, spaces.Box):
            raise TypeError("ObservationHistoryVecEnv only supports Box observations")
        if len(venv.observation_space.shape) != 1:
            raise ValueError("LipsNet++ currently requires vector observations")

        self.history_len = int(history_len)
        self.base_observation_space = venv.observation_space
        low = np.repeat(
            self.base_observation_space.low[None, :], self.history_len, axis=0
        )
        high = np.repeat(
            self.base_observation_space.high[None, :], self.history_len, axis=0
        )
        observation_space = spaces.Box(
            low=low,
            high=high,
            dtype=self.base_observation_space.dtype,
        )
        super().__init__(venv, observation_space=observation_space)
        self._history = np.zeros(
            (self.num_envs, *observation_space.shape),
            dtype=observation_space.dtype,
        )

    def _reset_history(self, observations: np.ndarray) -> np.ndarray:
        self._history.fill(0)
        self._history[:, 0, :] = observations
        return self._history.copy()

    def reset(self) -> VecEnvObs:
        observations = self.venv.reset()
        if not isinstance(observations, np.ndarray):
            raise TypeError("LipsNet++ history wrapper requires array observations")
        return self._reset_history(observations)

    def _advanced_history(
        self, env_index: int, next_observation: np.ndarray
    ) -> np.ndarray:
        advanced = np.zeros_like(self._history[env_index])
        advanced[0] = next_observation
        advanced[1:] = self._history[env_index, :-1]
        return advanced

    def step_wait(self) -> VecEnvStepReturn:
        observations, rewards, dones, infos = self.venv.step_wait()
        if not isinstance(observations, np.ndarray):
            raise TypeError("LipsNet++ history wrapper requires array observations")

        next_history = np.zeros_like(self._history)
        for env_index in range(self.num_envs):
            if dones[env_index]:
                terminal_observation: Optional[np.ndarray] = infos[env_index].get(
                    "terminal_observation"
                )
                if terminal_observation is not None:
                    infos[env_index]["terminal_observation"] = self._advanced_history(
                        env_index, terminal_observation
                    )
                next_history[env_index, 0] = observations[env_index]
            else:
                next_history[env_index] = self._advanced_history(
                    env_index, observations[env_index]
                )

        self._history[...] = next_history
        return self._history.copy(), rewards, dones, infos
