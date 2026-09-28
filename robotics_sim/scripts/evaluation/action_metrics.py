from __future__ import annotations

import numpy as np


def arm_smoothness(actions: np.ndarray, fs: float) -> float:
    values = np.asarray(actions, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] < 7:
        raise ValueError(
            "arm smoothness requires actions with shape (time, >=7) for the seven arm coordinates"
        )
    return frequency_weighted_smoothness(values[:, :7], fs=fs)


def frequency_weighted_smoothness(actions: np.ndarray, fs: float) -> float:
    values = np.asarray(actions, dtype=np.float64)
    if values.ndim == 1:
        values = values[:, None]
    if values.ndim != 2:
        raise ValueError(f"actions must have shape (time, action_dim), got {values.shape}")
    sample_count = values.shape[0]
    if sample_count < 2:
        return 0.0
    magnitudes = np.abs(np.fft.fft(values, axis=0)[: sample_count // 2])
    frequencies = np.fft.fftfreq(sample_count, d=1.0 / fs)[: sample_count // 2, None]
    per_coordinate = (2.0 / (sample_count * fs)) * np.sum(
        frequencies * magnitudes, axis=0
    )
    return float(np.mean(per_coordinate))


def mechanical_energy(
    applied_torque: np.ndarray,
    joint_velocity: np.ndarray,
    dt: float,
) -> tuple[float, float]:
    torque = np.asarray(applied_torque, dtype=np.float64)
    velocity = np.asarray(joint_velocity, dtype=np.float64)
    if torque.shape != velocity.shape or torque.ndim != 2:
        raise ValueError(f"torque and velocity must share (time, joint) shape, got {torque.shape}, {velocity.shape}")
    if dt <= 0.0:
        raise ValueError("dt must be positive")
    power = np.abs(torque * velocity).sum(axis=1)
    energy = float(power.sum() * dt)
    duration = torque.shape[0] * dt
    mean_power = energy / duration if duration > 0.0 else 0.0
    return energy, mean_power


def gripper_switch_count(gripper_actions: np.ndarray) -> int:
    values = np.asarray(gripper_actions, dtype=np.float64).reshape(-1)
    if values.size < 2:
        return 0
    binary_commands = values > 0.0
    return int(np.count_nonzero(binary_commands[1:] != binary_commands[:-1]))
