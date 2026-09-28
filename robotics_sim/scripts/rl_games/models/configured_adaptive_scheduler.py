from __future__ import annotations

import math


class ConfiguredAdaptiveScheduler:
    """RL-Games adaptive scheduler with configurable bounds and multiplier."""

    def __init__(
        self,
        kl_threshold: float,
        min_lr: float = 1.0e-6,
        max_lr: float = 1.0e-2,
        lr_multiplier: float = 1.5,
    ):
        self.kl_threshold = self._positive_finite("kl_threshold", kl_threshold)
        self.min_lr = self._positive_finite("min_lr", min_lr)
        self.max_lr = self._positive_finite("max_lr", max_lr)
        self.lr_multiplier = self._positive_finite("lr_multiplier", lr_multiplier)
        if self.max_lr < self.min_lr:
            raise ValueError("max_lr must be greater than or equal to min_lr")
        if self.lr_multiplier <= 1.0:
            raise ValueError("lr_multiplier must be greater than 1")

    @staticmethod
    def _positive_finite(name: str, value: float) -> float:
        value = float(value)
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError(f"{name} must be finite and greater than zero")
        return value

    def update(self, current_lr, entropy_coef, epoch, frames, kl_dist, **kwargs):
        del epoch, frames, kwargs
        learning_rate = float(current_lr)
        kl_value = float(kl_dist)
        if kl_value > 2.0 * self.kl_threshold:
            learning_rate = max(learning_rate / self.lr_multiplier, self.min_lr)
        elif kl_value < 0.5 * self.kl_threshold:
            learning_rate = min(learning_rate * self.lr_multiplier, self.max_lr)
        return learning_rate, entropy_coef


class ConfiguredAdaptiveSchedulerMixin:
    """Install the configured scheduler after the RL-Games agent initializes."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if not self.is_adaptive_lr:
            return

        self.scheduler = ConfiguredAdaptiveScheduler(
            kl_threshold=self.config.get("kl_threshold", 0.008),
            min_lr=self.config.get("min_lr", 1.0e-6),
            max_lr=self.config.get("max_lr", 1.0e-2),
            lr_multiplier=self.config.get("lr_multiplier", 1.5),
        )
        print(
            "[ADAPTIVE-SCHEDULER] "
            f"schedule_type={self.config.get('schedule_type', 'legacy')} "
            f"kl_threshold={self.scheduler.kl_threshold:g} "
            f"min_lr={self.scheduler.min_lr:g} "
            f"max_lr={self.scheduler.max_lr:g} "
            f"lr_multiplier={self.scheduler.lr_multiplier:g}",
            flush=True,
        )
