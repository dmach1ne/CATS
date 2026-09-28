from __future__ import annotations

import torch


class FiniteGradientGuardMixin:
    """Skip optimizer updates whose gradient norm is not finite."""

    def trancate_gradients_and_step(self):
        if not self.config.get("finite_gradient_guard", False) or getattr(
            self, "mixed_precision", False
        ):
            return super().trancate_gradients_and_step()

        if self.truncate_grads:
            self.scaler.unscale_(self.optimizer)
            gradients = [parameter.grad for parameter in self.model.parameters() if parameter.grad is not None]
            if gradients:
                total_norm = torch.norm(
                    torch.stack([gradient.detach().norm(2) for gradient in gradients]),
                    2,
                )
                if not torch.isfinite(total_norm):
                    self._finite_gradient_skip_count = getattr(
                        self, "_finite_gradient_skip_count", 0
                    ) + 1
                    if self._finite_gradient_skip_count <= 5:
                        affected = [
                            name
                            for name, parameter in self.model.named_parameters()
                            if parameter.grad is not None
                            and not torch.isfinite(parameter.grad).all()
                        ]
                        print(
                            "[FINITE-GRAD-GUARD] "
                            f"skip #{self._finite_gradient_skip_count} at epoch {self.epoch_num}; "
                            f"non-finite gradients: {affected[:8]}",
                            flush=True,
                        )
                    self.optimizer.zero_grad(set_to_none=True)
                    return

        return super().trancate_gradients_and_step()
