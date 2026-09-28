"""Paper- and released-code-compatible LipsNet++ policy module."""

from dataclasses import dataclass
from typing import Sequence

import torch as th
from torch import nn

try:
    from torch.func import jacrev, vmap
except ImportError:  # PyTorch versions used by the released implementation.
    from functorch import jacrev, vmap


@dataclass(frozen=True)
class LipsNetPlusPlusLoss:
    total: th.Tensor
    jacobian: th.Tensor
    filter: th.Tensor


def _initialize_linear(linear: nn.Linear, following: nn.Module) -> None:
    if isinstance(following, nn.ReLU):
        nn.init.kaiming_normal_(linear.weight, nonlinearity="relu")
    elif isinstance(following, nn.LeakyReLU):
        nn.init.kaiming_normal_(linear.weight, nonlinearity="leaky_relu")
    else:
        nn.init.xavier_normal_(linear.weight)


class LipsNetPlusPlus(nn.Module):
    """Fourier filter followed by a Jacobian-regularized controller.

    The filter penalty intentionally follows the released code:
    ``lambda_h * sum(filter_kernel ** 2)``. This differs from the unsquared
    Frobenius norm written in the paper.
    """

    def __init__(
        self,
        obs_dim: int,
        history_len: int,
        action_dim: int,
        hidden_sizes: Sequence[int],
        lambda_k: float,
        lambda_h: float,
        kernel_scale: float = 0.02,
        activation_fn: type[nn.Module] = nn.ReLU,
        output_activation: type[nn.Module] = nn.Tanh,
        regularization_output_tanh: bool = False,
    ) -> None:
        super().__init__()
        if obs_dim <= 0 or history_len <= 0 or action_dim <= 0:
            raise ValueError("obs_dim, history_len, and action_dim must be positive")
        if not hidden_sizes:
            raise ValueError("LipsNet++ requires at least one hidden layer")

        self.obs_dim = int(obs_dim)
        self.history_len = int(history_len)
        self.action_dim = int(action_dim)
        self.hidden_sizes = tuple(int(size) for size in hidden_sizes)
        self.lambda_k = float(lambda_k)
        self.lambda_h = float(lambda_h)
        self.kernel_scale = float(kernel_scale)
        self.activation_fn = activation_fn
        self.regularization_output_tanh = bool(regularization_output_tanh)

        real = th.ones(self.history_len, self.obs_dim // 2 + 1, 1)
        imaginary = th.randn_like(real) * self.kernel_scale
        self.filter_kernel = nn.Parameter(th.cat([real, imaginary], dim=2))

        hidden_layers: list[nn.Module] = []
        input_dim = self.obs_dim
        for hidden_dim in self.hidden_sizes:
            linear = nn.Linear(input_dim, hidden_dim)
            activation = self.activation_fn()
            _initialize_linear(linear, activation)
            hidden_layers.extend([linear, activation])
            input_dim = hidden_dim
        self.hidden_net = nn.Sequential(*hidden_layers)

        self.output_layer = nn.Linear(input_dim, self.action_dim)
        self.output_activation = output_activation()
        _initialize_linear(self.output_layer, self.output_activation)

    def _as_history(self, observations: th.Tensor) -> th.Tensor:
        if observations.ndim == 2:
            expected = self.history_len * self.obs_dim
            if observations.shape[1] != expected:
                raise ValueError(
                    f"flat history has width {observations.shape[1]}, expected {expected}"
                )
            observations = observations.reshape(-1, self.history_len, self.obs_dim)
        if observations.ndim != 3:
            raise ValueError(
                "LipsNet++ observations must have shape "
                "(batch, history_len, obs_dim)"
            )
        if observations.shape[1:] != (self.history_len, self.obs_dim):
            raise ValueError(
                f"history shape {tuple(observations.shape[1:])} does not match "
                f"({self.history_len}, {self.obs_dim})"
            )
        return observations

    def filter_history(self, observations: th.Tensor) -> th.Tensor:
        history = self._as_history(observations)
        spectrum = th.fft.rfft2(
            history,
            s=(self.history_len, self.obs_dim),
            dim=(1, 2),
            norm="ortho",
        )
        kernel = th.view_as_complex(self.filter_kernel)
        filtered_spectrum = spectrum * kernel
        return th.fft.irfft2(
            filtered_spectrum,
            s=(self.history_len, self.obs_dim),
            dim=(1, 2),
            norm="ortho",
        )

    def filtered_current(self, observations: th.Tensor) -> th.Tensor:
        return self.filter_history(observations)[:, 0, :]

    def controller(self, filtered_observation: th.Tensor) -> th.Tensor:
        latent = self.hidden_net(filtered_observation)
        return self.output_activation(self.output_layer(latent))

    def _regularized_controller(self, filtered_observation: th.Tensor) -> th.Tensor:
        output = self.controller(filtered_observation)
        if self.regularization_output_tanh:
            output = th.tanh(output)
        return output

    def forward_with_latent(
        self, observations: th.Tensor
    ) -> tuple[th.Tensor, th.Tensor]:
        filtered_observation = self.filtered_current(observations)
        latent = self.hidden_net(filtered_observation)
        output = self.output_activation(self.output_layer(latent))
        return output, latent

    def forward(self, observations: th.Tensor) -> th.Tensor:
        output, _ = self.forward_with_latent(observations)
        return output

    def regularization_loss(
        self, observations: th.Tensor
    ) -> LipsNetPlusPlusLoss:
        return self.regularization_loss_from_filtered(
            self.filtered_current(observations)
        )

    def regularization_loss_from_filtered(
        self, filtered_observation: th.Tensor
    ) -> LipsNetPlusPlusLoss:
        if filtered_observation.ndim != 2:
            raise ValueError(
                "filtered observations must have shape (batch, obs_dim)"
            )
        if filtered_observation.shape[1] != self.obs_dim:
            raise ValueError(
                f"filtered observation width {filtered_observation.shape[1]} "
                f"does not match {self.obs_dim}"
            )
        filtered_observation = filtered_observation.detach()
        jacobian = vmap(jacrev(self._regularized_controller))(filtered_observation)
        jacobian_norm = th.norm(jacobian, 2, dim=(1, 2)).mean()

        jacobian_loss = self.lambda_k * jacobian_norm
        filter_loss = self.lambda_h * self.filter_kernel.square().sum()
        return LipsNetPlusPlusLoss(
            total=jacobian_loss + filter_loss,
            jacobian=jacobian_loss,
            filter=filter_loss,
        )
