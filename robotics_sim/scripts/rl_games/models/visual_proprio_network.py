from __future__ import annotations

from collections.abc import Sequence
import math

import torch
from torch import nn
from torch.nn import functional as F

from rl_games.algos_torch import model_builder
from rl_games.algos_torch.network_builder import A2CBuilder, NetworkBuilder


LIFT_NETWORK_NAME = "visual_lift_proprio_actor_critic"
REACH_NETWORK_NAME = "visual_reach_proprio_actor_critic"
IMAGE_SHAPE = (84, 84, 3)
IMAGE_SIZE = 84 * 84 * 3


def policy_mean(model, observation, previous_actions):
    """Differentiable auxiliary policy forward, preserving wrapper normalization."""
    network = model.a2c_network
    actor = getattr(network, "forward_actor", None)
    if actor is not None:
        return actor({"obs": model.norm_obs(observation)})
    if (type(network) is A2CBuilder.Network and not network.has_rnn
            and network.is_continuous and not network.central_value):
        # Mirror stock feed-forward A2CBuilder's actor path, including NHWC conversion.
        obs = model.norm_obs(observation)
        if network.has_cnn and network.permute_input and obs.ndim == 4:
            obs = obs.permute(0, 3, 1, 2)
        features = network.actor_cnn(obs).flatten(1)
        return network.mu_act(network.mu(network.actor_mlp(features)))
    # Unknown/custom architectures retain their existing wrapper behavior.
    return model({"obs": observation, "is_train": True,
                  "prev_actions": previous_actions})["mus"]


def _as_tuple(values: Sequence[int], *, name: str, length: int) -> tuple[int, ...]:
    result = tuple(int(value) for value in values)
    if len(result) != length:
        raise ValueError(f"{name} must contain {length} entries, got {len(result)}")
    return result


def _conv_output_size(size: int, kernel: int, stride: int, padding: int) -> int:
    return (size + 2 * padding - kernel) // stride + 1


class VisualProprioBuilder(NetworkBuilder):
    """Shared or separate image-plus-proprio actor-critic for Visual Reach and Lift."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.params: dict = {}

    def load(self, params):
        self.params = params

    def build(self, name, **kwargs):
        return self.Network(self.params, **kwargs)

    class Network(NetworkBuilder.BaseNetwork):
        def __init__(self, params: dict, **kwargs):
            actions_num = int(kwargs.pop("actions_num"))
            input_shape = tuple(kwargs.pop("input_shape"))
            self.value_size = int(kwargs.pop("value_size", 1))
            self.num_seqs = int(kwargs.pop("num_seqs", 1))
            super().__init__()

            self.image_shape = tuple(params.get("image_shape", IMAGE_SHAPE))
            self.proprio_size = int(params["proprio_size"])
            self.image_size = math.prod(self.image_shape)
            self.observation_size = self.image_size + self.proprio_size
            if self.image_shape not in (IMAGE_SHAPE, (128, 128, 3)):
                raise ValueError(
                    f"Expected image_shape=(84, 84, 3) or (128, 128, 3), got {self.image_shape}"
                )
            if self.proprio_size <= 0:
                raise ValueError("proprio_size must be positive")
            if input_shape != (self.observation_size,):
                raise ValueError(
                    f"Expected flattened observation shape {(self.observation_size,)}, "
                    f"got {input_shape}"
                )
            self.separate = bool(params.get("separate", False))
            fusion = params.get("fusion", "raw_concat")
            self.purelift_gripper_width = fusion == "compact_purelift_gripper_width"
            self.compact = fusion in {"compact_proprio", "compact_proprio_goal_fixed_scale", "compact_purelift_gripper_width"}
            if fusion not in {"raw_concat", "compact_proprio", "compact_proprio_goal_fixed_scale", "compact_purelift_gripper_width"}:
                raise ValueError("Unknown proprio fusion")
            self.goal_fixed_scale = bool(params.get("proprio_fixed_scale", False))
            if (fusion in {"compact_proprio_goal_fixed_scale", "compact_purelift_gripper_width"}) != self.goal_fixed_scale:
                raise ValueError("GoalFixedScale fusion requires explicit proprio_fixed_scale")
            if self.purelift_gripper_width:
                if actions_num != 8 or self.proprio_size != 24:
                    raise ValueError("PureLiftGripperWidth requires 24-D proprio and 8 actions")
                self.register_buffer("proprio_scale", torch.tensor(
                    [1.0 / math.pi] * 7 + [0.5] * 7 + [1.0] * 8 + [2.0, 12.5],
                    dtype=torch.float32,
                ))
            elif self.goal_fixed_scale:
                if actions_num not in (7, 8) or self.proprio_size != 18 + actions_num:
                    raise ValueError("GoalFixedScale requires 25-D Reach or 26-D Lift proprio")
                self.register_buffer("proprio_scale", torch.tensor(
                    [1.0 / math.pi] * 7 + [0.5] * 7 + [1.0] * actions_num + [2.0] * 4,
                    dtype=torch.float32,
                ))
            if self.compact and params.get("normalise_channels", True):
                raise ValueError("Compact proprio requires features without LayerNorm")

            self.normalise_channels = bool(params.get("normalise_channels", True))
            cnn_cfg = params["cnn"]
            activation = cnn_cfg.get("activation", "relu")
            conv_specs = self._parse_conv_specs(cnn_cfg)
            pooling = cnn_cfg.get("pooling", "adaptive_avg")
            if pooling not in {"adaptive_avg", "flatten"}:
                raise ValueError(f"Unsupported visual pooling mode: {pooling}")

            self.actor_cnn = self._build_encoder(conv_specs, activation, pooling)
            if self.separate:
                self.critic_cnn = self._build_encoder(conv_specs, activation, pooling)
            else:
                self.critic_cnn = None

            cnn_output_size = self._cnn_output_size(conv_specs, pooling)
            if self.image_shape != IMAGE_SHAPE and not self.compact:
                raise ValueError("128 image_shape requires compact proprio fusion")
            if self.compact:
                if pooling != "flatten" or (self.image_shape == IMAGE_SHAPE and cnn_output_size != 14400):
                    raise ValueError("Compact proprio requires the 64-channel visual trunk")
                if self.image_shape != IMAGE_SHAPE and conv_specs != (
                    (16, 6, 2, 0), (32, 4, 2, 0), (64, 3, 1, 0), (64, 3, 1, 0)
                ):
                    raise ValueError("128 compact proprio requires unchanged compact conv specs")
                height, width = self.image_shape[:2]
                for _, kernel, stride, padding in conv_specs:
                    height = _conv_output_size(height, kernel, stride, padding)
                    width = _conv_output_size(width, kernel, stride, padding)
                self._compact_y_bounds = tuple((i * height // 4, ((i + 1) * height + 3) // 4) for i in range(4))
                self._compact_x_bounds = tuple((i * width // 4, ((i + 1) * width + 3) // 4) for i in range(4))
                self.visual_projection = self._build_sequential_mlp(1024, (256,), "relu")
                self.proprio_mlp = self._build_sequential_mlp(self.proprio_size, (64, 128), "relu")
                if self.separate:
                    self.critic_visual_projection = self._build_sequential_mlp(1024, (256,), "relu")
                    self.critic_proprio_mlp = self._build_sequential_mlp(self.proprio_size, (64, 128), "relu")

            mlp_cfg = params["mlp"]
            units = tuple(int(unit) for unit in mlp_cfg["units"])
            if not units:
                raise ValueError("Visual proprio MLP must have at least one hidden layer")
            fused_size = cnn_output_size + self.proprio_size
            if self.compact:
                fused_size = 384
            self.actor_mlp = self._build_sequential_mlp(
                fused_size, units, mlp_cfg.get("activation", "relu")
            )
            if self.separate:
                self.critic_mlp = self._build_sequential_mlp(
                    fused_size, units, mlp_cfg.get("activation", "relu")
                )
            else:
                self.critic_mlp = None

            continuous = params["space"]["continuous"]
            self.mu = nn.Linear(units[-1], actions_num)
            self.value = nn.Linear(units[-1], self.value_size)
            self.mu_act = self.activations_factory.create(
                continuous.get("mu_activation", "None")
            )
            self.value_act = self.activations_factory.create(
                params.get("value_activation", "None")
            )
            self.sigma_act = self.activations_factory.create(
                continuous.get("sigma_activation", "None")
            )
            self.fixed_sigma = bool(continuous.get("fixed_sigma", True))
            if self.fixed_sigma:
                self.sigma = nn.Parameter(torch.zeros(actions_num, dtype=torch.float32))
            else:
                self.sigma = nn.Linear(units[-1], actions_num)

            self._initialize_weights(
                cnn_cfg.get("initializer", {"name": "default"}),
                mlp_cfg.get("initializer", {"name": "default"}),
                continuous,
            )

        def _parse_conv_specs(self, cnn_cfg):
            if "convs" in cnn_cfg:
                return tuple(
                    (
                        int(spec["filters"]),
                        int(spec["kernel_size"]),
                        int(spec["strides"]),
                        int(spec.get("padding", 0)),
                    )
                    for spec in cnn_cfg["convs"]
                )

            filters = _as_tuple(cnn_cfg["filters"], name="filters", length=3)
            kernels = _as_tuple(cnn_cfg["kernel_sizes"], name="kernel_sizes", length=3)
            strides = _as_tuple(cnn_cfg["strides"], name="strides", length=3)
            return tuple(zip(filters, kernels, strides, (0, 0, 0), strict=True))

        def _cnn_output_size(self, conv_specs, pooling):
            if pooling == "adaptive_avg":
                return conv_specs[-1][0]
            height, width = self.image_shape[:2]
            for _, kernel, stride, padding in conv_specs:
                height = _conv_output_size(height, kernel, stride, padding)
                width = _conv_output_size(width, kernel, stride, padding)
            if height <= 0 or width <= 0:
                raise ValueError("CNN configuration collapses the image spatial dimensions")
            return conv_specs[-1][0] * height * width

        def _build_encoder(self, conv_specs, activation, pooling):
            layers: list[nn.Module] = []
            in_channels = self.image_shape[-1]
            for out_channels, kernel, stride, padding in conv_specs:
                layers.extend(
                    (
                        nn.Conv2d(
                            in_channels,
                            out_channels,
                            kernel_size=kernel,
                            stride=stride,
                            padding=padding,
                            bias=False,
                        ),
                        self.activations_factory.create(activation),
                    )
                )
                in_channels = out_channels
            if pooling == "adaptive_avg":
                layers.append(nn.AdaptiveAvgPool2d((1, 1)))
            return nn.Sequential(*layers)

        def _build_sequential_mlp(self, input_size, units, activation):
            layers: list[nn.Module] = []
            in_size = input_size
            for unit in units:
                layers.extend(
                    (nn.Linear(in_size, unit), self.activations_factory.create(activation))
                )
                in_size = unit
            return nn.Sequential(*layers)

        def _initialize_weights(self, cnn_initializer, mlp_initializer, continuous):
            cnn_init = self.init_factory.create(**cnn_initializer)
            mlp_init = self.init_factory.create(**mlp_initializer)
            for module in self.modules():
                if isinstance(module, nn.Conv2d):
                    cnn_init(module.weight)
                elif isinstance(module, nn.Linear):
                    mlp_init(module.weight)
                    if module.bias is not None:
                        nn.init.zeros_(module.bias)

            mu_init = self.init_factory.create(
                **continuous.get("mu_init", {"name": "default"})
            )
            sigma_init = self.init_factory.create(
                **continuous.get(
                    "sigma_init", {"name": "const_initializer", "val": 0}
                )
            )
            mu_init(self.mu.weight)
            if self.fixed_sigma:
                sigma_init(self.sigma)
            else:
                sigma_init(self.sigma.weight)

        def _split_and_preprocess(self, observation: torch.Tensor):
            if observation.ndim != 2 or observation.shape[-1] != self.observation_size:
                raise ValueError(
                    f"Expected observation shape (B, {self.observation_size}), "
                    f"got {tuple(observation.shape)}"
                )

            image = observation[:, : self.image_size].reshape(
                -1, *self.image_shape
            ).permute(0, 3, 1, 2)
            image = image.to(dtype=torch.float32) / 255.0
            if self.normalise_channels:
                image = F.layer_norm(
                    image,
                    normalized_shape=self.image_shape[:2],
                    weight=None,
                    bias=None,
                    eps=1.0e-6,
                )

            proprio = observation[:, self.image_size :].to(dtype=torch.float32)
            if self.goal_fixed_scale:
                proprio = proprio * self.proprio_scale
            return image, proprio

        def _compact_visual_features(self, visual, projection):
            # Exact adaptive bins, exported as static Slice/ReduceMean at either resolution.
            rows = [torch.cat([
                visual[:, :, y0:y1, x0:x1].mean((-2, -1), keepdim=True)
                for x0, x1 in self._compact_x_bounds
            ], dim=-1) for y0, y1 in self._compact_y_bounds]
            return projection(torch.cat(rows, dim=-2).flatten(1))

        def _actor_features(self, image, proprio):
            actor_visual = self.actor_cnn(image)
            actor_proprio = proprio
            if self.compact:
                actor_visual = self._compact_visual_features(actor_visual, self.visual_projection)
                actor_proprio = self.proprio_mlp(proprio)
            else:
                actor_visual = actor_visual.flatten(1)
            return self.actor_mlp(
                torch.cat((actor_visual, actor_proprio), dim=-1)
            )

        def forward_actor(self, obs_dict):
            image, proprio = self._split_and_preprocess(obs_dict["obs"])
            return self.mu_act(self.mu(self._actor_features(image, proprio)))

        def forward(self, obs_dict):
            image, proprio = self._split_and_preprocess(obs_dict["obs"])
            states = obs_dict.get("rnn_states")
            actor_features = self._actor_features(image, proprio)
            if self.separate:
                critic_visual = self.critic_cnn(image)
                critic_proprio = proprio
                if self.compact:
                    critic_visual = self._compact_visual_features(critic_visual, self.critic_visual_projection)
                    critic_proprio = self.critic_proprio_mlp(proprio)
                else:
                    critic_visual = critic_visual.flatten(1)
                critic_features = self.critic_mlp(
                    torch.cat((critic_visual, critic_proprio), dim=-1)
                )
            else:
                critic_features = actor_features

            mu = self.mu_act(self.mu(actor_features))
            value = self.value_act(self.value(critic_features))
            if self.fixed_sigma:
                sigma = mu * 0.0 + self.sigma_act(self.sigma)
            else:
                sigma = self.sigma_act(self.sigma(actor_features))
            return mu, sigma, value, states

        def is_separate_critic(self):
            return self.separate

        def is_rnn(self):
            return False

        def get_default_rnn_state(self):
            return None


def register_visual_proprio_networks() -> None:
    """Register both public names against the shared implementation."""
    model_builder.register_network(LIFT_NETWORK_NAME, VisualProprioBuilder)
    model_builder.register_network(REACH_NETWORK_NAME, VisualProprioBuilder)
