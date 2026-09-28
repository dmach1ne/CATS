from __future__ import annotations

import torch


class FrameStackBuffer:
    """Maintain per-environment image history in oldest-to-newest order."""

    def __init__(self, stack_num: int):
        if stack_num < 1:
            raise ValueError("stack_num must be at least one")
        self.stack_num = stack_num
        self._frames: list[torch.Tensor] = []
        self._frame_shape: tuple[int, ...] | None = None
        self._channel_dim: int | None = None

    @staticmethod
    def _infer_channel_dim(frame: torch.Tensor) -> int:
        if frame.ndim != 4:
            raise ValueError(f"Expected a batched 4-D image tensor, got shape {tuple(frame.shape)}")
        if frame.shape[-1] in (1, 3, 4):
            return -1
        if frame.shape[1] in (1, 3, 4):
            return 1
        raise ValueError(f"Cannot infer the channel dimension from shape {tuple(frame.shape)}")

    def clear(self) -> None:
        self._frames.clear()
        self._frame_shape = None
        self._channel_dim = None

    def update(self, frame: torch.Tensor, reset_mask: torch.Tensor | None = None) -> torch.Tensor:
        channel_dim = self._infer_channel_dim(frame)
        frame_shape = tuple(frame.shape)
        current = frame.detach().clone()

        if not self._frames:
            self._frames = [current.clone() for _ in range(self.stack_num)]
            self._frame_shape = frame_shape
            self._channel_dim = channel_dim
        else:
            if frame_shape != self._frame_shape or channel_dim != self._channel_dim:
                raise ValueError(
                    f"Frame shape changed from {self._frame_shape} to {frame_shape}"
                )
            self._frames = self._frames[1:] + [current]

        if reset_mask is not None:
            reset_mask = reset_mask.to(device=current.device, dtype=torch.bool)
            if reset_mask.ndim != 1 or reset_mask.shape[0] != current.shape[0]:
                raise ValueError(
                    f"reset_mask must have shape ({current.shape[0]},), got {tuple(reset_mask.shape)}"
                )
            for history_frame in self._frames:
                history_frame[reset_mask] = current[reset_mask]

        return torch.cat(self._frames, dim=self._channel_dim).detach()
