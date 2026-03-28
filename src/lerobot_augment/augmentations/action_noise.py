"""Action noise augmentation: Gaussian noise on action tensors."""

from __future__ import annotations

import torch

from lerobot_augment.augmentations.base import Augmentation


class ActionNoiseAugmentation(Augmentation):
    """Add Gaussian noise to action tensors for trajectory robustness."""

    def __init__(self, std: float = 0.01):
        self.std = std

    @property
    def name(self) -> str:
        return f"ActionNoise(std={self.std})"

    def __call__(self, frames: list[dict]) -> list[dict]:
        if not frames:
            return frames

        result = []
        for frame in frames:
            new_frame = dict(frame)
            if "action" in new_frame and isinstance(new_frame["action"], torch.Tensor):
                action = new_frame["action"].clone().float()
                noise = torch.randn_like(action) * self.std
                new_frame["action"] = action + noise
            result.append(new_frame)
        return result
