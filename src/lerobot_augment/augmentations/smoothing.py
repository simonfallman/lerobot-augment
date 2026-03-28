"""Trajectory smoothing augmentation."""

from __future__ import annotations

import torch
import numpy as np

from lerobot_augment.augmentations.base import Augmentation


class TrajectorySmoothingAugmentation(Augmentation):
    """Smooth noisy action trajectories with a moving average.

    Applies a simple moving average to action values across frames,
    reducing high-frequency noise while preserving the overall trajectory shape.
    """

    def __init__(self, window_size: int = 5):
        if window_size < 3:
            raise ValueError(f"Window size must be >= 3, got {window_size}")
        if window_size % 2 == 0:
            window_size += 1  # Ensure odd for symmetric window
        self.window_size = window_size

    @property
    def name(self) -> str:
        return f"TrajectorySmoothing(window={self.window_size})"

    def __call__(self, frames: list[dict]) -> list[dict]:
        if len(frames) < self.window_size:
            return frames

        # Extract actions
        actions = []
        for frame in frames:
            a = frame.get("action")
            if a is None:
                return frames  # No actions to smooth
            if isinstance(a, torch.Tensor):
                actions.append(a.detach().cpu().numpy())
            else:
                actions.append(np.asarray(a))

        actions = np.stack(actions)  # [T, D]
        smoothed = np.copy(actions)

        # Moving average with edge padding
        half = self.window_size // 2
        for i in range(len(actions)):
            start = max(0, i - half)
            end = min(len(actions), i + half + 1)
            smoothed[i] = np.mean(actions[start:end], axis=0)

        # Write smoothed actions back
        result = []
        for i, frame in enumerate(frames):
            new_frame = dict(frame)
            if isinstance(frame["action"], torch.Tensor):
                new_frame["action"] = torch.from_numpy(smoothed[i]).float()
            else:
                new_frame["action"] = smoothed[i]
            result.append(new_frame)

        return result
