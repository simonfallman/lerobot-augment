"""Temporal subsampling augmentation."""

from __future__ import annotations

from lerobot_augment.augmentations.base import Augmentation


class TemporalSubsampleAugmentation(Augmentation):
    """Create a subsampled episode by keeping every Nth frame.

    This effectively creates episodes at reduced temporal resolution,
    helping policies learn to be robust to timing variations.
    """

    def __init__(self, factor: int = 2):
        if factor < 2:
            raise ValueError(f"Subsample factor must be >= 2, got {factor}")
        self.factor = factor

    @property
    def name(self) -> str:
        return f"TemporalSubsample(factor={self.factor})"

    def __call__(self, frames: list[dict]) -> list[dict]:
        if not frames:
            return frames
        return frames[:: self.factor]
