"""Abstract base class for augmentations."""

from __future__ import annotations

from abc import ABC, abstractmethod


class Augmentation(ABC):
    """Base class for all augmentations.

    Augmentations operate on entire episodes (list of frame dicts) to support
    temporal consistency — e.g., applying the same color jitter across all
    frames of an episode.
    """

    @abstractmethod
    def __call__(self, frames: list[dict]) -> list[dict]:
        """Apply augmentation to an episode's frames.

        Args:
            frames: List of frame dicts. Each dict has keys like
                'observation.images.top', 'observation.state', 'action', etc.
                Image values are [C, H, W] float tensors in [0, 1].

        Returns:
            Modified list of frame dicts (same length for per-frame transforms,
            possibly shorter for subsampling).
        """
        ...

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable name for logging."""
        ...
