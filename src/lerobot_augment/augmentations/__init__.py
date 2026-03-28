"""Augmentation registry and chain builder."""

from __future__ import annotations

import argparse

from lerobot_augment.augmentations.visual import (
    ColorJitterAugmentation,
    GaussianBlurAugmentation,
    RandomErasingAugmentation,
)
from lerobot_augment.augmentations.action_noise import ActionNoiseAugmentation
from lerobot_augment.augmentations.temporal import TemporalSubsampleAugmentation
from lerobot_augment.augmentations.smoothing import TrajectorySmoothingAugmentation
from lerobot_augment.augmentations.base import Augmentation


def build_augmentation_chain(args: argparse.Namespace, image_keys: list[str]) -> list[Augmentation]:
    """Build a list of augmentations from CLI args."""
    chain: list[Augmentation] = []

    if args.color_jitter:
        chain.append(ColorJitterAugmentation(
            image_keys=image_keys,
            brightness=args.cj_brightness,
            contrast=args.cj_contrast,
            saturation=args.cj_saturation,
            hue=args.cj_hue,
        ))

    if args.gaussian_blur:
        chain.append(GaussianBlurAugmentation(
            image_keys=image_keys,
            kernel_size=args.gb_kernel_size,
            sigma_min=args.gb_sigma_min,
            sigma_max=args.gb_sigma_max,
        ))

    if args.random_erasing:
        chain.append(RandomErasingAugmentation(
            image_keys=image_keys,
            p=args.re_p,
            scale_min=args.re_scale_min,
            scale_max=args.re_scale_max,
        ))

    if args.action_noise:
        chain.append(ActionNoiseAugmentation(std=args.action_noise_std))

    if args.smooth_trajectory:
        chain.append(TrajectorySmoothingAugmentation(window_size=args.smooth_window))

    return chain


__all__ = [
    "build_augmentation_chain",
    "Augmentation",
    "ColorJitterAugmentation",
    "GaussianBlurAugmentation",
    "RandomErasingAugmentation",
    "ActionNoiseAugmentation",
    "TemporalSubsampleAugmentation",
    "TrajectorySmoothingAugmentation",
]
