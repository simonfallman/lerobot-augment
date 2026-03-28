"""Visual augmentations: color jitter, Gaussian blur, random erasing."""

from __future__ import annotations

import random

import torch
import torchvision.transforms.v2 as T
import torchvision.transforms.v2.functional as F

from lerobot_augment.augmentations.base import Augmentation


class ColorJitterAugmentation(Augmentation):
    """Apply consistent color jitter across all frames of an episode.

    Samples jitter parameters once per episode for temporal consistency.
    """

    def __init__(
        self,
        image_keys: list[str],
        brightness: float = 0.3,
        contrast: float = 0.3,
        saturation: float = 0.3,
        hue: float = 0.05,
    ):
        self.image_keys = image_keys
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.hue = hue

    @property
    def name(self) -> str:
        return "ColorJitter"

    def __call__(self, frames: list[dict]) -> list[dict]:
        if not frames or not self.image_keys:
            return frames

        # Sample parameters once for the whole episode
        bf = random.uniform(max(0, 1 - self.brightness), 1 + self.brightness)
        cf = random.uniform(max(0, 1 - self.contrast), 1 + self.contrast)
        sf = random.uniform(max(0, 1 - self.saturation), 1 + self.saturation)
        hf = random.uniform(-self.hue, self.hue)

        for frame in frames:
            for key in self.image_keys:
                if key not in frame:
                    continue
                img = frame[key]
                img = F.adjust_brightness(img, bf)
                img = F.adjust_contrast(img, cf)
                img = F.adjust_saturation(img, sf)
                img = F.adjust_hue(img, hf)
                frame[key] = torch.clamp(img, 0.0, 1.0)
        return frames


class GaussianBlurAugmentation(Augmentation):
    """Apply consistent Gaussian blur across all frames of an episode."""

    def __init__(
        self,
        image_keys: list[str],
        kernel_size: int = 5,
        sigma_min: float = 0.1,
        sigma_max: float = 2.0,
    ):
        self.image_keys = image_keys
        self.kernel_size = kernel_size if kernel_size % 2 == 1 else kernel_size + 1
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max

    @property
    def name(self) -> str:
        return "GaussianBlur"

    def __call__(self, frames: list[dict]) -> list[dict]:
        if not frames or not self.image_keys:
            return frames

        sigma = random.uniform(self.sigma_min, self.sigma_max)
        ks = [self.kernel_size, self.kernel_size]
        ss = [sigma, sigma]

        for frame in frames:
            for key in self.image_keys:
                if key not in frame:
                    continue
                frame[key] = F.gaussian_blur(frame[key], ks, ss)
        return frames


class RandomErasingAugmentation(Augmentation):
    """Apply random erasing (cutout) independently per frame.

    Unlike color jitter and blur, erasing is applied independently per frame
    because occlusions naturally vary frame-to-frame.
    """

    def __init__(
        self,
        image_keys: list[str],
        p: float = 0.3,
        scale_min: float = 0.02,
        scale_max: float = 0.15,
    ):
        self.image_keys = image_keys
        self.p = p
        self.scale_min = scale_min
        self.scale_max = scale_max
        self._transform = T.RandomErasing(
            p=self.p,
            scale=(self.scale_min, self.scale_max),
            ratio=(0.3, 3.3),
        )

    @property
    def name(self) -> str:
        return "RandomErasing"

    def __call__(self, frames: list[dict]) -> list[dict]:
        if not frames or not self.image_keys:
            return frames

        result = []
        for frame in frames:
            new_frame = dict(frame)
            for key in self.image_keys:
                if key not in new_frame:
                    continue
                img = new_frame[key].clone()
                img = self._transform(img)
                new_frame[key] = img
            result.append(new_frame)
        return result
