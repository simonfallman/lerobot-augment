"""Utility functions for dataset augmentation."""

from __future__ import annotations

from urllib.parse import quote

import torch
import numpy as np


# Keys that LeRobotDataset manages automatically via DEFAULT_FEATURES.
# These must not be included in the output features dict or frame dicts.
DEFAULT_FEATURE_KEYS = {
    "episode_index",
    "frame_index",
    "timestamp",
    "index",
    "task_index",
}

# Additional keys that need special handling
SPECIAL_KEYS = {
    "next.done",  # Shape mismatch: dataset[idx] returns () but add_frame expects (1,)
    "next.reward",
}

AUTO_MANAGED_KEYS = DEFAULT_FEATURE_KEYS | SPECIAL_KEYS


def get_image_keys(features: dict) -> list[str]:
    """Extract image/video feature keys from the dataset features dict."""
    image_keys = []
    for key, info in features.items():
        dtype = info.get("dtype", "")
        if dtype in ("image", "video"):
            image_keys.append(key)
    return image_keys


def prepare_frame_for_writer(frame: dict, image_keys: list[str]) -> dict:
    """Convert a frame dict from dataset[idx] format to add_frame() format.

    dataset[idx] returns CHW float tensors for images.
    add_frame() expects HWC numpy/PIL for images.
    Skips auto-managed keys.
    """
    out = {}
    for key, value in frame.items():
        if key in AUTO_MANAGED_KEYS:
            continue
        if key == "task":
            continue
        if key in image_keys:
            if isinstance(value, torch.Tensor):
                if value.dtype in (torch.float32, torch.float64):
                    arr = torch.round(value.clamp(0, 1) * 255).to(torch.uint8)
                else:
                    arr = value
                # CHW -> HWC
                arr = arr.permute(1, 2, 0).contiguous().cpu().numpy()
                out[key] = arr
            else:
                out[key] = value
        elif isinstance(value, torch.Tensor):
            out[key] = value.detach().cpu().numpy()
        elif isinstance(value, (int, float, str, bool, np.ndarray)):
            out[key] = value
    return out


def visualizer_url(repo_id: str, episode: int = 0) -> str:
    """Build the LeRobot visualizer URL for a dataset."""
    encoded = quote(f"/{repo_id}/episode_{episode}", safe="")
    return f"https://huggingface.co/spaces/lerobot/visualize_dataset?path={encoded}"


def dataset_url(repo_id: str) -> str:
    """Build the HF dataset URL."""
    return f"https://huggingface.co/datasets/{repo_id}"
