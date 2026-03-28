"""Utility functions for dataset augmentation."""

from __future__ import annotations

from urllib.parse import quote

import torch
import numpy as np
from PIL import Image


# Keys that LeRobotDataset manages automatically — skip when building add_frame() dicts.
# DEFAULT_FEATURES: timestamp, frame_index, episode_index, index, task_index
# Also skip next.done/next.reward which have shape mismatches between __getitem__ and add_frame.
AUTO_MANAGED_KEYS = {
    "episode_index",
    "frame_index",
    "timestamp",
    "index",
    "task_index",
    "next.done",
    "next.reward",
}


def get_image_keys(features: dict) -> list[str]:
    """Extract image/video feature keys from the dataset features dict."""
    image_keys = []
    for key, info in features.items():
        dtype = info.get("dtype", "")
        if dtype in ("image", "video"):
            image_keys.append(key)
    return image_keys


def chw_to_hwc(tensor: torch.Tensor) -> torch.Tensor:
    """Convert [C, H, W] tensor to [H, W, C]."""
    if tensor.ndim == 3 and tensor.shape[0] in (1, 3, 4):
        return tensor.permute(1, 2, 0)
    return tensor


def tensor_to_pil(tensor: torch.Tensor) -> Image.Image:
    """Convert a [C, H, W] float tensor in [0, 1] to a PIL Image."""
    if tensor.dtype in (torch.float32, torch.float64):
        arr = (tensor.clamp(0, 1) * 255).byte().permute(1, 2, 0).numpy()
    else:
        arr = tensor.permute(1, 2, 0).numpy()
    return Image.fromarray(arr)


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
            # task is handled separately in the pipeline
            continue
        if key in image_keys:
            # Convert CHW float -> HWC uint8 numpy for add_frame()
            if isinstance(value, torch.Tensor):
                if value.dtype in (torch.float32, torch.float64):
                    arr = (value.clamp(0, 1) * 255).byte()
                else:
                    arr = value
                # CHW -> HWC
                arr = arr.permute(1, 2, 0).contiguous().numpy()
                out[key] = arr
            else:
                out[key] = value
        elif isinstance(value, torch.Tensor):
            out[key] = value.numpy()
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
