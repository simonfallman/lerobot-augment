"""Smart episode filtering: action variance, idle trimming."""

from __future__ import annotations

import torch
import numpy as np


def compute_action_variance(frames: list[dict]) -> float:
    """Compute mean variance across action dimensions for an episode.

    Low variance = robot barely moved = likely a failed/stuck demo.
    """
    actions = []
    for frame in frames:
        if "action" in frame:
            a = frame["action"]
            if isinstance(a, torch.Tensor):
                a = a.detach().cpu().numpy()
            actions.append(a)

    if len(actions) < 2:
        return 0.0

    actions = np.stack(actions)
    return float(np.mean(np.var(actions, axis=0)))


def filter_by_action_variance(frames: list[dict], min_variance: float) -> bool:
    """Return True if episode has enough action variance to keep."""
    return compute_action_variance(frames) >= min_variance


def trim_idle_frames(frames: list[dict], threshold: float = 0.001) -> list[dict]:
    """Remove idle frames from the start and end of an episode.

    A frame is "idle" if the action magnitude (L2 norm of difference from
    previous frame) is below the threshold. This trims dead time where
    the robot sits still before/after the actual task.
    """
    if len(frames) < 3:
        return frames

    # Compute frame-to-frame action deltas
    deltas = []
    for i in range(len(frames)):
        if i == 0:
            deltas.append(0.0)
            continue
        curr = frames[i].get("action")
        prev = frames[i - 1].get("action")
        if curr is None or prev is None:
            deltas.append(0.0)
            continue
        if isinstance(curr, torch.Tensor):
            curr = curr.detach().cpu().numpy()
        if isinstance(prev, torch.Tensor):
            prev = prev.detach().cpu().numpy()
        deltas.append(float(np.linalg.norm(curr - prev)))

    # Find first and last non-idle frame
    start = 0
    for i, d in enumerate(deltas):
        if d > threshold:
            start = max(0, i - 1)  # Keep one frame before motion starts
            break

    end = len(frames)
    for i in range(len(deltas) - 1, -1, -1):
        if deltas[i] > threshold:
            end = min(len(frames), i + 2)  # Keep one frame after motion ends
            break

    trimmed = frames[start:end]

    # Never return fewer than 2 frames
    if len(trimmed) < 2:
        return frames

    return trimmed
