"""Main augmentation pipeline: load -> augment -> write -> push."""

from __future__ import annotations

import argparse
import random
import shutil
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from lerobot.datasets.lerobot_dataset import LeRobotDataset

from lerobot_augment.augmentations import build_augmentation_chain
from lerobot_augment.augmentations.temporal import TemporalSubsampleAugmentation
from lerobot_augment.augmentations.filtering import (
    filter_by_action_variance,
    trim_idle_frames,
)
from lerobot_augment.utils import (
    AUTO_MANAGED_KEYS,
    get_image_keys,
    prepare_frame_for_writer,
    visualizer_url,
    dataset_url,
)


def read_episode_frames(dataset: LeRobotDataset, episode_local_idx: int) -> list[dict]:
    """Read all frames for a given episode.

    episode_local_idx is the index into meta.episodes (0-based relative to
    the loaded subset). We use dataset_from_index / dataset_to_index from
    the episodes metadata to find frame boundaries.
    """
    ep_info = dataset.meta.episodes[episode_local_idx]
    ep_start = ep_info["dataset_from_index"]
    ep_end = ep_info["dataset_to_index"]

    frames = []
    for idx in range(ep_start, ep_end):
        frame = dataset[idx]
        frames.append(frame)
    return frames


def get_task_for_episode(dataset: LeRobotDataset, episode_local_idx: int) -> str | None:
    """Resolve the task string for an episode."""
    ep_info = dataset.meta.episodes[episode_local_idx]

    # v3 stores tasks as a list of strings per episode
    tasks = ep_info.get("tasks")
    if tasks and isinstance(tasks, list) and len(tasks) > 0:
        return tasks[0]

    # Fallback: read from first frame
    try:
        frame = dataset[ep_info["dataset_from_index"]]
        if "task" in frame and isinstance(frame["task"], str):
            return frame["task"]
    except Exception:
        pass

    return None


def build_output_features(source_features: dict) -> dict:
    """Clone source features, excluding auto-managed keys."""
    return {k: v for k, v in source_features.items() if k not in AUTO_MANAGED_KEYS}


def write_episode(
    dst: LeRobotDataset,
    frames: list[dict],
    image_keys: list[str],
    task: str | None,
) -> None:
    """Write an episode's frames to the destination dataset."""
    if not frames:
        return
    # Pre-create image directories — LeRobot only creates them on frame_index==0,
    # but parallel video encoding can delete sibling dirs mid-write.
    ep_idx = dst.episode_buffer["episode_index"] if dst.episode_buffer else dst.meta.total_episodes
    for key in image_keys:
        img_dir = dst.root / "images" / key / f"episode-{ep_idx:06d}"
        img_dir.mkdir(parents=True, exist_ok=True)
    for frame in frames:
        frame_dict = prepare_frame_for_writer(frame, image_keys)
        frame_dict["task"] = task if task is not None else "unknown"
        dst.add_frame(frame_dict)
    dst.save_episode()


def run_pipeline(args: argparse.Namespace) -> None:
    """Execute the full augmentation pipeline."""
    # Validate HF auth early if pushing
    if args.push_to_hub:
        try:
            from huggingface_hub import HfApi
            HfApi().whoami()
        except Exception:
            print("Error: --push-to-hub requires HuggingFace authentication.")
            print("Run: python3 -c \"from huggingface_hub import login; login()\"")
            return

    # Load source — optionally only specific episodes
    if args.episodes is not None:
        print(f"Loading source dataset: {args.source_repo_id} (episodes {args.episodes})")
        src = LeRobotDataset(args.source_repo_id, episodes=args.episodes)
    else:
        print(f"Loading source dataset: {args.source_repo_id}")
        src = LeRobotDataset(args.source_repo_id)

    features = src.meta.info["features"]
    image_keys = get_image_keys(features)
    output_features = build_output_features(features)

    print(f"  Episodes: {src.num_episodes}")
    print(f"  Frames: {src.num_frames}")
    print(f"  FPS: {src.fps}")
    print(f"  Image keys: {image_keys}")

    # Create output dataset
    print(f"\nCreating output dataset: {args.output_repo_id}")
    robot_type = src.meta.info.get("robot_type", "unknown")

    # Clear stale local cache so LeRobotDataset.create doesn't hit FileExistsError
    output_cache = Path.home() / ".cache" / "huggingface" / "lerobot" / args.output_repo_id
    if output_cache.exists():
        shutil.rmtree(output_cache)

    dst = LeRobotDataset.create(
        repo_id=args.output_repo_id,
        fps=src.fps,
        features=output_features,
        robot_type=robot_type,
        vcodec=args.vcodec,
        streaming_encoding=args.streaming_encoding,
    )

    # Build augmentation chain
    augmentations = build_augmentation_chain(args, image_keys)
    if augmentations:
        print(f"\nAugmentations: {', '.join(a.name for a in augmentations)}")
    else:
        print("\nNo augmentations enabled — copying dataset as-is")

    # Temporal subsampling (handled separately since it generates extra episodes)
    temporal_augs = []
    if args.temporal_subsample:
        for factor in args.temporal_subsample_factors:
            temporal_augs.append(TemporalSubsampleAugmentation(factor=factor))
        print(f"Temporal subsampling factors: {args.temporal_subsample_factors}")

    # Episode length filter
    min_len = args.min_episode_length
    max_len = args.max_episode_length

    # Filtering summary
    if args.min_action_variance > 0:
        print(f"Action variance filter: min={args.min_action_variance}")
    if args.trim_idle:
        print(f"Idle trimming: threshold={args.trim_idle_threshold}")

    total_written = 0
    skipped_variance = 0
    skipped_length = 0
    trimmed_count = 0
    num_episodes = src.num_episodes
    print()

    for local_idx in tqdm(range(num_episodes), desc="Processing episodes"):
        ep_info = src.meta.episodes[local_idx]
        ep_length = ep_info["length"]

        # Filter by length
        if min_len > 0 and ep_length < min_len:
            skipped_length += 1
            continue
        if max_len > 0 and ep_length > max_len:
            skipped_length += 1
            continue

        frames = read_episode_frames(src, local_idx)
        task = get_task_for_episode(src, local_idx)

        # Filter by action variance
        if args.min_action_variance > 0:
            if not filter_by_action_variance(frames, args.min_action_variance):
                skipped_variance += 1
                continue

        # Trim idle frames
        if args.trim_idle:
            original_len = len(frames)
            frames = trim_idle_frames(frames, threshold=args.trim_idle_threshold)
            if len(frames) < original_len:
                trimmed_count += 1

        # Write originals if requested
        if args.include_originals:
            write_episode(dst, frames, image_keys, task)
            total_written += 1

        # Generate augmented copies
        for copy_idx in range(args.num_augmented_copies):
            # Deterministic seed per episode+copy
            seed = args.seed + local_idx * 1000 + copy_idx
            random.seed(seed)
            torch.manual_seed(seed)
            np.random.seed(seed % (2**32))

            # Apply augmentation chain
            augmented = frames
            for aug in augmentations:
                augmented = aug(augmented)

            # Write main augmented episode
            if not temporal_augs:
                write_episode(dst, augmented, image_keys, task)
                total_written += 1
            else:
                # Write full augmented version
                write_episode(dst, augmented, image_keys, task)
                total_written += 1
                # Write temporal subsampled variants
                for t_aug in temporal_augs:
                    subsampled = t_aug(augmented)
                    if len(subsampled) >= 2:
                        write_episode(dst, subsampled, image_keys, task)
                        total_written += 1

    # Print filtering summary
    if skipped_length > 0:
        print(f"\n  Skipped {skipped_length} episodes (length filter)")
    if skipped_variance > 0:
        print(f"  Skipped {skipped_variance} episodes (low action variance)")
    if trimmed_count > 0:
        print(f"  Trimmed idle frames from {trimmed_count} episodes")

    if total_written == 0:
        print("\nWarning: No episodes were written. All episodes may have been filtered out.")
        print("Check your --min-episode-length, --max-episode-length, and --min-action-variance settings.")
        return

    # Finalize
    print(f"\nFinalizing dataset ({total_written} episodes written)...")
    dst.finalize()

    # Push to Hub
    if args.push_to_hub:
        print("Pushing to Hugging Face Hub...")
        dst.push_to_hub(private=args.private)
        print(f"\nDataset uploaded: {dataset_url(args.output_repo_id)}")
        print(f"Visualize: {visualizer_url(args.output_repo_id)}")
    else:
        print(f"\nDataset saved locally. Use --push-to-hub to upload.")
        print(f"Visualizer URL (after push): {visualizer_url(args.output_repo_id)}")

    print(f"\nDone! {total_written} episodes in output dataset.")
