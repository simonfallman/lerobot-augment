"""CLI entry point for lerobot-augment."""

from __future__ import annotations

import argparse
import sys


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="lerobot-augment",
        description="Augment LeRobot v3 datasets with visual and trajectory transformations.",
    )

    # Required
    parser.add_argument(
        "--source-repo-id",
        required=True,
        help="HF Hub repo ID of the source dataset (e.g., lerobot/aloha_static_cups_open)",
    )
    parser.add_argument(
        "--output-repo-id",
        required=True,
        help="HF Hub repo ID for the augmented output dataset",
    )

    # General
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    parser.add_argument(
        "--num-augmented-copies",
        type=int,
        default=1,
        help="Number of augmented copies per episode (default: 1)",
    )
    parser.add_argument(
        "--include-originals",
        action="store_true",
        help="Also include original (unaugmented) episodes in the output",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        nargs="+",
        default=None,
        help="Only process specific episode indices (default: all)",
    )

    # Color jitter
    parser.add_argument("--color-jitter", action="store_true", help="Enable color jitter augmentation")
    parser.add_argument("--cj-brightness", type=float, default=0.3, help="Color jitter brightness (default: 0.3)")
    parser.add_argument("--cj-contrast", type=float, default=0.3, help="Color jitter contrast (default: 0.3)")
    parser.add_argument("--cj-saturation", type=float, default=0.3, help="Color jitter saturation (default: 0.3)")
    parser.add_argument("--cj-hue", type=float, default=0.05, help="Color jitter hue (default: 0.05)")

    # Gaussian blur
    parser.add_argument("--gaussian-blur", action="store_true", help="Enable Gaussian blur augmentation")
    parser.add_argument("--gb-kernel-size", type=int, default=5, help="Gaussian blur kernel size (default: 5)")
    parser.add_argument("--gb-sigma-min", type=float, default=0.1, help="Gaussian blur sigma min (default: 0.1)")
    parser.add_argument("--gb-sigma-max", type=float, default=2.0, help="Gaussian blur sigma max (default: 2.0)")

    # Random erasing
    parser.add_argument("--random-erasing", action="store_true", help="Enable random erasing augmentation")
    parser.add_argument("--re-p", type=float, default=0.3, help="Random erasing probability (default: 0.3)")
    parser.add_argument("--re-scale-min", type=float, default=0.02, help="Random erasing min scale (default: 0.02)")
    parser.add_argument("--re-scale-max", type=float, default=0.15, help="Random erasing max scale (default: 0.15)")

    # Action noise
    parser.add_argument("--action-noise", action="store_true", help="Enable action noise augmentation")
    parser.add_argument("--action-noise-std", type=float, default=0.01, help="Action noise std dev (default: 0.01)")

    # Temporal subsampling
    parser.add_argument("--temporal-subsample", action="store_true", help="Enable temporal subsampling")
    parser.add_argument(
        "--temporal-subsample-factors",
        type=int,
        nargs="+",
        default=[2],
        help="Subsample factors (default: [2])",
    )

    # Episode filtering
    parser.add_argument("--min-episode-length", type=int, default=0, help="Skip episodes shorter than N frames")
    parser.add_argument("--max-episode-length", type=int, default=0, help="Skip episodes longer than N frames")

    # Hub upload
    parser.add_argument("--push-to-hub", action="store_true", help="Upload the augmented dataset to HF Hub")
    parser.add_argument("--private", action="store_true", help="Make the uploaded dataset private")

    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)

    # Import here to avoid slow import on --help
    from lerobot_augment.pipeline import run_pipeline

    run_pipeline(args)


if __name__ == "__main__":
    main()
