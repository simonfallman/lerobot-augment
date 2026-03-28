# lerobot-augment

A CLI tool for offline augmentation of [LeRobot v3](https://huggingface.co/docs/lerobot/en/lerobot-dataset-v3) datasets. It reads a source dataset from the Hugging Face Hub, applies configurable visual and trajectory augmentations, writes a new dataset, and uploads it back to the Hub.

## Why offline augmentation?

LeRobot already supports on-the-fly image transforms during training. This tool fills a different need:

- **Shareable**: Augmented datasets are standalone HF datasets anyone can load
- **Reproducible**: Deterministic seeding means identical outputs given the same seed
- **Framework-agnostic**: Consumers don't need LeRobot's transform pipeline
- **Composable**: Mix originals + augmented data in a single dataset

## Installation

```bash
# Requires Python >= 3.10
pip install -e .
```

Or with [uv](https://docs.astral.sh/uv/) (recommended):

```bash
uv venv --python 3.12 .venv
source .venv/bin/activate
uv pip install -e .
```

## Quick start

```bash
# Augment with color jitter + action noise, include originals for dataset mixing
lerobot-augment \
  --source-repo-id lerobot/aloha_static_cups_open \
  --output-repo-id your-username/aloha_cups_augmented \
  --color-jitter \
  --action-noise \
  --include-originals \
  --push-to-hub
```

This will:
1. Download the source dataset
2. For each episode: copy the original + create an augmented version
3. Upload the result to Hugging Face Hub
4. Print a visualizer link

## Augmentation types

| Augmentation | Flag | What it does | Key parameters |
|---|---|---|---|
| **Color Jitter** | `--color-jitter` | Adjusts brightness, contrast, saturation, hue (consistent across frames in an episode) | `--cj-brightness 0.3` `--cj-contrast 0.3` `--cj-saturation 0.3` `--cj-hue 0.05` |
| **Gaussian Blur** | `--gaussian-blur` | Applies random Gaussian blur (consistent per episode) | `--gb-kernel-size 5` `--gb-sigma-min 0.1` `--gb-sigma-max 2.0` |
| **Random Erasing** | `--random-erasing` | Cutout-style occlusion patches (independent per frame) | `--re-p 0.3` `--re-scale-min 0.02` `--re-scale-max 0.15` |
| **Action Noise** | `--action-noise` | Gaussian noise on action tensors for trajectory robustness | `--action-noise-std 0.01` |
| **Temporal Subsample** | `--temporal-subsample` | Creates episodes at reduced frame rates (every Nth frame) | `--temporal-subsample-factors 2 3` |

## Examples

### Visual-only augmentation (3 copies per episode)

```bash
lerobot-augment \
  --source-repo-id lerobot/aloha_static_cups_open \
  --output-repo-id your-username/aloha_visual_aug \
  --color-jitter --gaussian-blur \
  --num-augmented-copies 3 \
  --push-to-hub
```

### Action noise + originals for mixed training

```bash
lerobot-augment \
  --source-repo-id lerobot/aloha_static_cups_open \
  --output-repo-id your-username/aloha_mixed \
  --action-noise --action-noise-std 0.02 \
  --include-originals \
  --push-to-hub
```

### Process only specific episodes

```bash
lerobot-augment \
  --source-repo-id lerobot/aloha_static_cups_open \
  --output-repo-id your-username/aloha_subset \
  --episodes 0 1 2 \
  --color-jitter \
  --push-to-hub
```

### Full kitchen sink

```bash
lerobot-augment \
  --source-repo-id lerobot/aloha_static_cups_open \
  --output-repo-id your-username/aloha_full_aug \
  --color-jitter --cj-brightness 0.4 --cj-hue 0.1 \
  --gaussian-blur \
  --random-erasing --re-p 0.2 \
  --action-noise --action-noise-std 0.015 \
  --temporal-subsample --temporal-subsample-factors 2 \
  --include-originals \
  --num-augmented-copies 2 \
  --min-episode-length 50 \
  --seed 123 \
  --push-to-hub
```

## CLI reference

```
lerobot-augment --help
```

| Argument | Default | Description |
|---|---|---|
| `--source-repo-id` | required | Source dataset on HF Hub |
| `--output-repo-id` | required | Output dataset repo ID |
| `--seed` | 42 | Random seed for reproducibility |
| `--num-augmented-copies` | 1 | Augmented copies per episode |
| `--include-originals` | false | Also include unaugmented episodes |
| `--episodes` | all | Process only these episode indices |
| `--min-episode-length` | 0 | Skip shorter episodes |
| `--max-episode-length` | 0 | Skip longer episodes |
| `--push-to-hub` | false | Upload result to HF Hub |
| `--private` | false | Make uploaded dataset private |

## Design decisions

- **Episode-level consistency**: Visual augmentations (color jitter, blur) sample parameters once per episode and apply them uniformly across all frames, preventing temporal flickering
- **Independent erasing**: Random erasing is applied independently per frame because occlusions naturally vary frame-to-frame
- **Deterministic seeding**: Each episode+copy combination gets a unique seed (`base_seed + ep_idx * 1000 + copy_idx`), ensuring reproducibility
- **Offline materialization**: Writes a complete, standalone LeRobot v3 dataset rather than wrapping the original

## How this was built (AI coding agents)

This tool was built using **Claude Code** (Anthropic's AI coding agent). Here's how AI agents were utilized throughout development:

1. **Research phase**: Used Claude Code's web search and exploration agents to research the LeRobot v3 dataset format, the `LeRobotDataset` Python API (`create()`, `add_frame()`, `save_episode()`, `finalize()`, `push_to_hub()`), and robotics dataset augmentation best practices
2. **Architecture design**: Claude Code planned the modular architecture (augmentation base class, episode-level processing, frame conversion pipeline) and identified key technical challenges (CHW→HWC image format conversion, auto-managed field exclusion, task metadata resolution)
3. **Implementation**: Claude Code wrote all source files, iterating through API inspection (`inspect.signature`, `inspect.getsource`) to match the actual LeRobot v3 API exactly
4. **Debugging**: When the initial implementation failed validation (wrong image tensor format, `next.done` shape mismatch), Claude Code diagnosed the root causes by inspecting the LeRobot source and fixed the frame conversion pipeline
5. **End-to-end testing**: Claude Code ran the tool against `lerobot/aloha_static_cups_open`, verified correct video encoding, and validated the output dataset structure

The entire tool — from research to working CLI — was built in a single Claude Code session.

## License

Apache 2.0
