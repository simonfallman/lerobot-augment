# lerobot-augment

A CLI tool for offline augmentation and filtering of [LeRobot v3](https://huggingface.co/docs/lerobot/en/lerobot-dataset-v3) datasets. It reads a source dataset from the Hugging Face Hub, applies configurable visual transforms, trajectory augmentations, and quality filters, writes a new dataset, and uploads it back to the Hub.

## What it does

```
Source dataset (HF Hub) → Filter → Trim → Augment → Multiply → New dataset (HF Hub)
```

**The pipeline:**
1. Downloads a LeRobot v3 dataset from HuggingFace Hub
2. **Filters** out bad episodes (low action variance = robot barely moved)
3. **Trims** idle frames from the start/end of each episode
4. **Augments** with visual transforms (color, blur, occlusion) and trajectory modifications (noise, smoothing)
5. **Multiplies** by creating N augmented copies per episode
6. Uploads the result and prints a visualizer link

## Why?

Recording robot demonstrations is expensive. If you have 50 recordings of "pick up the cup," augmentation turns them into 200+ varied episodes — with different lighting, slight trajectory variations, and cleaned-up data — making trained models more robust without recording a single new demo.

## Installation

```bash
# Requires Python >= 3.10
uv venv --python 3.12 .venv
source .venv/bin/activate
uv pip install -e .
```

Or just `pip install -e .` if you don't use [uv](https://docs.astral.sh/uv/).

## Quick start

```bash
# Log in to HuggingFace first
python3 -c "from huggingface_hub import login; login()"

# Augment with color jitter + action noise, include originals
lerobot-augment \
  --source-repo-id lerobot/aloha_static_cups_open \
  --output-repo-id your-username/aloha_cups_augmented \
  --color-jitter \
  --action-noise \
  --trim-idle \
  --include-originals \
  --push-to-hub
```

Output:
```
Dataset uploaded: https://huggingface.co/datasets/your-username/aloha_cups_augmented
Visualize: https://huggingface.co/spaces/lerobot/visualize_dataset?path=%2Fyour-username%2Faloha_cups_augmented%2Fepisode_0
```

## Live demo

Augmented dataset on HuggingFace Hub: [simonfallman/aloha_cups_augmented](https://huggingface.co/datasets/simonfallman/aloha_cups_augmented)

[View in visualizer](https://huggingface.co/spaces/lerobot/visualize_dataset?path=%2Fsimonfallman%2Faloha_cups_augmented%2Fepisode_0) — switch between episodes to compare original vs augmented.

## Features

### Augmentation (increase variation)

| Type | Flag | What it does |
|---|---|---|
| **Color Jitter** | `--color-jitter` | Adjusts brightness, contrast, saturation, hue (consistent per episode to avoid flickering) |
| **Gaussian Blur** | `--gaussian-blur` | Random blur (consistent per episode) |
| **Random Erasing** | `--random-erasing` | Cutout-style occlusion patches (independent per frame) |
| **Action Noise** | `--action-noise` | Gaussian noise on action tensors for trajectory robustness |
| **Trajectory Smoothing** | `--smooth-trajectory` | Moving average on actions to reduce sensor noise |
| **Temporal Subsample** | `--temporal-subsample` | Creates episodes at reduced frame rates |

### Filtering (improve quality)

| Type | Flag | What it does |
|---|---|---|
| **Action Variance Filter** | `--min-action-variance 0.001` | Removes episodes where the robot barely moved (stuck/failed demos) |
| **Idle Trimming** | `--trim-idle` | Removes dead frames from start/end of episodes where robot sits still |
| **Length Filter** | `--min-episode-length 50` | Removes episodes that are too short or too long |

### Multiplying

| Type | Flag | What it does |
|---|---|---|
| **Copies** | `--num-augmented-copies 3` | Creates N augmented variants per episode |
| **Include Originals** | `--include-originals` | Keeps original episodes alongside augmented ones (dataset mixing) |

## Examples

### Clean up a noisy dataset (filter only, no augmentation)

```bash
lerobot-augment \
  --source-repo-id lerobot/aloha_static_cups_open \
  --output-repo-id your-username/aloha_cleaned \
  --min-action-variance 0.0005 \
  --trim-idle \
  --min-episode-length 50 \
  --include-originals \
  --push-to-hub
```

### Heavy visual augmentation (3x dataset)

```bash
lerobot-augment \
  --source-repo-id lerobot/aloha_static_cups_open \
  --output-repo-id your-username/aloha_visual_aug \
  --color-jitter --cj-brightness 0.6 --cj-hue 0.1 \
  --gaussian-blur \
  --random-erasing --re-p 0.3 \
  --num-augmented-copies 3 \
  --push-to-hub
```

### Full pipeline (filter + trim + augment + multiply)

```bash
lerobot-augment \
  --source-repo-id lerobot/aloha_static_cups_open \
  --output-repo-id your-username/aloha_full \
  --min-action-variance 0.0005 \
  --trim-idle \
  --color-jitter --cj-brightness 0.5 --cj-hue 0.08 \
  --action-noise --action-noise-std 0.02 \
  --smooth-trajectory \
  --include-originals \
  --num-augmented-copies 2 \
  --push-to-hub
```

## CLI reference

Run `lerobot-augment --help` for full details. Key arguments:

| Argument | Default | Description |
|---|---|---|
| `--source-repo-id` | required | Source dataset on HF Hub |
| `--output-repo-id` | required | Output dataset repo ID |
| `--seed` | 42 | Random seed for reproducibility |
| `--episodes` | all | Only process specific episode indices |
| `--push-to-hub` | false | Upload result to HF Hub |
| `--private` | false | Make uploaded dataset private |

## Design decisions

- **Episode-level consistency**: Visual augmentations (color jitter, blur) sample parameters once per episode, preventing temporal flickering across frames
- **Deterministic seeding**: Each episode+copy gets a unique seed (`base_seed + ep_idx * 1000 + copy_idx`), ensuring reproducibility
- **Filter before augment**: Bad episodes are removed before augmentation, so you don't waste time augmenting junk data
- **Idle trimming preserves context**: Keeps one frame before motion starts and one after it ends, so the model still sees the transition

## How this was built (AI coding agents)

This tool was built in a single session using **Claude Code** (Anthropic's AI coding agent). I had no prior experience with LeRobot or robotics datasets — I described the challenge requirements and worked with Claude Code iteratively to go from zero to a working, deployed tool.

### How Claude Code was used at each stage

**1. Research (I didn't know the domain)**

I gave Claude Code the challenge description. It launched parallel exploration agents — one to research the LeRobot v3 dataset format (parquet files, video structure, metadata), another to research which augmentation techniques are most valuable for robotics (color jitter, action noise, trajectory smoothing, etc.). This gave me a crash course in a domain I'd never worked in.

**2. Architecture & planning**

Claude Code entered a planning mode where it designed the full architecture before writing any code — the augmentation base class, episode-level processing pattern, frame format conversion pipeline, and CLI interface. I reviewed and approved the plan before implementation started.

**3. Implementation**

Claude Code wrote all source files. Rather than trusting documentation alone, it used `inspect.signature` and `inspect.getsource` on the installed LeRobot package to match the actual v3 API. It set up the Python environment from scratch (installing `uv`, creating a venv, installing dependencies).

**4. Debugging (the interesting part)**

The first end-to-end test failed — three issues:
- Image tensors were CHW (channel-first) from `dataset[idx]` but `add_frame()` expected HWC (height-width-channel)
- `next.done` had shape `()` but the writer expected `(1,)`
- Python's multiprocessing crashed when running from stdin

Claude Code diagnosed each by reading the LeRobot source code, fixed the frame conversion pipeline, and got a successful run.

**5. Iterative feature development**

After the core pipeline worked, I asked Claude Code to add filtering and trimming features. It implemented action variance filtering (skip episodes where the robot didn't move), idle frame trimming (cut dead time), and trajectory smoothing — all wired into the CLI with new flags.

**6. Testing across datasets**

We tested on two completely different robot setups:
- ALOHA robot (4 cameras, 14-DOF) — `lerobot/aloha_static_cups_open`
- xArm robot (1 camera, different action space) — `lerobot/xarm_lift_medium`

Both worked without any code changes, proving the tool is general-purpose.

### What I did vs what Claude Code did

- **Me**: Described the challenge, made decisions (CLI vs notebook, which features to prioritize, when to ship), reviewed output in the visualizer, noticed issues (augmentations not visible enough, missing random erasing), asked for explanations when I didn't understand something
- **Claude Code**: All research, architecture design, code writing, environment setup, debugging, testing, git operations, and HuggingFace uploads

## License

Apache 2.0
