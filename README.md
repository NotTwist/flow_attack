# Flow Attack: Adversarial Patch Attacks on Optical Flow Models

This repository contains code for generating adversarial attacks on optical flow estimation models, with support for monocular depth estimation (MDE) and semantic segmentation (SS) models. The main focus is on **patch-based adversarial attacks** that can be applied in real-world scenarios.

## Table of Contents

- [Installation](#installation)
- [Configuration](#configuration)
- [Directory Structure](#directory-structure)
- [Running Patch Attacks](#running-patch-attacks)
- [Understanding Arguments](#understanding-arguments)
- [Examples](#examples)

---

## Installation

### Prerequisites

- Python 3.8+
- CUDA 12.0+ (recommended for GPU acceleration)
- pip or conda

### Step 1: Clone the Repository

```bash
git clone https://github.com/NotTwist/flow_attack.git
cd flow_attack
```

### Step 2: Install Dependencies

Install the main requirements:

```bash
conda env create -f environment.yml
```

The repository also includes submodule requirements for specific components:

```bash
pip install -r flow_library/requirements.txt
pip install -r attacks/adversarial_attacks_pytorch/requirements.txt
```

### Step 3: Prepare Datasets

Download and configure datasets as described in the [Configuration](#configuration) section.

---

## Configuration

### Dataset Configuration: `configs/datasets.yaml`

This file defines the paths and properties of datasets used for attacks. Example configuration:

```yaml
datasets:
  Sintel:
    root: "/path/to/sintel"
    split: "training"
    dstype: "clean"
    has_gt: true
    frames: 2

  Kitti15:
    root: "/path/to/kitti15"
    split: "training"
    has_gt: true
    frames: 2

  carla:
    root: "/path/to/carla_dataset"
    split: "eval"
    has_gt: false
    frames: 2
    camera_config: "camera_config.json"
    n_images: -1
```

**Key Configuration Fields:**
- `root`: Absolute path to dataset directory
- `split`: Dataset split ("training", "testing", "eval")
- `has_gt`: Whether ground truth flow is available
- `frames`: Number of frames per sample (typically 2 for optical flow)
- `n_images`: Number of images to use (-1 for all)

### Model Configuration: `configs/models.yaml`

This file specifies model architectures, weights paths, and hyperparameters.

**Optical Flow Models:** (not used, since we use models from ptlflow library)
```yaml
RAFT:
  config:
    epsilon: 1e-8
    small: false
    mixed_precision: false
    alternate_correlation: false
  weights_path: "models/_pretrained_weights/raft-sintel.pth"
  other_args: {}
```

**Segmentation Models:**
```yaml
pspnet_cityscapes:
  framework: mmseg
  type: pspnet
  config: "/path/to/PSPNet/pspnet_r50-d8_4xb2-40k_cityscapes-512x1024.py"
  weights_path: "/path/to/PSPNet/pspnet_r50-d8_512x1024_40k_cityscapes.pth"
```

---

## Directory Structure

### `attacks/`
Contains attack implementations:
- `patch_attack.py` - Patch-based adversarial attack training
- `patch_projection.py` - 3D patch projection onto scenes
- `get_attacks.py` - Attack method selection
- `DetectionDefenses/` - Defense mechanisms
- `attack_base.py` -     Abstract base class for **globally** attacking optical flow models.
### `models/` (not used, since we use ptlflow library for optical flow models)
Model definitions and pre-trained weights:
- `model_utils.py` - Model loading utilities
- `_pretrained_weights/` - Directory for storing pre-trained model weights
- `models/` - Model architecture implementations (FlowNet, RAFT, etc.)

### `metrics/`
Evaluation and metrics tracking:
- `attack_metrics.py` - Tracks and logs attack performance metrics using mlflow, saves artifacts
### `datasets_utils/`
Dataset handling and loading:
- `dataset_utils.py` - DataLoader creation for various datasets
- Supports: Sintel, KITTI-15, CARLA, custom datasets

### `utils/`
Utility functions:
- `args.py` - Command-line argument parser with all attack parameters
- `seed.py` - Random seed management for reproducibility
- `targets.py` - Target generation for attacks (zero flow, untargeted, scene-based, etc.)
- `process_images.py` - Image preprocessing and postprocessing

### `configs/`
Configuration files:
- `attacks.yaml` - Attack method configurations
- `datasets.yaml` - Dataset paths and properties
- `models.yaml` - Model specifications and weights paths

### `flow_library/`
Optical flow visualization and utility library

### `analysis/`
Analysis scripts and jupiter notebooks for post-attack evaluation

### `carla/`
CARLA simulator integration

---

## Running Patch Attacks

### Basic Usage

```bash
python run_patch_attack.py  --attack_mde  --attack_ss --patch_size 100  --n 2  --y_scale 3  --save_artifacts --patch_projection --small_run
```
This command starts patch attack on dataset Kitti15 (default) with patch size of 100 px and y_scale of 3 (creating oval patch form that works better with projection). Patch is projected onto the road using models' depth and segmentation predictions.

The attack is learned for 2 epochs (`--n 2`) and only first 32 images of the dataset are used (`--small_run`). `--save_artifacts` saves dataset images, along with model predictions with and without the patch. 

### Full Command Example with Multiple Targets

```bash
python run_patch_attack.py \
  --dataset Kitti15 \
  --model_name raft \
  --patch_size 100 \
  --n 50 \
  --attack_mde \
  --attack_ss \
  --optimizer adam \
  --lr 0.1 \
  --patch_projection \
  --loss_weights 1.0 0.1 1.0 \
  --save_artifacts \
  --output_dir experiment_results
```

---

## Understanding Arguments

### Core Optical Flow Attack Parameters

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--model_name` | str | `raft` | Optical flow model to attack. Choices: raft, gma, pwcnet, spynet, flownet2, meflow, memflow |
| `--dataset` | str | `Kitti15` | Dataset to use. Choices: `Kitti15`, `Sintel`, `carla` |
| `--target` | str | `zero` | Attack target for optical flow. Choices: `zero` (set flow to 0), `neg_flow` (negative flow), `untargeted` (random), `camera`, `scene`, `down` |

### Patch Configuration

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--patch_size` | int | `100` | Size of the adversarial patch in pixels |
| `--trained_patch` | str | `` (empty) | Path to a pre-trained patch. If empty, a new patch will be trained |
| `--change_of_variables` | flag | `False` | Use change-of-variable trick in patch optimization for unrestricted perturbations |
| `--random_loc` | flag | `False` | Randomize patch location during training (set `--random_loc` to enable; default is fixed location) |
| `--patch_projection` | flag | `False` | Enable 3D patch projection onto scene geometry using depth maps |
| `--y_scale` | float | `1.0` | Patch scaling factor on Y-axis (useful for elliptical patches) |

### Optimization Parameters

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--optimizer` | str | `ifgsm` | Optimization algorithm. Choices: `adam`, `sgd`, `ifgsm`, `clipped-pgd`, `lbfgs` |
| `--lr` | float | `0.1` | Learning rate for optimizer |
| `--max_delta` | float | `0.008` | Maximum delta for parameter updates (gradient clipping) |
| `--n` | int | `50` | Number of training epochs for patch optimization |
| `--steps` | int | `20` | Number of steps for iterative optimization per epoch |

### Loss Function Parameters

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--loss` | str list | `aee` | Optical flow loss metrics to optimize. Format: `name` or `name:weight`. Choices: `aee`, `cosim`, `mse`, `focal`, `huber`, `charbonnier`. Example: `--loss aee:1.0 mse:0.5` |
| `--normalize_losses` | flag | `False` | Normalize loss weights to sum to 1.0 |
| `--loss_weights` | float list (3 values) | `1.0 0.1 1.0` | Weights for [optical_flow_loss, MDE_loss, SS_loss] |
| `--tv_weight` | float | `0.0` | Weight for total variation regularization (encourages smooth patches) |
| `--nps_weight` | float | `0.0` | Weight for non-printability score regularization (for physical patches) |

### Monocular Depth Estimation (MDE) Attack

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--attack_mde` | flag | `False` | Attack MDE model simultaneously with optical flow model |
| `--mde_model` | str | `depth-anything-v2` | MDE model. Choices: `depth-anything-v2`, `marigold` |
| `--mde_target` | str | `zero` | MDE attack target. Choices: `zero`, `untargeted`, `infinite`, `scene`, `p90` (90th percentile) |

### Semantic Segmentation (SS) Attack

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--attack_ss` | flag | `False` | Attack semantic segmentation model simultaneously |
| `--ss_model` | str | `pspnet_cityscapes` | SS model. Choices: `deeplabv3`, `pspnet_cityscapes`, `segformer_cityscapes`, `mask2former_cityscapes` |
| `--ss_target` | str | `targeted` | SS attack target. Choices: `targeted`, `untargeted` |

### Defense Mechanisms

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--defense` | str | `none` | Defense to evaluate. Choices: `none`, `lgs` (Local Gradient Smoothing), `ilp` (Inpainting-based) |
| `--k` | int | `16` | Block size for defense |
| `--o` | int | `8` | Overlap for block-based defense |
| `--t` | float | `0.15` | Blockwise filtering threshold |
| `--s` | float | `15.0` | Smoothing/scaling parameter |
| `--r` | int | `5` | Inpainting radius |

### General Parameters

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--small_run` | flag | `False` | Run on a small subset for testing/debugging |
| `--output_dir` | str | `experiment_data` | Directory to save results and artifacts |
| `--experiment_name` | str | `attack_experiment` | Name for MLflow experiment tracking |
| `--save_artifacts` | flag | `False` | Save attacked images, flows, depth maps, and segmentation maps |
| `--epsilon` | float | `8/255` | Perturbation budget (for reference, not used in patch attacks) |
| `--alpha` | float | `0.01` | Step size for iterative methods |

---

## Examples

### Example 1: Basic Patch Attack on RAFT

Train a patch to fool RAFT optical flow model:

```bash
python run_patch_attack.py \
  --dataset Kitti15 \
  --model_name raft \
  --patch_size 100 \
  --optimizer adam \
  --lr 0.1 \
  --n 50 \
  --target zero \
  --save_artifacts \
  --output_dir results/basic_patch
```

**What this does:**
- Trains an adversarial patch of 100×100 pixels
- Uses Adam optimizer with learning rate 0.1
- Trains for 50 epochs
- Goal: Make optical flow predictions become zero
- Saves generated patches and attacked images

### Example 2: Multi-Model Attack with 3D Projection

```bash
python run_patch_attack.py \
  --dataset Kitti15 \
  --model_name raft \
  --patch_size 150 \
  --attack_mde \
  --attack_ss \
  --patch_projection \
  --optimizer adam \
  --lr 0.1 \
  --n 100 \
  --loss_weights 1.0 0.5 0.5 \
  --mde_model depth-anything-v2 \
  --ss_model segformer_cityscapes \
  --save_artifacts \
  --output_dir results/multi_model_attack
```

**What this does:**
- Trains a patch against three models simultaneously
- Uses 3D projection to apply patch realistically to scene geometry
- Balances attack on OF (weight 1.0), MDE (weight 0.5), SS (weight 0.5)
- Saves all intermediate results

### Example 3: Smooth Patch with Regularization

```bash
python run_patch_attack.py \
  --dataset Sintel \
  --model_name gma \
  --patch_size 80 \
  --optimizer adam \
  --lr 0.05 \
  --n 150 \
  --loss aee:1.0 \
  --tv_weight 100 \
  --nps_weight 50 \
  --save_artifacts \
  --output_dir results/smooth_patch
```

**What this does:**
- Creates a more visually smooth and print-friendly patch
- `tv_weight=100`: Total variation regularization for smoothness
- `nps_weight=50`: Non-printability score for physical realizability
- Trains longer (150 epochs) for better convergence

### Example 4: Test Pre-trained Patch

Use an existing trained patch without retraining:

```bash
python run_patch_attack.py \
  --dataset Kitti15 \
  --model_name raft \
  --trained_patch results/basic_patch/patch.png \
  --patch_size 100 \
  --attack_mde \
  --output_dir results/test_pretrained
```

**What this does:**
- Loads a pre-trained patch from file
- Evaluates it on the test set
- Tests against both optical flow and MDE models
- No training occurs

### Example 5: Quick Test Run

```bash
python run_patch_attack.py \
  --small_run \
  --dataset Kitti15 \
  --model_name raft \
  --patch_size 100 \
  --n 2 \
  --optimizer adam \
  --save_artifacts
```

**What this does:**
- Runs on a small subset of data (useful for debugging)
- Trains for only 2 epochs
- Perfect for testing setup and pipelines

---

## Understanding Loss Functions

### Optical Flow Loss Options

- **aee** (Average Endpoint Error): L2 distance between predicted and target flow
- **cosim** (Cosine Similarity): Angle-based distance, robust to magnitude changes
- **mse** (Mean Squared Error): Pixel-wise L2 loss
- **focal** (Focal Loss): Focuses on hard-to-predict pixels
- **huber** (Huber Loss): Smooth L1 variant, robust to outliers
- **charbonnier** (Charbonnier Loss): Smooth version of L1 loss

### Attack Targets

- **zero**: Make all flow predictions equal to zero
- **neg_flow**: Make flow predictions negative (reverse direction)
- **untargeted**: Maximize flow error without specific target
- **scene**: Target based on scene geometry
- **camera**: Target related to camera motion
- **down**: Force downward motion

---

## Output and Artifacts

When running with `--save_artifacts`, the following are saved:

- **Patches**: `patch.png` - Visual representation of the trained patch
- **Attacked Images**: `batch_*_attacked_image.png` - Images with applied patch
- **Optical Flow**: `batch_*_attacked_flow.flo` - Predicted flow after attack
- **Depth Maps**: `batch_*_attacked_depth.npy` - Predicted depth (if `--attack_mde`)
- **Segmentation**: `batch_*_attacked_ss.png` - Segmentation maps (if `--attack_ss`)
- **Metrics**: `metrics.json` - Attack success metrics