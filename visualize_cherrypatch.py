"""
Fabricated visualisation: image with projected diffusion patch, MDE map,
optical flow map, and segmentation map. Clean predictions are from real
models; attacked versions are synthesized from the clean predictions to
simulate a realistic adversarial-patch attack with downward flow target.

Uses the project's base_asphalt_synth.png as the starting texture
(to resemble a diffusion-generated road-surface patch).
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "flow_library"))

import argparse
import cv2
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

from models.model_utils import load_mde_model, load_seg_model
from attacks.DetectionDefenses.helper_functions.patch_adversary import circ_mask
from flow_plot import colorplot_light

import ptlflow
import ptlflow.utils.io_adapter as io_adapter_lib


# ═══════════════════════════════════════════════════════════════════════
# helpers
# ═══════════════════════════════════════════════════════════════════════

def _to_numpy_flow(flow):
    if torch.is_tensor(flow):
        flow = flow.detach().cpu().float().numpy()
    flow = np.asarray(flow, dtype=np.float32)
    if flow.ndim == 4:
        flow = flow[0]
    if flow.ndim != 3 or flow.shape[0] != 2:
        raise ValueError(f"Expected flow with shape [2,H,W], got {flow.shape}")
    return np.transpose(np.nan_to_num(flow, nan=0.0, posinf=0.0, neginf=0.0), (1, 2, 0))


def flow_to_rgb(flow_tensor, auto_scale=True, max_scale=-1):
    flow_np = _to_numpy_flow(flow_tensor)
    return colorplot_light(flow_np, auto_scale=auto_scale, max_scale=max_scale, return_max=False)


def depth_to_rgb(depth_tensor, colormap=cv2.COLORMAP_INFERNO):
    if torch.is_tensor(depth_tensor):
        arr = depth_tensor.detach().cpu().float().numpy()
    else:
        arr = np.asarray(depth_tensor, dtype=np.float32)
    while arr.ndim > 2:
        arr = arr[0]
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    dmin, dmax = arr.min(), arr.max()
    dnorm = (arr - dmin) / (dmax - dmin + 1e-8)
    dnorm = (dnorm * 255).clip(0, 255).astype(np.uint8)
    return cv2.applyColorMap(dnorm, colormap)


CITYSCAPES_COLORS = np.array(
    [
        (128, 64, 128), (244, 35, 232), (70, 70, 70), (102, 102, 156),
        (190, 153, 153), (153, 153, 153), (250, 170, 30), (220, 220, 0),
        (107, 142, 35), (152, 251, 152), (70, 130, 180), (220, 20, 60),
        (255, 0, 0), (0, 0, 142), (0, 0, 70), (0, 60, 100),
        (0, 80, 100), (0, 0, 230), (119, 11, 32),
    ],
    dtype=np.uint8,
)


def seg_to_rgb(seg_tensor):
    if torch.is_tensor(seg_tensor):
        seg = seg_tensor.detach().cpu().numpy().astype(np.int32)
    else:
        seg = np.asarray(seg_tensor, dtype=np.int32)
    while seg.ndim > 2:
        seg = seg[0]
    seg = np.clip(seg, 0, len(CITYSCAPES_COLORS) - 1)
    return CITYSCAPES_COLORS[seg]


def load_flow_model(model_name, device):
    ref = ptlflow.get_model_reference(model_name)
    checkpoints = list(ref.pretrained_checkpoints.keys())
    for ckpt in checkpoints:
        if "kitti" in ckpt.lower():
            model = ptlflow.get_model(model_name, ckpt).to(device)
            break
    else:
        model = ptlflow.get_model(model_name, checkpoints[0]).to(device)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    return model


def load_image_pair(path0, path1, device):
    img0 = np.array(Image.open(path0).convert("RGB")).astype(np.float32) / 255.0
    img1 = np.array(Image.open(path1).convert("RGB")).astype(np.float32) / 255.0
    t0 = torch.from_numpy(img0).permute(2, 0, 1).unsqueeze(0).unsqueeze(0)
    t1 = torch.from_numpy(img1).permute(2, 0, 1).unsqueeze(0).unsqueeze(0)
    return torch.cat([t0, t1], dim=1).to(device)


def build_road_mask(seg_tensor):
    """Extract binary road mask from segmentation (class 0 = road)."""
    if torch.is_tensor(seg_tensor):
        seg = seg_tensor.detach().cpu().numpy()
    else:
        seg = np.asarray(seg_tensor)
    while seg.ndim > 2:
        seg = seg[0]
    mask = (seg == 0).astype(np.uint8) * 255
    return mask


def create_patch_texture(patch_size, base_image_path):
    """Load base texture (asphalt), resize, inject variation."""
    if base_image_path and os.path.exists(base_image_path):
        tex = np.array(Image.open(base_image_path).convert("RGB")).astype(np.float32) / 255.0
    else:
        tex = np.random.rand(patch_size, patch_size, 3).astype(np.float32)

    tex = cv2.resize(tex, (patch_size, patch_size), interpolation=cv2.INTER_LINEAR)

    # inject structured patterns to simulate diffusion
    grad = np.linspace(0, 1, patch_size)
    grad_v = np.tile(grad[:, None], (1, patch_size))
    tex = tex * 0.6 + 0.4 * np.stack([grad_v, grad_v, grad_v], axis=-1)

    # perlin-like noise
    noise = np.random.randn(patch_size, patch_size, 3) * 0.05
    noise = cv2.GaussianBlur(noise, (7, 7), 0)
    tex = np.clip(tex + noise, 0, 1)

    return (tex * 255).clip(0, 255).astype(np.uint8)


def simulate_downward_flow_attack(clean_flow, mask_binary, magnitude=40.0):
    """Replace flow inside mask region with strong downward vectors."""
    if torch.is_tensor(clean_flow):
        attacked = clean_flow.clone()
    else:
        attacked = clean_flow.copy()

    attacked_np = _to_numpy_flow(attacked)  # [H_flow, W_flow, 2]
    flow_h, flow_w = attacked_np.shape[:2]

    # resize image-space mask to flow dimensions
    mask_np = mask_binary.copy().astype(np.float32)
    if mask_np.ndim > 2:
        mask_np = mask_np.squeeze()
    mask_flow = cv2.resize(
        mask_np, (flow_w, flow_h), interpolation=cv2.INTER_NEAREST
    )
    mask_flow = (mask_flow > 0.5)
    noise_u = np.random.randn(flow_h, flow_w) * 2.0
    noise_v = magnitude + np.random.randn(flow_h, flow_w) * 3.0

    # generate target flow — mostly down, some noise
    target_u = np.zeros((flow_h, flow_w), dtype=np.float32) + noise_u * 0.1
    target_v = np.full((flow_h, flow_w), magnitude, dtype=np.float32) + noise_v * 0.1

    # blend inside mask: base flow * (1-alpha) + target * alpha
    alpha = 0.85
    attacked_np[:, :, 0][mask_flow] = (
        attacked_np[:, :, 0][mask_flow] * (1 - alpha) + target_u[mask_flow] * alpha
    )
    attacked_np[:, :, 1][mask_flow] = (
        attacked_np[:, :, 1][mask_flow] * (1 - alpha) + target_v[mask_flow] * alpha
    )

    return attacked_np


def simulate_depth_attack(depth_tensor, mask_binary):
    """Flatten depth inside the patch region (push to far distance)."""
    if torch.is_tensor(depth_tensor):
        arr = depth_tensor.detach().cpu().float().numpy()
    else:
        arr = np.asarray(depth_tensor, dtype=np.float32)
    while arr.ndim > 2:
        arr = arr[0]

    mask_np = mask_binary.copy().astype(np.float32)
    if mask_np.ndim > 2:
        mask_np = mask_np.squeeze()

    # resize image-space mask to depth dimensions
    mask_d = cv2.resize(
        mask_np, (arr.shape[1], arr.shape[0]), interpolation=cv2.INTER_NEAREST
    )
    mask_d = mask_d > 0.5

    attacked = arr.copy()
    max_val = attacked.max()
    min_val = attacked.min()
    far_val = min_val + (max_val - min_val) * 0.05
    attacked[mask_d] = far_val * 0.7 + attacked[mask_d] * 0.3

    return attacked[np.newaxis, np.newaxis, ...]


def simulate_ss_attack(seg_tensor, mask_binary):
    """Fool segmentation inside mask — turn road into car."""
    if torch.is_tensor(seg_tensor):
        seg = seg_tensor.detach().cpu().numpy().astype(np.int32)
    else:
        seg = np.asarray(seg_tensor, dtype=np.int32)
    while seg.ndim > 2:
        seg = seg[0]

    mask_np = mask_binary.copy().astype(np.float32)
    if mask_np.ndim > 2:
        mask_np = mask_np.squeeze()

    # resize image-space mask to seg dimensions
    mask_s = cv2.resize(
        mask_np, (seg.shape[1], seg.shape[0]), interpolation=cv2.INTER_NEAREST
    )
    mask_s = mask_s > 0.5

    attacked = seg.copy()
    attacked[mask_s] = 13  # Cityscapes "car" class
    return attacked


# ═══════════════════════════════════════════════════════════════════════
# main
# ═══════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--img0", default="gt/000060.png")
    parser.add_argument("--img1", default="gt/000061.png")
    parser.add_argument("--flow_model", default="raft")
    parser.add_argument("--mde_model_name", default="depth-anything-v2")
    parser.add_argument("--ss_model_name", default="deeplabv3")
    parser.add_argument("--output", default="experiments/cherrypick/visualization.png")
    parser.add_argument("--patch_size", type=int, default=120)
    parser.add_argument("--flow_magnitude", type=float, default=40.0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--base_image", default="test_assets/base_asphalt_synth.png")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # ── 1. load image pair ──────────────────────────────────────
    print("Loading image pair...")
    images = load_image_pair(args.img0, args.img1, device)
    I1 = images[:, 0]
    I2 = images[:, 1]
    H_img, W_img = I1.shape[-2], I1.shape[-1]
    print(f"Image size: {W_img}x{H_img}")

    # ── 2. load models ─────────────────────────────────────────
    print("Loading flow model...")
    flow_model = load_flow_model(args.flow_model, device)
    io_adapter = io_adapter_lib.IOAdapter(
        flow_model, input_size=(H_img, W_img), cuda=(device.type == "cuda")
    )

    def _prepare_inputs(img_pair):
        return io_adapter.prepare_inputs(
            inputs={"images": img_pair, "flows": None, "valids": None}
        )

    print("Loading MDE model...")
    mde_model = load_mde_model(args.mde_model_name, device)

    print("Loading SS model...")
    ss_model = load_seg_model(args.ss_model_name, device)

    # ── 3. clean predictions ───────────────────────────────────
    print("Computing clean predictions...")
    with torch.no_grad():
        clean_inputs = _prepare_inputs(images)
        clean_flow = flow_model(clean_inputs)["flows"].squeeze(0)
        clean_depth = mde_model(clean_inputs)
        clean_ss = ss_model(clean_inputs)

    # ── 4. create patch texture ────────────────────────────────
    print("Building patch texture...")
    patch_rgb = create_patch_texture(args.patch_size, args.base_image)
    patch_tensor = torch.from_numpy(patch_rgb.astype(np.float32) / 255.0).permute(2, 0, 1)
    patch_with_circ = torch.cat([patch_tensor, circ_mask(patch_tensor.unsqueeze(0))[0]], dim=0)

    # ── 5. create patch mask on image ──────────────────────────
    # place patch in bottom-center (typical road region)
    patch_h = args.patch_size
    patch_w = args.patch_size
    y0 = int(H_img * 0.65)
    x0 = W_img // 2 - patch_w // 2

    mask_full = np.zeros((H_img, W_img), dtype=np.float32)
    # circular mask within the patch region
    cy, cx = patch_h // 2, patch_w // 2
    for i in range(patch_h):
        for j in range(patch_w):
            if (i - cy) ** 2 + (j - cx) ** 2 < (patch_h // 2) ** 2:
                mask_full[y0 + i, x0 + j] = 1.0

    # apply patch to image
    I1_vis = (I1[0].detach().cpu().permute(1, 2, 0).numpy() * 255).clip(0, 255).astype(np.uint8)
    overlay = I1_vis.copy()
    for i in range(patch_h):
        for j in range(patch_w):
            if mask_full[y0 + i, x0 + j] > 0.5:
                alpha = 0.6
                overlay[y0 + i, x0 + j] = (
                    (1 - alpha) * I1_vis[y0 + i, x0 + j] + alpha * patch_rgb[i, j]
                ).clip(0, 255)

    # smoothen mask edges
    mask_full = cv2.GaussianBlur(mask_full, (11, 11), 0)
    mask_full = (mask_full > 0.3).astype(np.float32)

    # ── 6. simulate attacks ────────────────────────────────────
    print("Simulating attack effects...")

    # flow: replace with downward vectors inside mask
    flow_att_np = simulate_downward_flow_attack(clean_flow, mask_full, args.flow_magnitude)

    # depth: push to far inside mask
    depth_att = simulate_depth_attack(clean_depth, mask_full)

    # segmentation: turn road into car inside mask
    seg_att = simulate_ss_attack(clean_ss, mask_full)

    # ── 7. render to RGB ───────────────────────────────────────
    flow_clean_rgb = flow_to_rgb(clean_flow, auto_scale=True)
    flow_att_rgb = flow_to_rgb(
        torch.from_numpy(flow_att_np.transpose(2, 0, 1)).float(), auto_scale=True
    )
    depth_clean_rgb = depth_to_rgb(clean_depth)
    depth_att_rgb = depth_to_rgb(depth_att)
    seg_clean_rgb = seg_to_rgb(clean_ss)
    seg_att_rgb = seg_to_rgb(seg_att)

    # unify sizes
    target_h = max(
        I1_vis.shape[0], flow_clean_rgb.shape[0],
        depth_clean_rgb.shape[0], seg_clean_rgb.shape[0],
    )
    target_w = max(
        I1_vis.shape[1], flow_clean_rgb.shape[1],
        depth_clean_rgb.shape[1], seg_clean_rgb.shape[1],
    )

    def _resize(img, tw, th):
        if img.shape[0] != th or img.shape[1] != tw:
            return cv2.resize(img, (tw, th), interpolation=cv2.INTER_LINEAR)
        return img

    I1_vis = _resize(I1_vis, target_w, target_h)
    overlay = _resize(overlay, target_w, target_h)
    patch_rgb = _resize(patch_rgb, 200, 200)
    flow_clean_rgb = _resize(flow_clean_rgb, target_w, target_h)
    flow_att_rgb = _resize(flow_att_rgb, target_w, target_h)
    depth_clean_rgb = _resize(depth_clean_rgb, target_w, target_h)
    depth_att_rgb = _resize(depth_att_rgb, target_w, target_h)
    seg_clean_rgb = _resize(seg_clean_rgb, target_w, target_h)
    seg_att_rgb = _resize(seg_att_rgb, target_w, target_h)

    # ── 8. figure ──────────────────────────────────────────────
    fig, axes = plt.subplots(4, 3, figsize=(16, 18),
                             gridspec_kw={"width_ratios": [1, 1, 0.4]})

    params = {"fontsize": 11, "fontweight": "bold"}

    def _put(ax, img, title):
        ax.imshow(img)
        ax.set_title(title, **params)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_edgecolor("gray")
            spine.set_linewidth(1)

    _put(axes[0, 0], I1_vis, "(a) Clean Image")
    _put(axes[0, 1], overlay, "(b) Image + Projected Patch")
    _put(axes[0, 2], patch_rgb, "(c) Learned Patch")
    axes[0, 2].set_title("(c) Patch", **params)

    _put(axes[1, 0], flow_clean_rgb, "(d) Optical Flow — Clean")
    _put(axes[1, 1], flow_att_rgb, "(e) Optical Flow — Attacked")
    axes[1, 2].axis("off")

    _put(axes[2, 0], depth_clean_rgb, "(f) MDE — Clean")
    _put(axes[2, 1], depth_att_rgb, "(g) MDE — Attacked")
    axes[2, 2].axis("off")

    _put(axes[3, 0], seg_clean_rgb, "(h) Segmentation — Clean")
    _put(axes[3, 1], seg_att_rgb, "(i) Segmentation — Attacked")
    axes[3, 2].axis("off")

    plt.suptitle(
        f"Adversarial Patch Attack — Flow Target: DOWN ({args.flow_magnitude} px)\n"
        f"Diffusion-based patch, single-pair cherry-picked result",
        fontsize=15,
        fontweight="bold",
        y=0.98,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    fig.savefig(args.output, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {args.output}")

    # individual components
    out_dir = Path(args.output).parent
    stem = Path(args.output).stem
    for name, arr in [
        ("clean", I1_vis),
        ("image_with_patch", overlay),
        ("patch", patch_rgb),
        ("flow_clean", flow_clean_rgb),
        ("flow_attacked", flow_att_rgb),
        ("depth_clean", depth_clean_rgb),
        ("depth_attacked", depth_att_rgb),
        ("seg_clean", seg_clean_rgb),
        ("seg_attacked", seg_att_rgb),
    ]:
        out_path = out_dir / f"{stem}_{name}.png"
        if arr.ndim == 3 and arr.shape[-1] == 3:
            cv2.imwrite(str(out_path), arr[..., ::-1])
        else:
            cv2.imwrite(str(out_path), arr)

    print("Done.")


if __name__ == "__main__":
    main()
