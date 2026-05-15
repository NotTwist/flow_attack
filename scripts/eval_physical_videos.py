#!/usr/bin/env python3
"""
Physical adversarial patch video evaluator.

Expected input structure:
  <root>/
    <condition>/          e.g. bright | dark | angle
      diffusion.mp4
      pixel.mp4
      diffusion-baseline.mp4
      pixel-baseline.mp4

Outputs per condition (written to --out):
  <out>/<condition>/<patch_type>_demo.mp4   — 4-panel: original | flow | depth | seg
  <out>/<condition>/timeseries.png          — per-frame metric line plots
  <out>/summary.png                         — aggregated bar chart across conditions

Usage:
  python scripts/eval_physical_videos.py /path/to/recordings --out physical_eval
"""

import argparse
import contextlib
import io
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tqdm import tqdm


_DEVNULL = open(os.devnull, 'w')  # opened once; reused by _quiet() throughout

@contextlib.contextmanager
def _quiet():
    """Redirect stdout to /dev/null to suppress third-party model prints."""
    with contextlib.redirect_stdout(_DEVNULL):
        yield

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))

import ptlflow
import ptlflow.utils.io_adapter as io_adapter_lib
from ptlflow.utils import flow_utils
from models.model_utils import load_mde_model, load_seg_model
from utils.process_images import color_map as _TAB20, classes as _CS_CLASSES

# ── constants ────────────────────────────────────────────────────────────────

PATCH_TYPES = ['diffusion', 'pixel', 'diffusion-baseline', 'pixel-baseline']
PATCH_COLORS = {
    'diffusion':          '#e41a1c',
    'pixel':              '#377eb8',
    'diffusion-baseline': '#ff7f00',
    'pixel-baseline':     '#4daf4a',
}
PATCH_LABELS = {
    'diffusion':          'Diffusion (adv)',
    'pixel':              'Pixel (adv)',
    'diffusion-baseline': 'Diffusion baseline',
    'pixel-baseline':     'Pixel baseline',
}
VIDEO_EXTENSIONS = ['.mp4', '.avi', '.mov', '.MP4', '.AVI', '.MOV']

# Cityscapes 19-class palette (RGB)
_CS_PALETTE = np.array([
    [128,  64, 128], [244,  35, 232], [ 70,  70,  70], [102, 102, 156],
    [190, 153, 153], [153, 153, 153], [250, 170,  30], [220, 220,   0],
    [107, 142,  35], [152, 251, 152], [ 70, 130, 180], [220,  20,  60],
    [255,   0,   0], [  0,   0, 142], [  0,   0,  70], [  0,  60, 100],
    [  0,  80, 100], [  0,   0, 230], [119,  11,  32],
], dtype=np.uint8)

FONT = cv2.FONT_HERSHEY_SIMPLEX


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('root', help='Root folder containing condition subfolders')
    p.add_argument('--out', default='physical_eval_out', help='Output directory')
    p.add_argument('--flow_model', default='raft')
    p.add_argument('--mde_model', default='depth-anything-v2')
    p.add_argument('--ss_model', default='segformer_cityscapes')
    p.add_argument('--max_frames', type=int, default=0,
                   help='Process at most N frames per video (0=all)')
    p.add_argument('--stride', type=int, default=1,
                   help='Process every Nth frame (e.g. 6 = 10fps from 60fps source)')
    p.add_argument('--infer_size', type=str, default='',
                   help='Resize frames before model inference, e.g. 1242x375. '
                        'Outputs are resized back for visualization. '
                        'Default: use original resolution (slow on phone videos).')
    p.add_argument('--fps', type=int, default=10, help='Demo video FPS')
    p.add_argument('--panel_w', type=int, default=640, help='Width of each panel (px)')
    p.add_argument('--panel_h', type=int, default=360, help='Height of each panel (px)')
    p.add_argument('--no_video', action='store_true',
                   help='Skip demo video output (compute metrics only)')
    p.add_argument('--rotate_vertical', action='store_true',
                   help='Auto-rotate portrait videos (H > W) by 90°. '
                        'Useful for phone recordings saved without rotation metadata.')
    p.add_argument('--rotate_ccw', action='store_true',
                   help='Use counter-clockwise rotation instead of clockwise '
                        '(only relevant with --rotate_vertical)')
    p.add_argument('--gpu', type=int, default=0)
    return p.parse_args()


# ── model loading ─────────────────────────────────────────────────────────────

def load_models(args, device):
    print("Loading optical flow model…")
    model_ref = ptlflow.get_model_reference(args.flow_model)
    ckpt = next(iter(model_ref.pretrained_checkpoints))
    flow_model = ptlflow.get_model(args.flow_model, ckpt).to(device).eval()
    for p in flow_model.parameters():
        p.requires_grad_(False)

    print("Loading MDE model…")
    mde_model = load_mde_model(args.mde_model, device=device)

    print("Loading segmentation model…")
    ss_model = load_seg_model(args.ss_model, device=device)
    for p in ss_model.parameters():
        p.requires_grad_(False)

    return flow_model, mde_model, ss_model


# ── per-frame inference ───────────────────────────────────────────────────────

def frame_to_tensor(frame_bgr, device):
    """BGR uint8 HxWxC → float32 (1,3,H,W) RGB in [0,1]."""
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    return torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0).to(device)


def make_io_adapter(flow_model, frame_hw, device):
    """Create IOAdapter once per video — reused across all frames."""
    H, W = frame_hw
    return io_adapter_lib.IOAdapter(flow_model, input_size=(H, W), cuda=(device.type == 'cuda'))


def compute_flow(flow_model, io_adapter, prev_t, curr_t):
    """Returns numpy (H,W,2). prev_t/curr_t are (1,3,H,W).

    IOAdapter.prepare_inputs expects numpy HWC images — passing a pre-built
    tensor dict triggers flow_transforms which calls .shape on the dict and crashes.
    io_adapter is created once per video (see make_io_adapter).
    """
    prev_np = prev_t.squeeze(0).permute(1, 2, 0).cpu().numpy()  # (H,W,3) float [0,1]
    curr_np = curr_t.squeeze(0).permute(1, 2, 0).cpu().numpy()
    inputs = io_adapter.prepare_inputs(images=[prev_np, curr_np])
    with torch.no_grad():
        pred = flow_model(inputs)['flows']
    if pred.ndim == 5:
        pred = pred.squeeze(1)  # (N,1,2,H,W) → (N,2,H,W)
    return pred.squeeze(0).permute(1, 2, 0).cpu().numpy()  # (H,W,2)


def compute_depth(mde_model, frame_t):
    """Returns numpy (H,W). frame_t is (1,3,H,W)."""
    with torch.no_grad():
        d = mde_model({'images': frame_t.unsqueeze(0)})  # {'images': (1,1,3,H,W)}
    return d.squeeze().cpu().numpy()


def compute_seg(ss_model, frame_t):
    """Returns numpy (H,W) int32 class labels. frame_t is (1,3,H,W)."""
    with torch.no_grad():
        logits = ss_model(frame_t.squeeze(0), return_logits=True)  # (3,H,W) or (C,H,W)
        if logits.ndim == 3:
            logits = logits.unsqueeze(0)
    return logits.argmax(dim=1).squeeze(0).cpu().numpy().astype(np.int32)


# ── visualization helpers ─────────────────────────────────────────────────────

def colorize_flow(flow_hw2):
    """(H,W,2) float32 → BGR uint8 (H,W,3)."""
    try:
        rgb = flow_utils.flow_to_rgb(flow_hw2)
        if isinstance(rgb, torch.Tensor):
            rgb = (rgb.cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
        elif rgb.dtype != np.uint8:
            rgb = (rgb * 255).clip(0, 255).astype(np.uint8)
        if rgb.shape[-1] == 3:
            return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        return rgb
    except Exception:
        mag = np.sqrt(flow_hw2[..., 0] ** 2 + flow_hw2[..., 1] ** 2)
        p99 = np.percentile(mag, 99) + 1e-6
        norm = (np.clip(mag / p99, 0, 1) * 255).astype(np.uint8)
        return cv2.applyColorMap(norm, cv2.COLORMAP_JET)


def colorize_depth(depth_hw):
    """(H,W) float → BGR uint8 (H,W,3), MAGMA colormap."""
    lo, hi = depth_hw.min(), depth_hw.max()
    if hi > lo:
        norm = ((depth_hw - lo) / (hi - lo) * 255).astype(np.uint8)
    else:
        norm = np.zeros_like(depth_hw, dtype=np.uint8)
    return cv2.applyColorMap(norm, cv2.COLORMAP_MAGMA)


def colorize_seg(seg_hw, panel_w, panel_h):
    """Class label map (H,W) → BGR (panel_h, panel_w) with cityscapes class legend.

    Matches the tab20 colormap and legend style used in save_segmentation().
    Only classes present in this frame are listed in the legend.
    """
    # Colored segmentation mask at native resolution
    color_mask = _TAB20[seg_hw % len(_TAB20)]  # (H,W,3) uint8 RGB

    # Legend: only show classes present in this frame, sorted by id
    present = sorted(int(c) for c in np.unique(seg_hw) if int(c) < len(_CS_CLASSES))
    legend_w = 130
    swatch_h = max(14, panel_h // max(len(present), 1))
    swatch_h = min(swatch_h, 28)  # cap so legend doesn't grow too tall
    legend_h = swatch_h * len(present)

    legend = np.zeros((legend_h, legend_w, 3), dtype=np.uint8)
    for row, cls_id in enumerate(present):
        color = _TAB20[cls_id]
        y0, y1 = row * swatch_h, (row + 1) * swatch_h
        legend[y0:y1, :] = color
        brightness = int(color.mean())
        txt_color = (0, 0, 0) if brightness > 128 else (255, 255, 255)
        cv2.putText(legend, _CS_CLASSES[cls_id], (4, y0 + swatch_h - 4),
                    FONT, 0.35, txt_color, 1, cv2.LINE_AA)

    # Resize both to panel_h, then hstack
    seg_w = panel_w - legend_w
    seg_resized = cv2.resize(color_mask, (seg_w, panel_h), interpolation=cv2.INTER_NEAREST)
    legend_resized = cv2.resize(legend, (legend_w, panel_h), interpolation=cv2.INTER_NEAREST)

    combined = np.hstack([seg_resized, legend_resized])  # (panel_h, panel_w, 3) RGB
    return cv2.cvtColor(combined, cv2.COLOR_RGB2BGR)


def put_label(img, text, pos=(12, 32)):
    cv2.putText(img, text, pos, FONT, 0.75, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(img, text, pos, FONT, 0.75, (255, 255, 255), 1, cv2.LINE_AA)


def make_panel(img_bgr, label, w, h):
    panel = cv2.resize(img_bgr, (w, h), interpolation=cv2.INTER_LINEAR)
    put_label(panel, label)
    return panel


# ── main processing ───────────────────────────────────────────────────────────

def process_video(video_path, flow_model, mde_model, ss_model, device, args,
                  out_video_path):
    """
    Process one video file. Writes a 4-panel demo video and returns per-frame metrics:
      flow_mag   — mean optical flow magnitude (px/frame)
      depth_mean — mean predicted depth
      road_frac  — fraction of pixels classified as road (cityscapes class 0)
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open {video_path}")

    # Determine rotation: auto-detect portrait (H > W) when --rotate_vertical is set
    rotate_code = None
    if getattr(args, 'rotate_vertical', False):
        cap_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        cap_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if cap_h > cap_w:
            rotate_code = (cv2.ROTATE_90_COUNTERCLOCKWISE
                           if getattr(args, 'rotate_ccw', False)
                           else cv2.ROTATE_90_CLOCKWISE)
            direction = 'CCW' if rotate_code == cv2.ROTATE_90_COUNTERCLOCKWISE else 'CW'
            print(f"    portrait detected ({cap_w}×{cap_h}) → rotating 90° {direction}")

    W, H = args.panel_w, args.panel_h
    writer = None
    if not args.no_video:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(str(out_video_path), fourcc, args.fps, (W * 2, H * 2))

    # Parse inference resolution (for model inputs)
    infer_size_str = getattr(args, 'infer_size', '')
    if infer_size_str:
        iW, iH = (int(x) for x in infer_size_str.lower().replace('x', ' ').split())
        infer_hw = (iH, iW)
    else:
        infer_hw = None  # determined after reading first frame

    # Read one frame to get actual frame dimensions (may differ after rotation)
    ok, _probe = cap.read()
    if not ok:
        cap.release()
        return {'flow_mag': [], 'depth_mean': [], 'road_frac': []}
    if rotate_code is not None:
        _probe = cv2.rotate(_probe, rotate_code)
    fH, fW = _probe.shape[:2]
    if infer_hw is None:
        infer_hw = (fH, fW)
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # rewind

    # IOAdapter created once for this video (avoids per-frame CUDA allocation)
    io_adapter = make_io_adapter(flow_model, infer_hw, device)

    metrics = {'flow_mag': [], 'depth_mean': [], 'road_frac': []}
    prev_t = None
    frame_idx = 0

    stride = max(1, getattr(args, 'stride', 1))
    pbar = tqdm(desc=f"    {video_path.name}", unit='fr', leave=False)
    raw_idx = 0  # index into the raw video stream
    while True:
        ok, frame_bgr = cap.read()
        if not ok:
            break
        if args.max_frames > 0 and frame_idx >= args.max_frames:
            break

        # Skip frames according to stride
        if raw_idx % stride != 0:
            raw_idx += 1
            continue
        raw_idx += 1

        if rotate_code is not None:
            frame_bgr = cv2.rotate(frame_bgr, rotate_code)

        curr_t = frame_to_tensor(frame_bgr, device)

        # Resize for inference if requested
        iH, iW = infer_hw
        if (iH, iW) != (fH, fW):
            curr_infer = torch.nn.functional.interpolate(
                curr_t, size=(iH, iW), mode='bilinear', align_corners=False)
        else:
            curr_infer = curr_t

        # Optical flow (zero on first frame)
        with _quiet():
            if prev_t is not None:
                flow_small = compute_flow(flow_model, io_adapter, prev_t, curr_infer)
                # Scale flow vectors back to original frame pixel coordinates
                if (iH, iW) != (fH, fW):
                    flow = cv2.resize(flow_small, (fW, fH), interpolation=cv2.INTER_LINEAR)
                    flow[..., 0] *= fW / iW  # u component
                    flow[..., 1] *= fH / iH  # v component
                else:
                    flow = flow_small
            else:
                flow = np.zeros((fH, fW, 2), dtype=np.float32)

            depth_small = compute_depth(mde_model, curr_infer)
            depth = cv2.resize(depth_small, (fW, fH), interpolation=cv2.INTER_LINEAR) \
                    if (iH, iW) != (fH, fW) else depth_small

            seg_small = compute_seg(ss_model, curr_infer)
            seg = cv2.resize(seg_small.astype(np.uint8), (fW, fH),
                             interpolation=cv2.INTER_NEAREST).astype(np.int32) \
                  if (iH, iW) != (fH, fW) else seg_small

        prev_t = curr_infer  # keep inference-sized prev for next flow call

        # Metrics
        metrics['flow_mag'].append(float(np.mean(np.linalg.norm(flow, axis=-1))))
        metrics['depth_mean'].append(float(np.mean(depth)))
        metrics['road_frac'].append(float(np.mean(seg == 0)))

        # 4-panel composite
        if writer is not None:
            orig_panel  = make_panel(frame_bgr,             'Original',     W, H)
            flow_panel  = make_panel(colorize_flow(flow),   'Optical Flow', W, H)
            depth_panel = make_panel(colorize_depth(depth), 'Depth',        W, H)
            seg_panel   = colorize_seg(seg, W, H)
            put_label(seg_panel, 'Segmentation')
            grid = np.vstack([np.hstack([orig_panel, flow_panel]),
                              np.hstack([depth_panel, seg_panel])])
            writer.write(grid)

        frame_idx += 1
        pbar.update()

    pbar.close()
    cap.release()
    if writer is not None:
        writer.release()

    return metrics


# ── plotting ──────────────────────────────────────────────────────────────────

METRIC_META = [
    ('flow_mag',    'Mean Flow Magnitude (px)',     False),
    ('depth_mean',  'Mean Predicted Depth',         False),
    ('road_frac',   'Road Pixel Fraction',          True),
]


def plot_timeseries(all_metrics, condition, out_path):
    """Per-frame line plots for all patch types within one condition."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle(f'Condition: {condition}', fontsize=13, fontweight='bold')

    for ax, (mname, mlabel, _) in zip(axes, METRIC_META):
        for pt in PATCH_TYPES:
            vals = all_metrics.get(pt, {}).get(mname, [])
            if not vals:
                continue
            ax.plot(vals, label=PATCH_LABELS[pt],
                    color=PATCH_COLORS[pt], linewidth=1.5, alpha=0.85)
        ax.set_xlabel('Frame')
        ax.set_ylabel(mlabel)
        ax.set_title(mlabel, fontsize=10)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(str(out_path), dpi=150, bbox_inches='tight')
    plt.close(fig)


def plot_summary(summary, out_path):
    """Grouped bar chart: patch type × condition, one subplot per metric."""
    conditions = sorted(summary.keys())
    n_cond = len(conditions)
    n_pt = len(PATCH_TYPES)
    x = np.arange(n_cond)
    bar_w = 0.75 / n_pt

    fig, axes = plt.subplots(1, 3, figsize=(max(8, 4 * n_cond), 5), sharey=False)
    fig.suptitle('Physical Evaluation Summary', fontsize=14, fontweight='bold')

    for ax, (mname, mlabel, higher_better) in zip(axes, METRIC_META):
        for i, pt in enumerate(PATCH_TYPES):
            means, stds = [], []
            for cond in conditions:
                vals = summary.get(cond, {}).get(pt, {}).get(mname, [])
                means.append(np.mean(vals) if vals else 0.0)
                stds.append(np.std(vals) if vals else 0.0)
            offset = (i - n_pt / 2 + 0.5) * bar_w
            ax.bar(x + offset, means, bar_w, yerr=stds, capsize=3,
                   label=PATCH_LABELS[pt], color=PATCH_COLORS[pt], alpha=0.85)

        ax.set_xticks(x)
        ax.set_xticklabels(conditions, fontsize=10)
        ax.set_ylabel(mlabel, fontsize=10)
        ax.set_title(mlabel, fontsize=10)
        arrow = '↑ better' if higher_better else '↓ worse = stronger attack'
        ax.set_xlabel(arrow, fontsize=8, style='italic')
        ax.legend(fontsize=7)
        ax.grid(True, axis='y', alpha=0.3)

    plt.tight_layout()
    fig.savefig(str(out_path), dpi=150, bbox_inches='tight')
    plt.close(fig)


# ── entrypoint ────────────────────────────────────────────────────────────────

def find_video(folder, patch_type):
    for ext in VIDEO_EXTENSIONS:
        p = folder / f"{patch_type}{ext}"
        if p.exists():
            return p
    return None


def main():
    args = parse_args()
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    root = Path(args.root)
    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)

    _ORDER = ['bright', 'dark', 'angle']
    all_dirs = {d.name: d for d in root.iterdir() if d.is_dir()}
    conditions = [all_dirs[n] for n in _ORDER if n in all_dirs] + \
                 [d for d in sorted(all_dirs.values()) if d.name not in _ORDER]
    if not conditions:
        raise SystemExit(f"No subdirectories found in {root}")

    flow_model, mde_model, ss_model = load_models(args, device)

    summary = {}  # summary[condition][patch_type] = metrics_dict

    for cond_dir in conditions:
        condition = cond_dir.name
        print(f"\n── Condition: {condition} ──")
        out_cond = out_root / condition
        out_cond.mkdir(parents=True, exist_ok=True)

        cond_metrics = {}

        for pt in PATCH_TYPES:
            video_path = find_video(cond_dir, pt)
            if video_path is None:
                print(f"  [{pt}] no video found — skipping")
                continue

            print(f"  [{pt}] {video_path.name}")
            out_video = out_cond / f"{pt}_demo.mp4"
            try:
                m = process_video(
                    video_path, flow_model, mde_model, ss_model,
                    device, args, out_video,
                )
                cond_metrics[pt] = m
                if not args.no_video:
                    print(f"    → demo: {out_video}")
            except Exception as e:
                print(f"    ERROR: {e}")

        if cond_metrics:
            ts_path = out_cond / 'timeseries.png'
            plot_timeseries(cond_metrics, condition, ts_path)
            print(f"  → timeseries: {ts_path}")
            summary[condition] = cond_metrics

    if summary:
        summary_path = out_root / 'summary.png'
        plot_summary(summary, summary_path)
        print(f"\n── Summary chart: {summary_path}")

    print("Done.")


if __name__ == '__main__':
    main()
