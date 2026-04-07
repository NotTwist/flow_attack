#!/usr/bin/env python3
"""
Run optical flow (ptlflow), MDE, and semantic segmentation on a video (clean frames, no patch).

Example:
  python run_video_inference.py \\
    --video datasets/video_2026-03-25_16-55-56.mp4 \\
    --out_dir outputs/video_run1

Note: use ``from ptlflow.utils.io_adapter import IOAdapter`` — ``ptlflow.utils.io_adapter``
is not available as a nested attribute after ``import ptlflow`` alone.
"""
from __future__ import annotations

import argparse
import os

import cv2
import numpy as np
import ptlflow
import torch
from ptlflow.utils.io_adapter import IOAdapter
from tqdm import tqdm

from models.model_utils import load_mde_model, load_seg_model
from utils.process_images import quickvis_flow, save_depth, save_segmentation


def load_flow_model(model_name: str, dataset_hint: str):
    model_ref = ptlflow.get_model_reference(model_name)
    checkpoints = list(model_ref.pretrained_checkpoints.keys())
    for c in checkpoints:
        if c.lower() in dataset_hint.lower():
            return ptlflow.get_model(model_name, c)
        print(f"Using checkpoint from other dataset!: {c}")
        return ptlflow.get_model(model_name, c)
    raise RuntimeError(f"No checkpoint for {model_name}")


def read_all_frames_rgb01(video_path: str, max_frames: int | None):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")
    frames = []
    while True:
        ret, bgr = cap.read()
        if not ret:
            break
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        t = torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0
        frames.append(t)
        if max_frames is not None and len(frames) >= max_frames:
            break
    cap.release()
    if len(frames) < 2:
        raise RuntimeError("Need at least 2 frames for optical flow.")
    return frames


def parse_args():
    p = argparse.ArgumentParser(description="Video inference: flow + MDE + SS (no patch)")
    p.add_argument("--video", type=str, required=True, help="Path to .mp4 (or other OpenCV-readable video)")
    p.add_argument("--out_dir", type=str, default="video_inference_out", help="Directory for outputs")
    p.add_argument("--model_name", type=str, default="raft", help="ptlflow model name")
    p.add_argument(
        "--dataset",
        type=str,
        default="kitti15",
        help="Substring to pick pretrained checkpoint (e.g. kitti15, sintel)",
    )
    p.add_argument("--mde_model", type=str, default="depth-anything-v2")
    p.add_argument("--ss_model", type=str, default="pspnet_cityscapes")
    p.add_argument("--max_pairs", type=int, default=None, help="Max frame pairs to process (default: all)")
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.out_dir, exist_ok=True)

    frames = read_all_frames_rgb01(args.video, None)
    n_pairs = min(len(frames) - 1, args.max_pairs) if args.max_pairs else len(frames) - 1

    flow_model = load_flow_model(args.model_name, args.dataset).to(device)
    flow_model.eval()
    for p in flow_model.parameters():
        p.requires_grad = False

    mde_model = load_mde_model(model_name=args.mde_model, device=device)
    ss_model = load_seg_model(model_name=args.ss_model, device=device)

    H, W = frames[0].shape[1], frames[0].shape[2]
    io_adapter = IOAdapter(
        flow_model,
        input_size=(H, W),
        cuda=torch.cuda.is_available(),
    )

    for idx in tqdm(range(n_pairs), desc="pairs"):
        I1 = frames[idx].unsqueeze(0).to(device)
        I2 = frames[idx + 1].unsqueeze(0).to(device)
        pair = torch.stack([I1, I2], dim=1).contiguous()

        flow_gt = torch.zeros(1, 2, H, W, device=device)
        valid = torch.ones(1, H, W, device=device, dtype=torch.bool)
        wrapped = {"images": pair, "flows": flow_gt, "valids": valid}
        inputs = io_adapter.prepare_inputs(inputs=wrapped)

        with torch.no_grad():
            flow_pred = flow_model(inputs)["flows"].squeeze(0)
            depth_pred = mde_model(inputs)
            ss_logits = ss_model(inputs, return_logits=True)
            ss_mask = ss_logits.argmax(dim=1)

        stem = f"pair_{idx:05d}"
        quickvis_flow(flow_pred, os.path.join(args.out_dir, f"{stem}_flow.png"))
        save_depth(depth_pred, os.path.join(args.out_dir, f"{stem}_depth.png"))
        save_segmentation(ss_mask, os.path.join(args.out_dir, f"{stem}_seg.png"))

        mid = (I1.squeeze(0).cpu().clamp(0, 1).numpy() * 255).astype(np.uint8).transpose(1, 2, 0)
        cv2.imwrite(
            os.path.join(args.out_dir, f"{stem}_frame0.png"),
            cv2.cvtColor(mid, cv2.COLOR_RGB2BGR),
        )

    print(f"Done. Wrote {n_pairs} triples to {args.out_dir}")


if __name__ == "__main__":
    main()
