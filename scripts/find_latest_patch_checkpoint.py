#!/usr/bin/env python3
"""Print the latest usable patch checkpoint metadata path.

This small helper is used by experiment shell scripts that need to evaluate
the latest trained diffusion or pixel patch without duplicating discovery
logic in bash.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path


PREFIXES = {
    "diffusion": "diffusion_patch_kitti15_no_projection",
    "pixel": "pixel_patch_kitti15_no_projection",
}


def epoch_from_name(path: Path) -> int:
    match = re.search(r"patch_epoch_(\d+)", path.name)
    return int(match.group(1)) if match else -1


def step_from_metadata_name(path: Path) -> int:
    match = re.search(r"step_(\d+)", path.name)
    if match:
        return int(match.group(1))
    return epoch_from_name(path) * 1_000_000


def patch_exists(metadata_path: Path) -> bool:
    png_path = metadata_path.with_suffix(".png")
    return png_path.exists()


def is_completed_epoch(path: Path) -> bool:
    return re.fullmatch(r"patch_epoch_\d+\.json", path.name) is not None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kind", choices=sorted(PREFIXES), required=True)
    parser.add_argument("--experiment_root", type=Path, default=Path("experiment_data"))
    parser.add_argument("--run_dir", type=Path, default=None)
    parser.add_argument("--allow_partial", action="store_true")
    args = parser.parse_args()

    if args.run_dir is not None:
        run_dirs = [args.run_dir]
    else:
        prefix = PREFIXES[args.kind]
        run_dirs = [p for p in args.experiment_root.glob(f"{prefix}_*") if (p / "patch_checkpoints").is_dir()]
        run_dirs.sort(key=lambda p: (p.stat().st_mtime, p.name), reverse=True)

    for run_dir in run_dirs:
        checkpoint_dir = run_dir / "patch_checkpoints"
        metadata = sorted(checkpoint_dir.glob("patch_epoch_*.json"))
        metadata = [p for p in metadata if patch_exists(p)]
        completed = [p for p in metadata if is_completed_epoch(p)]
        if completed:
            completed.sort(key=epoch_from_name)
            print(completed[-1])
            return
        if args.allow_partial and metadata:
            metadata.sort(key=step_from_metadata_name)
            print(metadata[-1])
            return

    raise SystemExit(f"No usable {args.kind} checkpoint found.")


if __name__ == "__main__":
    main()
