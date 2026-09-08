from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from .manifest import discover_episodes
from .video import assigned_to_worker, probe_video, write_video


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Persistent OSCAR episode worker")
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--controls-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--oscar-repo", type=Path, required=True)
    parser.add_argument("--gpu-index", type=int, required=True)
    parser.add_argument("--worker-index", type=int, required=True)
    parser.add_argument("--worker-count", type=int, required=True)
    parser.add_argument("--episode-ids")
    parser.add_argument("--num-steps", type=int, default=5)
    parser.add_argument("--num-frames", type=int, default=81)
    parser.add_argument("--fps", type=float, default=24.0)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    return parser


def _selected_episode_ids(value: str | None) -> set[int] | None:
    if not value:
        return None
    return {int(item) for item in value.split(",") if item.strip()}


def _valid_existing(path: Path, frames: int, width: int, height: int) -> bool:
    if not path.is_file():
        return False
    try:
        probe = probe_video(path)
        return probe.frame_count == frames and (probe.width, probe.height) == (width, height)
    except Exception:
        return False


def main() -> int:
    args = build_parser().parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_index)
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ["MASTER_PORT"] = str(29500 + args.gpu_index)
    os.environ.setdefault("RANK", "0")
    os.environ.setdefault("LOCAL_RANK", "0")
    os.environ.setdefault("WORLD_SIZE", "1")
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    episodes = discover_episodes(args.dataset_root)
    selected = _selected_episode_ids(args.episode_ids)
    if selected is not None:
        episodes = [item for item in episodes if item.episode_id in selected]
    assigned_ids = set(
        assigned_to_worker(
            [item.episode_id for item in episodes], args.worker_index, args.worker_count
        )
    )
    episodes = [item for item in episodes if item.episode_id in assigned_ids]
    if not episodes:
        print(
            json.dumps(
                {
                    "event": "empty_assignment",
                    "gpu": args.gpu_index,
                    "worker_index": args.worker_index,
                }
            ),
            flush=True,
        )
        return 0

    args.output_dir.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(args.oscar_repo.resolve()))

    from oscar_diffusers import OSCARDiffusersPipeline

    print(json.dumps({"event": "loading_model", "gpu": args.gpu_index}), flush=True)
    pipeline = OSCARDiffusersPipeline.from_pretrained(str(args.checkpoint))
    print(json.dumps({"event": "model_loaded", "gpu": args.gpu_index}), flush=True)

    for position, episode in enumerate(episodes, 1):
        output = args.output_dir / episode.output_name
        if _valid_existing(output, args.num_frames, args.width, args.height):
            print(json.dumps({"event": "skip", "episode": episode.episode_id}), flush=True)
            continue
        control = args.controls_dir / episode.output_name
        if not control.is_file():
            raise FileNotFoundError(control)
        started = time.time()
        result = pipeline(
            first_frame=episode.first_frame_path,
            skeleton_video=control,
            prompt=episode.instruction,
            num_inference_steps=args.num_steps,
            guidance_scale=6.0,
            seed=episode.episode_id,
            num_frames=args.num_frames,
            height=args.height,
            width=args.width,
            fps=args.fps,
        )
        frames = np.stack(result.frames)
        first_frame = np.asarray(
            Image.open(episode.first_frame_path).convert("RGB").resize(
                (args.width, args.height), Image.Resampling.BILINEAR
            )
        )
        frames[0] = first_frame
        partial = output.with_name(f"{output.stem}.partial.mp4")
        write_video(frames.astype(np.uint8), partial, fps=args.fps)
        if not _valid_existing(partial, args.num_frames, args.width, args.height):
            raise ValueError(f"generated video failed validation: {partial}")
        os.replace(partial, output)
        print(
            json.dumps(
                {
                    "event": "complete",
                    "episode": episode.episode_id,
                    "position": position,
                    "assigned": len(episodes),
                    "seconds": round(time.time() - started, 2),
                }
            ),
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
