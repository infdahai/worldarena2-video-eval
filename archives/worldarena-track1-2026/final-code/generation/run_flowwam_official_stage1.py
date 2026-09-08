#!/usr/bin/env python3
"""Run released FlowWAM Stage-1 while explicitly disabling Stage-2 refinement."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

from worldarena_baseline.flowwam_official_pipeline import (
    OFFICIAL_FLOWWAM_COMMIT,
    official_stage1_contract,
)
from worldarena_baseline.flowwam_v14 import validate_checkpoint_file
from worldarena_baseline.flowwam_v16 import filter_v16_lora_state_for_inference

FORMAL_ROOT = Path("/data/di/worldarena2_track1_20260815")
OFFICIAL_SOURCE = Path("/home/huazhi/nlh/FlowWAM_WorldArena")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _under(path: Path, root: Path) -> Path:
    resolved = path.resolve(strict=False)
    resolved.relative_to(root)
    return resolved


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-root", type=Path, default=FORMAL_ROOT)
    parser.add_argument("--official-source", type=Path, default=OFFICIAL_SOURCE)
    parser.add_argument("--test-dataset-dir", type=Path, required=True)
    conditioning = parser.add_mutually_exclusive_group(required=True)
    conditioning.add_argument("--robot-only-dir", type=Path)
    conditioning.add_argument("--embodiment-dir", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--lora-checkpoint", type=Path)
    parser.add_argument("--lora-rank", type=int, default=8)
    parser.add_argument("--lora-alpha", type=float, default=8.0)
    parser.add_argument(
        "--lora-scale",
        type=float,
        choices=(0.5, 0.75, 1.0),
        default=1.0,
    )
    parser.add_argument(
        "--lora-attention-scope",
        choices=("all", "self"),
        default="all",
    )
    parser.add_argument(
        "--lora-source-layout",
        choices=("all-qv", "self-qv"),
        default="all-qv",
    )
    parser.add_argument("--local-model-path", type=Path, required=True)
    parser.add_argument("--expected-count", type=int, default=4)
    parser.add_argument("--max-episodes", type=int)
    parser.add_argument(
        "--sample-manifest",
        type=Path,
        help="Immutable JSONL rows naming the exact episode IDs for this shard.",
    )
    parser.add_argument(
        "--resume-valid-existing",
        action="store_true",
        help="Skip only existing MP4s that pass the frozen 121f/640x480/black0 gate.",
    )
    parser.add_argument("--instruction-variant", type=int, choices=(0, 1, 2), default=0)
    parser.add_argument(
        "--physical-gpu", type=int, choices=(0, 2, 3, 4, 5, 6), default=6
    )
    parser.add_argument("--seed", type=int, choices=(1, 2, 3, 4), default=1)
    parser.add_argument(
        "--flow-max-magnitude",
        type=float,
        choices=(16.0, 20.0, 24.0),
        default=20.0,
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def _validate_instruction_files(
    input_root: Path,
    *,
    instruction_variant: int,
    expected_count: int,
    episode_names: tuple[str, ...] | None = None,
) -> tuple[Path, ...]:
    """Fail closed before GPU startup when a prompt variant is incomplete."""
    subdir = (
        "instructions"
        if instruction_variant == 0
        else f"instructions_{instruction_variant}"
    )
    if episode_names is None:
        frames = sorted(
            (input_root / "first_frame" / "fixed_scene_task").glob("episode*.png")
        )[:expected_count]
        if len(frames) != expected_count:
            raise ValueError(
                f"expected {expected_count} first frames, found {len(frames)}"
            )
    else:
        frames = [
            input_root / "first_frame" / "fixed_scene_task" / f"{name}.png"
            for name in episode_names
        ]
        missing = [path for path in frames if not path.is_file()]
        if missing:
            raise FileNotFoundError(missing[0])
    prompts = []
    for frame in frames:
        prompt_path = input_root / subdir / "fixed_scene_task" / f"{frame.stem}.json"
        if not prompt_path.is_file():
            raise FileNotFoundError(prompt_path)
        payload = json.loads(prompt_path.read_text())
        if not str(payload.get("instruction", "")).strip():
            raise ValueError(f"empty instruction: {prompt_path}")
        prompts.append(prompt_path)
    return tuple(prompts)


def _load_sample_manifest(
    path: Path,
    *,
    expected_count: int,
) -> tuple[str, ...]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"sample manifest row {line_number} is not an object")
            episode_id = row.get("episode_id")
            if not isinstance(episode_id, int) or episode_id <= 0:
                raise ValueError(
                    f"sample manifest row {line_number} has invalid episode_id"
                )
            expected_name = f"episode{episode_id}"
            if row.get("episode_name", expected_name) != expected_name:
                raise ValueError(
                    f"sample manifest row {line_number} episode name mismatch"
                )
            rows.append(expected_name)
    if len(rows) != expected_count:
        raise ValueError(
            f"sample manifest expected {expected_count} rows, found {len(rows)}"
        )
    if len(set(rows)) != len(rows):
        raise ValueError("sample manifest episode IDs must be unique")
    return tuple(rows)


def _validate_existing_video(path: Path) -> dict[str, Any]:
    import cv2

    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise ValueError(f"cannot decode existing video: {path}")
    declared = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    decoded = 0
    black_frames = 0
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        decoded += 1
        if float(frame.mean()) < 1.0 or float(frame.std()) < 0.5:
            black_frames += 1
    capture.release()
    actual = (declared, decoded, width, height, black_frames)
    expected = (121, 121, 640, 480, 0)
    if actual != expected:
        raise ValueError(f"existing video failed frozen validation: {path}: {actual}")
    return {
        "name": path.name,
        "sha256": _sha256_file(path),
        "declared_frames": declared,
        "decoded_frames": decoded,
        "width": width,
        "height": height,
        "black_frames": black_frames,
    }


def _install_identity_refiner() -> None:
    """Make the released main execute Stage-1 without loading/running SeedVR2."""
    import numpy as np

    package = ModuleType("refiner")
    package.__path__ = []
    runtime = ModuleType("refiner.runtime")
    runtime.load_runner = lambda **_kwargs: object()
    runtime.refine_clip = lambda _runner, frames, **_kwargs: np.asarray(frames)
    blend = ModuleType("refiner.temporal_blend")
    blend.blend_arrays = lambda _refined, original, alpha: np.asarray(original)
    sys.modules["refiner"] = package
    sys.modules["refiner.runtime"] = runtime
    sys.modules["refiner.temporal_blend"] = blend


def _install_lora_hotload(
    world_model_inference: ModuleType,
    checkpoint: Path,
    *,
    rank: int,
    alpha: float,
    scale: float,
    attention_scope: str,
    source_layout: str,
) -> dict[str, Any]:
    """Hotload a PEFT q/v adapter after official VRAM wrapping."""
    if rank != 8 or alpha != 8.0:
        raise ValueError("v16 inference requires rank/alpha 8/8")
    runtime: dict[str, Any] = {"loaded_pairs": 0}
    original = world_model_inference.build_pipeline

    def build_pipeline_with_lora(device, full_path, local_model_path=None):
        from diffsynth import load_state_dict

        pipe, flow_stream = original(device, full_path, local_model_path)
        state = load_state_dict(
            str(checkpoint),
            torch_dtype=pipe.torch_dtype,
            device=pipe.device,
        )
        state, filter_report = filter_v16_lora_state_for_inference(
            state,
            attention_scope=attention_scope,
            rank=rank,
            expected_source_pairs=60 if source_layout == "self-qv" else 120,
            require_self_only_source=source_layout == "self-qv",
        )
        pair_count = filter_report["loaded_pairs"]
        pipe.load_lora(
            pipe.dit,
            state_dict=state,
            alpha=scale,
            hotload=True,
        )
        loaded_pairs = sum(
            len(getattr(module, "lora_A_weights", ())) for module in pipe.dit.modules()
        )
        if loaded_pairs != pair_count:
            raise RuntimeError(
                f"LoRA hotload coverage mismatch: {loaded_pairs}/{pair_count}"
            )
        runtime.update(filter_report)
        runtime["loaded_pairs"] = loaded_pairs
        return pipe, flow_stream

    world_model_inference.build_pipeline = build_pipeline_with_lora
    return runtime


def main() -> None:
    args = _parse_args()
    if args.sample_manifest is not None and args.max_episodes is not None:
        raise ValueError("sample manifest and max episodes are mutually exclusive")
    contract = official_stage1_contract()
    contract["seed"] = args.seed
    contract["flow_max_magnitude"] = args.flow_max_magnitude
    root = args.artifact_root.resolve(strict=True)
    for path in (
        args.test_dataset_dir,
        args.output_dir,
        args.checkpoint,
        args.local_model_path,
        args.robot_only_dir or args.embodiment_dir,
    ):
        _under(path, root)
    if args.sample_manifest is not None:
        _under(args.sample_manifest, root)
    if args.lora_checkpoint is not None:
        _under(args.lora_checkpoint, root)
    payload = {
        "contract": "flowwam-worldarena-official-stage1-only/1",
        "stage1": contract,
        "inference_sweep": {
            "is_official_default": args.seed == 1 and args.flow_max_magnitude == 20.0,
            "overrides": {
                "flow_max_magnitude": args.flow_max_magnitude,
                "seed": args.seed,
            },
        },
        "stage2_refiner": "identity-disabled",
        "starts_gpu": not args.dry_run,
        "max_episodes": args.max_episodes,
        "sample_manifest": (
            str(args.sample_manifest) if args.sample_manifest is not None else None
        ),
        "instruction_variant": args.instruction_variant,
        "physical_gpu": args.physical_gpu,
        "robot_conditioning": (
            {
                "mode": "pre-generated",
                "robot_only_dir": str(args.robot_only_dir),
            }
            if args.robot_only_dir is not None
            else {
                "mode": "sapien",
                "embodiment_dir": str(args.embodiment_dir),
            }
        ),
    }
    if args.lora_checkpoint is not None:
        targets = (
            ["self_attn.q", "self_attn.v"]
            if args.lora_attention_scope == "self"
            else ["q", "v"]
        )
        payload["lora_adapter"] = {
            "checkpoint": str(args.lora_checkpoint),
            "rank": args.lora_rank,
            "alpha": args.lora_alpha,
            "scale": args.lora_scale,
            "attention_scope": args.lora_attention_scope,
            "target_modules": targets,
            "hotload": True,
        }
        if args.lora_source_layout != "all-qv":
            payload["lora_adapter"]["source_layout"] = args.lora_source_layout
    if args.dry_run:
        if args.sample_manifest is not None:
            payload["sample_manifest_sha256"] = _sha256_file(args.sample_manifest)
        print(json.dumps(payload, sort_keys=True))
        return

    if root != FORMAL_ROOT:
        raise ValueError("formal Stage-1 artifact root mismatch")
    episode_names = (
        _load_sample_manifest(args.sample_manifest, expected_count=args.expected_count)
        if args.sample_manifest is not None
        else None
    )
    _validate_instruction_files(
        args.test_dataset_dir,
        instruction_variant=args.instruction_variant,
        expected_count=args.expected_count,
        episode_names=episode_names,
    )
    if os.environ.get("CUDA_VISIBLE_DEVICES") != str(args.physical_gpu):
        raise ValueError(
            f"official Stage-1 must be isolated to physical GPU{args.physical_gpu}"
        )
    source = args.official_source.resolve(strict=True)
    if source != OFFICIAL_SOURCE:
        raise ValueError("official FlowWAM source path mismatch")
    commit = subprocess.run(
        ["git", "-C", str(source), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if commit != OFFICIAL_FLOWWAM_COMMIT:
        raise ValueError("official FlowWAM source commit mismatch")
    checkpoint_sha = validate_checkpoint_file(args.checkpoint)

    suffix = (
        "FlowWAMOfficialStage1_test"
        if args.instruction_variant == 0
        else f"FlowWAMOfficialStage1_test_{args.instruction_variant}"
    )
    video_dir = args.output_dir / suffix
    requested_names = episode_names
    if requested_names is None:
        subdir = "first_frame/fixed_scene_task"
        frames = sorted((args.test_dataset_dir / subdir).glob("episode*.png"))
        requested_names = tuple(path.stem for path in frames[: args.expected_count])
    missing_names = list(requested_names)
    if args.resume_valid_existing:
        missing_names = []
        for name in requested_names:
            path = video_dir / f"{name}.mp4"
            if path.exists():
                _validate_existing_video(path)
            else:
                missing_names.append(name)

    _install_identity_refiner()
    inference = source / "inference"
    sys.path.insert(0, str(source))
    sys.path.insert(0, str(inference))
    import world_model_inference

    lora_runtime = None
    if args.lora_checkpoint is not None:
        lora_runtime = _install_lora_hotload(
            world_model_inference,
            args.lora_checkpoint,
            rank=args.lora_rank,
            alpha=args.lora_alpha,
            scale=args.lora_scale,
            attention_scope=args.lora_attention_scope,
            source_layout=args.lora_source_layout,
        )

    sys.argv = [
        str(inference / "world_model_inference.py"),
        "--test_dataset_dir",
        str(args.test_dataset_dir),
        "--output_dir",
        str(args.output_dir),
        "--model_name",
        "FlowWAMOfficialStage1",
        "--full_path",
        str(args.checkpoint),
        "--local_model_path",
        str(args.local_model_path),
        "--instruction_variant",
        str(args.instruction_variant),
        "--num_output_frames",
        str(contract["num_output_frames"]),
        "--size",
        str(contract["width"]),
        str(contract["height"]),
        "--fps",
        "24",
        "--camera",
        "head_camera",
        "--flow_method",
        "raft",
        "--flow_device",
        "cuda",
        "--flow_max_magnitude",
        str(contract["flow_max_magnitude"]),
        "--flow_resolution",
        str(contract["flow_width"]),
        str(contract["flow_height"]),
        "--max_stride",
        str(contract["max_stride"]),
        "--max_rollouts",
        str(contract["max_rollouts"]),
        "--num_inference_steps",
        str(contract["num_inference_steps"]),
        "--sigma_shift",
        str(contract["sigma_shift"]),
        "--seed",
        str(contract["seed"]),
        "--num_workers",
        "0",
    ]
    if args.robot_only_dir is not None:
        sys.argv.extend(["--robot_only_dir", str(args.robot_only_dir)])
    else:
        sys.argv.extend(["--embodiment_dir", str(args.embodiment_dir)])
    if args.max_episodes is not None:
        sys.argv.extend(["--max_episodes", str(args.max_episodes)])
    if missing_names:
        sys.argv.extend(["--episodes", *missing_names])
        world_model_inference.main()

    videos = [video_dir / f"{name}.mp4" for name in requested_names]
    unexpected = (
        sorted(
            path.name
            for path in video_dir.glob("*.mp4")
            if path.stem not in set(requested_names)
        )
        if video_dir.is_dir()
        else []
    )
    if unexpected:
        raise RuntimeError(
            f"official Stage-1 output contains unexpected videos: {unexpected[:5]}"
        )
    if any(not video.is_file() for video in videos):
        raise RuntimeError(
            f"official Stage-1 expected {args.expected_count} named outputs"
        )
    validated_videos = [_validate_existing_video(video) for video in videos]
    payload.update(
        {
            "official_commit": commit,
            "checkpoint_sha256": checkpoint_sha,
            "sample_manifest_sha256": (
                _sha256_file(args.sample_manifest)
                if args.sample_manifest is not None
                else None
            ),
            "requested_episode_names": list(requested_names),
            "resumed_valid_count": args.expected_count - len(missing_names),
            "generated_this_run_count": len(missing_names),
            "videos": validated_videos,
        }
    )
    if args.lora_checkpoint is not None:
        payload["lora_adapter"].update(
            {
                "checkpoint_sha256": _sha256_file(args.lora_checkpoint),
                "loaded_pairs": lora_runtime["loaded_pairs"],
                "source_pairs": lora_runtime["source_pairs"],
                "filtered_cross_attention_pairs": lora_runtime[
                    "filtered_cross_attention_pairs"
                ],
            }
        )
    receipt = args.output_dir / "stage1-only.receipt.json"
    partial = receipt.with_suffix(".partial")
    partial.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(partial, receipt)
    print(json.dumps({"receipt": str(receipt), "videos": len(videos)}, sort_keys=True))


if __name__ == "__main__":
    main()
