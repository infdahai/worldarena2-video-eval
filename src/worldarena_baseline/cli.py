from __future__ import annotations

import argparse
import json
from pathlib import Path

from .manifest import discover_episodes
from .pipeline import prepare_control_video, select_smoke_episode_ids
from .skeleton import AlohaSkeletonRenderer
from .submission import (
    build_submission_archive,
    render_model_readme,
    validate_submission_archive,
)
from .video import probe_video


def _episode_filter(value: str | None) -> set[int] | None:
    if not value:
        return None
    return {int(item) for item in value.split(",") if item.strip()}


def command_inspect(args: argparse.Namespace) -> int:
    episodes = discover_episodes(args.dataset_root)
    payload = {
        "episode_count": len(episodes),
        "smoke_episode_ids": select_smoke_episode_ids(episodes, count=3),
        "min_length": min(item.trajectory_length for item in episodes),
        "max_length": max(item.trajectory_length for item in episodes),
    }
    print(json.dumps(payload, sort_keys=True))
    return 0


def command_prepare(args: argparse.Namespace) -> int:
    episodes = discover_episodes(args.dataset_root)
    selected = _episode_filter(args.episode_ids)
    if selected is not None:
        episodes = [item for item in episodes if item.episode_id in selected]
    renderer = AlohaSkeletonRenderer(
        args.urdf, width=args.width, height=args.height, fovy_degrees=args.fovy
    )
    for position, episode in enumerate(episodes, 1):
        output = prepare_control_video(
            episode,
            renderer,
            args.controls_dir,
            num_frames=args.num_frames,
            fps=args.fps,
        )
        print(f"[{position}/{len(episodes)}] episode{episode.episode_id} -> {output}", flush=True)
    return 0


def command_write_readme(args: argparse.Namespace) -> int:
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(render_model_readme(), encoding="utf-8")
    print(output)
    return 0


def command_validate_videos(args: argparse.Namespace) -> int:
    videos = Path(args.videos_dir)
    expected = _episode_filter(args.episode_ids)
    names = sorted(videos.glob("episode_*.mp4"))
    if expected is None:
        expected = set(range(1, args.expected_count + 1))
    found = {int(path.stem.split("_")[-1]) for path in names}
    if found != expected:
        raise ValueError(f"video episode set mismatch: expected {len(expected)}, found {len(found)}")
    for path in names:
        probe = probe_video(path)
        if probe.frame_count != args.num_frames or (probe.width, probe.height) != (
            args.width,
            args.height,
        ):
            raise ValueError(f"invalid video {path}: {probe}")
    print(json.dumps({"video_count": len(names), "valid": True}))
    return 0


def command_package(args: argparse.Namespace) -> int:
    build_submission_archive(args.staging_dir, args.archive)
    report = validate_submission_archive(args.archive, args.expected_count)
    print(json.dumps({"video_count": report.video_count, "archive": str(args.archive)}))
    return 0


def command_validate_archive(args: argparse.Namespace) -> int:
    report = validate_submission_archive(args.archive, args.expected_count)
    print(json.dumps({"video_count": report.video_count, "metadata": report.metadata}))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="WorldArena Track 1 OSCAR baseline")
    sub = parser.add_subparsers(required=True)

    inspect_parser = sub.add_parser("inspect")
    inspect_parser.add_argument("--dataset-root", type=Path, required=True)
    inspect_parser.set_defaults(func=command_inspect)

    prepare = sub.add_parser("prepare")
    prepare.add_argument("--dataset-root", type=Path, required=True)
    prepare.add_argument("--urdf", type=Path, required=True)
    prepare.add_argument("--controls-dir", type=Path, required=True)
    prepare.add_argument("--episode-ids")
    prepare.add_argument("--num-frames", type=int, default=81)
    prepare.add_argument("--fps", type=float, default=24.0)
    prepare.add_argument("--width", type=int, default=640)
    prepare.add_argument("--height", type=int, default=480)
    prepare.add_argument("--fovy", type=float, default=37.0)
    prepare.set_defaults(func=command_prepare)

    readme = sub.add_parser("write-readme")
    readme.add_argument("--output", type=Path, required=True)
    readme.set_defaults(func=command_write_readme)

    videos = sub.add_parser("validate-videos")
    videos.add_argument("--videos-dir", type=Path, required=True)
    videos.add_argument("--expected-count", type=int, default=1000)
    videos.add_argument("--episode-ids")
    videos.add_argument("--num-frames", type=int, default=81)
    videos.add_argument("--width", type=int, default=640)
    videos.add_argument("--height", type=int, default=480)
    videos.set_defaults(func=command_validate_videos)

    package = sub.add_parser("package")
    package.add_argument("--staging-dir", type=Path, required=True)
    package.add_argument("--archive", type=Path, required=True)
    package.add_argument("--expected-count", type=int, default=1000)
    package.set_defaults(func=command_package)

    validate = sub.add_parser("validate-archive")
    validate.add_argument("--archive", type=Path, required=True)
    validate.add_argument("--expected-count", type=int, default=1000)
    validate.set_defaults(func=command_validate_archive)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
