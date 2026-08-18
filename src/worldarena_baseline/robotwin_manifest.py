from __future__ import annotations

import json
import os
import hashlib
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

from .robotwin_episode import sample_name


@dataclass(frozen=True)
class RobotwinTrainingEpisode:
    sample: str
    task: str
    variant: str
    episode_index: int
    hdf5: str
    instruction: str
    seen_prompts: tuple[str, ...]
    unseen_prompts: tuple[str, ...]


def _prompt_list(payload: dict, field: str, *, required: bool) -> tuple[str, ...]:
    values = payload.get(field, [])
    if not isinstance(values, list) or any(not isinstance(value, str) for value in values):
        raise ValueError(f"instruction field {field!r} must be a list of strings")
    cleaned = tuple(value.strip() for value in values if value.strip())
    if required and not cleaned:
        raise ValueError(f"instruction field {field!r} must be non-empty")
    return cleaned


def discover_training_episodes(
    root: Path | str,
    *,
    variant: str = "aloha-agilex_clean_50",
) -> list[RobotwinTrainingEpisode]:
    root = Path(root)
    episodes: list[RobotwinTrainingEpisode] = []
    for variant_root in sorted(root.glob(f"*/{variant}")):
        task = variant_root.parent.name
        for hdf5_path in sorted((variant_root / "data").glob("episode*.hdf5")):
            suffix = hdf5_path.stem.removeprefix("episode")
            if not suffix.isdigit():
                continue
            instruction_path = variant_root / "instructions" / f"{hdf5_path.stem}.json"
            if not instruction_path.is_file():
                continue
            payload = json.loads(instruction_path.read_text(encoding="utf-8"))
            if not isinstance(payload, dict):
                raise ValueError(f"instruction must be a JSON object: {instruction_path}")
            episode_index = int(suffix)
            episodes.append(
                RobotwinTrainingEpisode(
                    sample=sample_name(task, variant, episode_index),
                    task=task,
                    variant=variant,
                    episode_index=episode_index,
                    hdf5=hdf5_path.relative_to(root).as_posix(),
                    instruction=instruction_path.relative_to(root).as_posix(),
                    seen_prompts=_prompt_list(payload, "seen", required=True),
                    unseen_prompts=_prompt_list(payload, "unseen", required=False),
                )
            )
    return sorted(episodes, key=lambda item: (item.task, item.episode_index))


def write_manifest_jsonl(
    output: Path | str,
    episodes: Iterable[RobotwinTrainingEpisode],
) -> int:
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [json.dumps(asdict(episode), sort_keys=True) for episode in episodes]
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text("\n".join(rows) + ("\n" if rows else ""), encoding="utf-8")
    os.replace(temporary, path)
    return len(rows)


def split_episodes_by_task(
    episodes: Iterable[RobotwinTrainingEpisode],
    *,
    train_tasks: int,
    dev_tasks: int,
    blind_tasks: int,
    seed: int = 20260815,
) -> dict[str, list[RobotwinTrainingEpisode]]:
    if min(train_tasks, dev_tasks, blind_tasks) < 0:
        raise ValueError("task split counts must be non-negative")
    rows = sorted(episodes, key=lambda item: (item.task, item.episode_index))
    tasks = sorted({episode.task for episode in rows})
    expected = train_tasks + dev_tasks + blind_tasks
    if len(tasks) != expected:
        raise ValueError(f"expected {expected} tasks, found {len(tasks)}")
    ranked = sorted(
        tasks,
        key=lambda task: hashlib.sha256(f"{seed}:{task}".encode()).digest(),
    )
    split_tasks = {
        "train": set(ranked[:train_tasks]),
        "dev": set(ranked[train_tasks : train_tasks + dev_tasks]),
        "blind": set(ranked[train_tasks + dev_tasks :]),
    }
    return {
        split: [episode for episode in rows if episode.task in selected]
        for split, selected in split_tasks.items()
    }
