from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

import h5py

from .naming import output_filename


EPISODE_RE = re.compile(r"^episode(\d+)\.hdf5$")


@dataclass(frozen=True)
class EpisodeSpec:
    episode_id: int
    scene: str
    hdf5_path: Path
    instruction_path: Path
    first_frame_path: Path
    instruction: str
    trajectory_length: int

    @property
    def output_name(self) -> str:
        return output_filename(self.episode_id)


def discover_episodes(dataset_root: Path | str) -> list[EpisodeSpec]:
    """Discover and validate Track 1 episodes below an extracted dataset root."""
    root = Path(dataset_root)
    episodes: list[EpisodeSpec] = []
    for hdf5_path in root.glob("data/*/episode*.hdf5"):
        match = EPISODE_RE.match(hdf5_path.name)
        if not match:
            continue
        episode_id = int(match.group(1))
        scene = hdf5_path.parent.name
        instruction_path = root / "instructions" / scene / f"episode{episode_id}.json"
        first_frame_path = root / "first_frame" / scene / f"episode{episode_id}.png"
        if not instruction_path.is_file() or not first_frame_path.is_file():
            raise FileNotFoundError(f"episode{episode_id} is missing instruction or first frame")

        with h5py.File(hdf5_path, "r") as handle:
            if "joint_action/vector" not in handle:
                raise ValueError(f"episode{episode_id}: missing /joint_action/vector")
            shape = handle["joint_action/vector"].shape
        if len(shape) != 2 or shape[1] != 14:
            raise ValueError(f"episode{episode_id}: expected joint14, got shape {shape}")

        payload = json.loads(instruction_path.read_text(encoding="utf-8"))
        instruction = payload.get("instruction")
        if not isinstance(instruction, str) or not instruction.strip():
            raise ValueError(f"episode{episode_id}: missing instruction text")
        episodes.append(
            EpisodeSpec(
                episode_id=episode_id,
                scene=scene,
                hdf5_path=hdf5_path,
                instruction_path=instruction_path,
                first_frame_path=first_frame_path,
                instruction=instruction.strip(),
                trajectory_length=int(shape[0]),
            )
        )
    return sorted(episodes, key=lambda episode: episode.episode_id)
