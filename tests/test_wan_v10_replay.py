from __future__ import annotations

from worldarena_baseline.wan_v10_replay import build_v10_replay, replay_sha256


def _roles():
    return {
        **{f"s-{index}": "single_dominant" for index in range(8)},
        **{f"b-{index}": "bimanual_heavy" for index in range(7)},
        **{f"m-{index}": "mixed" for index in range(4)},
        "q-0": "quiet",
    }


def test_replay_is_deterministic_and_alternates_only_reverse_swap() -> None:
    first = build_v10_replay(_roles(), steps=8, world_size=1, seed=19)
    second = build_v10_replay(dict(reversed(list(_roles().items()))), steps=8, world_size=1, seed=19)
    assert first == second
    assert [row["negative_family"] for row in first] == ["reverse", "swap"] * 4
    assert all(row["action_dropout"] is False for row in first)
    assert replay_sha256(first) == replay_sha256(second)

