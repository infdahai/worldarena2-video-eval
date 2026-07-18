from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]


def test_shell_scripts_have_valid_syntax() -> None:
    scripts = [ROOT / "scripts" / "bootstrap.sh", ROOT / "scripts" / "campaign.sh"]
    for script in scripts:
        subprocess.run(["bash", "-n", str(script)], check=True)


def test_campaign_uses_checkpoint_root_and_waits_for_hf_auth() -> None:
    text = (ROOT / "scripts" / "campaign.sh").read_text(encoding="utf-8")
    assert 'CHECKPOINT="${CHECKPOINT:-$ROOT/checkpoints}"' in text
    assert 'COSMOS_REASON_PATH="${COSMOS_REASON_PATH:-$ROOT/cosmos-reason1-7b}"' in text
    assert "WAITING_FOR_HF_AUTH" in text
    assert '"$HF" auth whoami' in text


def test_campaign_supports_selected_gpus_and_local_only_packaging() -> None:
    text = (ROOT / "scripts" / "campaign.sh").read_text(encoding="utf-8")
    assert 'GPU_INDICES="${GPU_INDICES:-0,1,2,3,4,5,6,7}"' in text
    assert 'MIN_FREE_MIB="${MIN_FREE_MIB:-22000}"' in text
    assert 'MAX_UTIL_PERCENT="${MAX_UTIL_PERCENT:-10}"' in text
    assert 'SMOKE_COUNT="${SMOKE_COUNT:-3}"' in text
    assert 'GATE_COUNT="${GATE_COUNT:-20}"' in text
    assert 'for GPU in "${GPU_LIST[@]}"; do' in text
    assert '--worker-index "$WORKER_INDEX" --worker-count "$WORKER_COUNT"' in text
    assert '--smoke-count "$SMOKE_COUNT" --gate-count "$GATE_COUNT"' in text
    assert 'NO_PUBLISH="${NO_PUBLISH:-0}"' in text
    assert 'if [ "$NO_PUBLISH" = "1" ]; then' in text


@pytest.mark.parametrize(
    ("gpu_indices", "expected"),
    [
        (
            "0,1,2,3,4,5,6,7",
            [(0, 0), (1, 1), (2, 2), (3, 3), (4, 4), (5, 5), (6, 6), (7, 7)],
        ),
        ("0", [(0, 0)]),
        ("0,2", [(0, 0), (1, 2)]),
    ],
)
def test_campaign_gpu_indices_are_parsed_into_worker_assignments(
    gpu_indices: str, expected: list[tuple[int, int]]
) -> None:
    helper = ROOT / "scripts" / "gpu_indices.sh"
    command = (
        'source "$1"; parse_gpu_indices "$2"; '
        'for worker_index in "${!GPU_LIST[@]}"; do '
        'printf "%s:%s\\n" "$worker_index" "${GPU_LIST[$worker_index]}"; '
        "done"
    )

    result = subprocess.run(
        ["bash", "-c", command, "--", str(helper), gpu_indices],
        check=True,
        capture_output=True,
        text=True,
        env=os.environ.copy(),
    )

    assert result.stdout.splitlines() == [
        f"{worker_index}:{gpu_index}" for worker_index, gpu_index in expected
    ]


def test_campaign_runs_resumable_smoke_and_gate_before_full_generation() -> None:
    text = (ROOT / "scripts" / "campaign.sh").read_text(encoding="utf-8")
    assert 'SMOKE_DIR="${SMOKE_DIR:-$ROOT/smoke/videos}"' in text
    assert 'GATE_DIR="${GATE_DIR:-$ROOT/gate/videos}"' in text
    assert 'run_workers "$SMOKE_IDS" "$SMOKE_DIR" "smoke"' in text
    assert 'run_workers "$GATE_IDS" "$GATE_DIR" "gate"' in text
    assert 'validate-videos --videos-dir "$SMOKE_DIR" --episode-ids "$SMOKE_IDS"' in text
    assert 'validate-videos --videos-dir "$GATE_DIR" --episode-ids "$GATE_IDS"' in text
    assert 'copy_validated_videos "$SMOKE_DIR" "$SMOKE_IDS"' in text
    assert 'copy_validated_videos "$GATE_DIR" "$GATE_IDS"' in text
    assert text.index('run_workers "$SMOKE_IDS"') < text.index('run_workers "$GATE_IDS"')
    assert text.index('run_workers "$GATE_IDS"') < text.index('full generation')


def test_bootstrap_downloads_official_inputs_and_prepares_controls() -> None:
    text = (ROOT / "scripts" / "bootstrap.sh").read_text(encoding="utf-8")
    assert "WorldArena/WorldArena2.0" in text
    assert "TianxingChen/RoboTwin2.0" in text
    assert "zywu2115/OSCAR-2B" in text
    assert "nvidia/Cosmos-Reason1-7B" in text
    assert "worldarena_baseline.cli prepare" in text


def test_bootstrap_uses_local_transformer_engine_compatibility_layer() -> None:
    text = (ROOT / "scripts" / "bootstrap.sh").read_text(encoding="utf-8")
    assert "transformer_engine[pytorch]" not in text
    assert "compat/transformer_engine" in text
    assert (ROOT / "compat" / "transformer_engine" / "pytorch" / "attention" / "rope.py").is_file()


def test_bootstrap_uses_local_megatron_compatibility_layer() -> None:
    text = (ROOT / "scripts" / "bootstrap.sh").read_text(encoding="utf-8")
    assert "compat/megatron" in text
    assert (ROOT / "compat" / "megatron" / "core" / "parallel_state.py").is_file()
    assert "requirements_minimal.txt" in text
    assert "grep -vE" in text
    assert "megatron-core" in text


def test_bootstrap_uses_pid_guarded_background_launcher_instead_of_tmux() -> None:
    text = (ROOT / "scripts" / "bootstrap.sh").read_text(encoding="utf-8")
    assert "tmux" not in text
    assert 'CAMPAIGN_PID_FILE="${CAMPAIGN_PID_FILE:-$ROOT/campaign.pid}"' in text
    assert 'CAMPAIGN_LAUNCH_LOG="${CAMPAIGN_LAUNCH_LOG:-$ROOT/campaign.launch.log}"' in text
    assert 'kill -0 "$CAMPAIGN_PID"' in text
    assert "nohup bash" in text
