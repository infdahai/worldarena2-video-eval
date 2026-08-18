from __future__ import annotations

import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_v7_launcher_excludes_gpu7_and_cannot_skip_audit25() -> None:
    result = subprocess.run(
        ["bash", str(ROOT / "scripts/run_wan_se3_probe_v7.sh"), "dry-run"],
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6" in result.stdout
    assert "audit25" in result.stdout
    assert "train50 requires audit25 pass" in result.stdout
    assert "GPU7" not in result.stdout


def test_v7_trainer_has_no_hot_path_encoder_or_stage_b_symbols() -> None:
    source = (ROOT / "scripts/train_wan_se3_probe_v7_fsdp.py").read_text()
    assert "WanActionCachedDataset" in source
    assert "T5EncoderModel" not in source
    assert "WanVAE" not in source
    assert "LoRA" not in source
    assert "gripper_bias" not in source


def test_v7_entrypoint_help_has_bounded_modes() -> None:
    result = subprocess.run(
        [sys.executable, str(ROOT / "scripts/train_wan_se3_probe_v7_fsdp.py"), "--help"],
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "preflight" in result.stdout
    assert "smoke" in result.stdout
    assert "train" in result.stdout
    assert "audit" in result.stdout


def test_v7_stage_a_does_not_accept_the_frozen_official_test_as_input() -> None:
    """Official test remains unavailable until final checkpoint selection."""
    launcher = (ROOT / "scripts/run_wan_se3_probe_v7.sh").read_text()
    trainer = (ROOT / "scripts/train_wan_se3_probe_v7_fsdp.py").read_text()

    assert "--official-test-manifest" not in launcher
    assert "--official-test-manifest" not in trainer
    assert '"official_test_manifest") != {"status": "unavailable"}' in trainer
