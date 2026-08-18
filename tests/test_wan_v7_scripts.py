from __future__ import annotations

import subprocess
import sys
import importlib.util
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]


def _trainer_module():
    path = ROOT / "scripts/train_wan_se3_probe_v7_fsdp.py"
    spec = importlib.util.spec_from_file_location("wan_v7_trainer_script", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


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


def test_v7_single_gpu_launcher_uses_only_gpu6() -> None:
    result = subprocess.run(
        ["bash", str(ROOT / "scripts/run_wan_se3_probe_v7_single_gpu.sh"), "dry-run"],
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "CUDA_VISIBLE_DEVICES=6" in result.stdout
    assert "nproc_per_node=1" in result.stdout
    assert "v7-se3-single-gpu" in result.stdout
    assert "retirement-audit20" in result.stdout
    assert "zero-gate" in result.stdout
    assert "step10" in result.stdout
    assert "step25" in result.stdout
    assert "train50" not in result.stdout
    assert "CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6" not in result.stdout


def test_v7_single_gpu_launcher_passes_the_isolated_topology() -> None:
    launcher = (ROOT / "scripts/run_wan_se3_probe_v7_single_gpu.sh").read_text()
    assert "--topology single-gpu" in launcher
    assert "--nproc_per_node=1" in launcher
    assert "runs/v7-se3-single-gpu" in launcher


def test_v7_launcher_validates_committed_closure_before_torchrun() -> None:
    launcher = (ROOT / "scripts/run_wan_se3_probe_v7.sh").read_text()
    for phase in ("preflight", "smoke", "train10", "train25", "audit25", "train50", "audit50"):
        section = launcher.split(f"  {phase})", 1)[1].split("    ;;", 1)[0]
        assert section.index("validate_source_closure") < section.index("run --mode")


def test_v7_preflight_receipt_rejects_a_stale_source_closure_before_cuda() -> None:
    module = _trainer_module()
    current = {"source_manifest_sha256": "a" * 64, "source_code_sha256": "b" * 64}
    module._validate_preflight_source_hashes(
        {"source_hashes": current},
        source_manifest_sha256=current["source_manifest_sha256"],
        source_code_sha256=current["source_code_sha256"],
    )
    with pytest.raises(RuntimeError, match="source closure digest mismatch"):
        module._validate_preflight_source_hashes(
            {"source_hashes": {**current, "source_code_sha256": "c" * 64}},
            source_manifest_sha256=current["source_manifest_sha256"],
            source_code_sha256=current["source_code_sha256"],
        )


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
    assert "--audit-set" in result.stdout
    assert "--audit-zero-gate" in result.stdout


def test_v7_stage_a_does_not_accept_the_frozen_official_test_as_input() -> None:
    """Official test remains unavailable until final checkpoint selection."""
    launcher = (ROOT / "scripts/run_wan_se3_probe_v7.sh").read_text()
    trainer = (ROOT / "scripts/train_wan_se3_probe_v7_fsdp.py").read_text()

    assert "--official-test-manifest" not in launcher
    assert "--official-test-manifest" not in trainer
    assert '"official_test_manifest") != {"status": "unavailable"}' in trainer
