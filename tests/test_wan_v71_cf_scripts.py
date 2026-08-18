from __future__ import annotations

import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_cf_launcher_is_fresh_geometry_only_and_bounded() -> None:
    launcher = ROOT / "scripts/run_wan_se3_geometry_lora_cf_v71_single_gpu.sh"
    result = subprocess.run(
        ["bash", str(launcher), "dry-run"],
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "CUDA_VISIBLE_DEVICES=6" in result.stdout
    assert "v71-geometry-lora-cf" in result.stdout
    assert "fresh-init-from-clean-gated-step10" in result.stdout
    assert "geometry-only-counterfactual" in result.stdout
    assert "reverse -> shift+1 -> swap -> reverse -> shift-1 -> swap" in result.stdout
    assert "step25 requires 12/20 for reverse,shift,swap" in result.stdout
    assert "step50 requires 14/20 for reverse,shift,swap" in result.stdout
    assert "step100" not in result.stdout


def test_cf_launcher_never_uses_failed_v71_step25_as_parent() -> None:
    source = (
        ROOT / "scripts/run_wan_se3_geometry_lora_cf_v71_single_gpu.sh"
    ).read_text(encoding="utf-8")
    assert "runs/v6-clean-gated-parent/clean-gated-step10.pt" in source
    assert "runs/v71-se3-geometry-lora-single-gpu/mechanism/step-000025.pt" not in source
    assert "--architecture v71-geometry-lora-cf" in source
