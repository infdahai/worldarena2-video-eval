from __future__ import annotations

import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_shell_scripts_have_valid_syntax() -> None:
    scripts = [ROOT / "scripts" / "bootstrap.sh", ROOT / "scripts" / "campaign.sh"]
    for script in scripts:
        subprocess.run(["bash", "-n", str(script)], check=True)


def test_campaign_uses_checkpoint_root_and_waits_for_hf_auth() -> None:
    text = (ROOT / "scripts" / "campaign.sh").read_text(encoding="utf-8")
    assert 'CHECKPOINT="$ROOT/checkpoints"' in text
    assert 'COSMOS_REASON_PATH="${COSMOS_REASON_PATH:-$ROOT/cosmos-reason1-7b}"' in text
    assert "WAITING_FOR_HF_AUTH" in text
    assert '"$HF" auth whoami' in text


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
