from __future__ import annotations

from pathlib import Path

from worldarena_baseline.wan_v11_sync_closure import (
    LEGACY_BYTE_PINS,
    discover_v11_source_files,
)


ROOT = Path(__file__).resolve().parents[1]


def test_launcher_is_gpu6_only_and_validates_source_before_gpu_python() -> None:
    source = (ROOT / "scripts/run_wan_v11_gpu6.sh").read_text()

    assert "ROOT=/data/di/worldarena2_track1_20260815" in source
    assert "RUN=$ROOT/runs/v11-worldarena-balanced-bimanual-world-controller" in source
    assert "export CUDA_VISIBLE_DEVICES=6" in source
    assert "--query-compute-apps=gpu_uuid" in source
    assert "memory >= 1024 || util >= 10" in source
    assert source.index("validate_wan_v11_sync_closure.py") < source.index("train_wan_v11_bimanual.py")
    assert "kill " not in source
    assert "pkill" not in source


def test_trainer_has_no_t5_vae_or_post_gate_modules() -> None:
    source = (ROOT / "scripts/train_wan_v11_bimanual.py").read_text().lower()

    for forbidden in ("t5encoder", "wanvae", "v-jepa", "depth_branch", "coordination_branch"):
        assert forbidden not in source
    assert "torch.cuda.set_device" in source
    assert source.index("_validate_inputs(args)") < source.index("torch.cuda.set_device")


def test_recursive_source_closure_contains_runtime_dependencies() -> None:
    files = set(discover_v11_source_files(ROOT))

    for required in (
        "scripts/run_wan_v11_gpu6.sh",
        "scripts/train_wan_v11_bimanual.py",
        "scripts/train_wan_v10_relational.py",
        "scripts/train_wan_se3_probe_v7_fsdp.py",
        "src/worldarena_baseline/wan_v11_model.py",
        "src/worldarena_baseline/wan_v11_objective.py",
        "src/worldarena_baseline/wan_cached_dataset.py",
    ):
        assert required in files


def test_only_reviewed_legacy_skeleton_bytes_are_pinned() -> None:
    assert LEGACY_BYTE_PINS == {
        "src/worldarena_baseline/skeleton.py": (
            "93cb2aa340a725e7b0a7acd37450334bf46c6483c7baf3646bd14ca3e09a8143"
        )
    }


def test_dry_run_contract_names_all_bounded_phases() -> None:
    source = (ROOT / "scripts/run_wan_v11_gpu6.sh").read_text()
    for phase in (
        "replay",
        "preflight",
        "smoke",
        "train100",
        "audit100",
        "train500",
        "audit500",
        "train2060",
        "audit2060",
    ):
        assert phase in source
