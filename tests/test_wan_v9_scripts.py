from __future__ import annotations

import subprocess
from pathlib import Path

import pytest


ROOT = Path(__file__).parents[1]


def test_v9_production_entrypoints_exist() -> None:
    for relative in (
        "scripts/train_wan_v9_phase_locked.py",
        "scripts/run_wan_v9_phase_locked_gpu6.sh",
        "scripts/validate_wan_v9_sync_closure.py",
        "src/worldarena_baseline/wan_v9_sync_closure.py",
    ):
        assert (ROOT / relative).is_file(), relative


def test_gpu6_launcher_dry_run_has_bounded_phase_order_and_no_process() -> None:
    completed = subprocess.run(
        ["bash", "scripts/run_wan_v9_phase_locked_gpu6.sh", "dry-run"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    lines = completed.stdout.splitlines()
    assert lines == [
        "CUDA_VISIBLE_DEVICES=6",
        "preflight -> smoke3 -> train25 -> audit25 -> train100 -> audit100",
        "conditional train250 -> audit250 -> conditional phaseT500 -> audit500",
        "81 frames / 480x640 / cached latent+text / no T5 / no VAE",
    ]


def test_trainer_checks_lineage_before_cuda_and_has_no_t5_vae_hot_path() -> None:
    source = (ROOT / "scripts/train_wan_v9_phase_locked.py").read_text(encoding="utf-8")
    validate_index = source.index("_validate_inputs(args)")
    cuda_index = source.index("torch.cuda.set_device(0)")
    assert validate_index < cuda_index
    assert "T5" not in source and "load_vae" not in source
    assert "CUDA_VISIBLE_DEVICES" in source
    assert "transition_arm_present" in source
    assert "sequential_pairwise_backward" in source


def test_launcher_rechecks_gpu_ownership_before_every_gpu_phase() -> None:
    source = (ROOT / "scripts/run_wan_v9_phase_locked_gpu6.sh").read_text(encoding="utf-8")
    assert "check_gpu6" in source
    assert source.count("check_gpu6") >= 2
    assert "CUDA_VISIBLE_DEVICES=6" in source
    assert "nvidia-smi -i 6" in source
    assert '"$0" train25' in source and '"$0" audit500' in source
    assert 'require_decision "$RUN/audit-100.json" continue' in source
    assert 'require_decision "$RUN/audit-250.json" pass' in source


def test_phase_t_is_real_and_step250_audit_is_not_placeholder() -> None:
    source = (ROOT / "scripts/train_wan_v9_phase_locked.py").read_text(encoding="utf-8")
    assert "def _trajectory_terms(" in source
    assert "def _calibrate_trajectory(" in source
    assert "trajectory=trajectory" in source
    assert 'metrics.update(position_improvement=0.0' not in source
    assert "_audit_comparison(" in source


def test_replay_is_deterministic_balanced_and_leak_free() -> None:
    import importlib.util

    path = ROOT / "scripts/build_wan_v9_replay.py"
    spec = importlib.util.spec_from_file_location("build_wan_v9_replay", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    source = [
        {"optimizer_step": step, "sample": f"sample-{step:03d}", "sample_role": "mixed"}
        for step in range(1, 251)
    ]
    audit = [{"sample": f"audit-{index:02d}"} for index in range(20)]
    first = module.build(source, audit)
    second = module.build(source, audit)
    assert first == second
    assert [row["negative_family"] for row in first[:10]] == [
        "shift+1", "reverse", "shift-1", "swap", "shift+1",
        "shift-1", "shift+1", "reverse", "shift-1", "swap",
    ]
    assert not ({str(row["sample"]) for row in first} & {str(row["sample"]) for row in audit})


def test_replay_rejects_audit_leak() -> None:
    import importlib.util

    path = ROOT / "scripts/build_wan_v9_replay.py"
    spec = importlib.util.spec_from_file_location("build_wan_v9_replay_leak", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    source = [
        {"optimizer_step": step, "sample": f"sample-{step:03d}"}
        for step in range(1, 251)
    ]
    audit = [{"sample": "sample-001"}] + [
        {"sample": f"audit-{index:02d}"} for index in range(1, 20)
    ]
    with pytest.raises(ValueError, match="leaks audit20"):
        module.build(source, audit)


def test_source_closure_discovers_phase_t_and_v9_transitive_dependencies() -> None:
    from worldarena_baseline.wan_v9_sync_closure import discover_v9_source_files

    discovered = set(discover_v9_source_files(ROOT, set()))
    for relative in (
        "src/worldarena_baseline/wan_gripper_trajectory_loss.py",
        "src/worldarena_baseline/wan_gripper_probe.py",
        "src/worldarena_baseline/wan_v9_attention.py",
        "src/worldarena_baseline/wan_v9_model.py",
        "src/worldarena_baseline/wan_v9_objective.py",
        "src/worldarena_baseline/wan_v9_training.py",
        "scripts/train_wan_se3_probe_v7_fsdp.py",
    ):
        assert relative in discovered
