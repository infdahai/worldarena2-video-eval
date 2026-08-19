from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")


def _module():
    path = Path(__file__).parents[1] / "scripts" / "audit_wan_v11_health.py"
    spec = importlib.util.spec_from_file_location("audit_wan_v11_health", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_soft_argmax_decodes_independent_arm_time_peaks() -> None:
    module = _module()
    logits = torch.full((1, 2, 3, 15, 20), -100.0)
    logits[0, 0, 1, 7, 19] = 100.0
    logits[0, 1, 2, 14, 0] = 100.0

    positions = module._soft_argmax(logits)

    assert positions.shape == (1, 2, 3, 2)
    assert positions[0, 0, 1].tolist() == pytest.approx([1.0, 0.5])
    assert positions[0, 1, 2].tolist() == pytest.approx([0.0, 1.0])


def test_health_audit_is_fail_closed_to_formal_root() -> None:
    module = _module()
    with pytest.raises(ValueError, match="escapes formal root"):
        module._formal(Path("/tmp/not-formal"))


def test_motion_activity_is_padded_only_at_latent_zero() -> None:
    motion = torch.ones(1, 2, 20, dtype=torch.bool)
    padded = torch.cat((torch.zeros(1, 2, 1, dtype=torch.bool), motion), dim=2)
    assert padded.shape == (1, 2, 21)
    assert torch.count_nonzero(padded[:, :, 0]).item() == 0
    assert torch.all(padded[:, :, 1:])
