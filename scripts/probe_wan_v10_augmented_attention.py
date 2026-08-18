#!/usr/bin/env python3
"""Fail-closed production Flash-Attention probe for Wan v10 head width 144."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import tempfile
import time

import torch

from worldarena_baseline.wan_ti2v_import import install_wan_ti2v_package


FORMAL_ROOT = Path("/data/di/worldarena2_track1_20260815")
PRODUCTION_SEQUENCE = 25200
PRODUCTION_HEADS = 24
NATIVE_HEAD_DIM = 128
RELATION_RANK = 16


def _guard_output(path: Path) -> Path:
    lexical = path.expanduser().absolute()
    try:
        lexical.relative_to(FORMAL_ROOT)
    except ValueError as exc:
        raise ValueError("v10 feasibility output must be below the formal root") from exc
    parent = lexical.parent
    parent.mkdir(parents=True, exist_ok=True)
    if lexical.is_symlink() or parent.resolve() != parent:
        raise ValueError("v10 feasibility output may not traverse a symlink")
    return lexical


def _atomic_json(path: Path, payload: dict[str, object]) -> None:
    data = (json.dumps(payload, sort_keys=True, indent=2) + "\n").encode()
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=f".{path.name}.", delete=False) as handle:
        temporary = Path(handle.name)
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())
    try:
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _run_attention(attention, *, sequence: int, head_dim: int) -> tuple[torch.Tensor, float]:
    shape = (1, sequence, PRODUCTION_HEADS, head_dim)
    q = torch.randn(shape, device="cuda", dtype=torch.bfloat16, requires_grad=False)
    k = torch.randn_like(q)
    v = torch.randn_like(q)
    torch.cuda.synchronize()
    started = time.monotonic()
    output = attention(
        q=q,
        k=k,
        v=v,
        k_lens=torch.tensor([sequence], device="cuda", dtype=torch.long),
        window_size=(-1, -1),
        softmax_scale=1.0 / math.sqrt(128),
    )
    torch.cuda.synchronize()
    elapsed = time.monotonic() - started
    if output.shape != shape or not torch.isfinite(output).all():
        raise RuntimeError("production fused attention returned an invalid output")
    return output, elapsed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--wan-source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = _guard_output(args.output)
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("v10 feasibility requires exactly one visible CUDA device")
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "6":
        raise RuntimeError("v10 feasibility is authorized only on physical GPU6")

    install_wan_ti2v_package(args.wan_source)
    from wan.modules import attention as wan_attention_module

    flash2 = bool(wan_attention_module.FLASH_ATTN_2_AVAILABLE)
    flash3 = bool(wan_attention_module.FLASH_ATTN_3_AVAILABLE)
    if not (flash2 or flash3):
        raise RuntimeError("real Flash Attention is unavailable; quadratic fallback is forbidden")

    torch.cuda.set_device(0)
    torch.cuda.empty_cache()
    _run_attention(wan_attention_module.attention, sequence=32, head_dim=NATIVE_HEAD_DIM)
    torch.cuda.reset_peak_memory_stats()
    _, native_seconds = _run_attention(
        wan_attention_module.attention,
        sequence=PRODUCTION_SEQUENCE,
        head_dim=NATIVE_HEAD_DIM,
    )
    _, augmented_seconds = _run_attention(
        wan_attention_module.attention,
        sequence=PRODUCTION_SEQUENCE,
        head_dim=NATIVE_HEAD_DIM + RELATION_RANK,
    )
    payload: dict[str, object] = {
        "contract": "wan-v10-attention-feasibility/1",
        "passed": True,
        "quadratic_fallback": False,
        "flash_attention_2": flash2,
        "flash_attention_3": flash3,
        "sequence_length": PRODUCTION_SEQUENCE,
        "num_heads": PRODUCTION_HEADS,
        "native_head_dim": NATIVE_HEAD_DIM,
        "augmented_head_dim": NATIVE_HEAD_DIM + RELATION_RANK,
        "softmax_scale": 1.0 / math.sqrt(NATIVE_HEAD_DIM),
        "native_seconds": native_seconds,
        "augmented_seconds": augmented_seconds,
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
        "gpu_name": torch.cuda.get_device_name(0),
        "cuda_visible_devices": os.environ["CUDA_VISIBLE_DEVICES"],
        "torch_version": torch.__version__,
        "python": platform.python_version(),
        "wan_attention_sha256": hashlib.sha256(
            Path(wan_attention_module.__file__).read_bytes()
        ).hexdigest(),
    }
    _atomic_json(output, payload)
    print(json.dumps(payload, sort_keys=True))


if __name__ == "__main__":
    main()

