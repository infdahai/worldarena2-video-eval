"""Pinned contracts for the released FlowWAM WorldArena inference chain."""

from __future__ import annotations

import copy
import hashlib
import math
from collections.abc import Iterable
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from types import MethodType

import yaml

OFFICIAL_FLOWWAM_COMMIT = "f06fa46042e97738c6619c868f1097be6749d48d"
SEEDVR2_REVISION = "37255ff8cccfb01071b87f635a5948ca8d53117c"
SEEDVR2_WEIGHTS = {
    "ema_vae.pth": (
        1_002_691_902,
        "c7df8a67e68b7f9aca3d5d2153d2ce8ab4373687741a0f9ce87cb356ace51cac",
    ),
    "seedvr2_ema_3b.pth": (
        13_566_090_228,
        "6bcc5ac59447e97b100477480aebb01be2ec724c8340bb83faae21f64848604b",
    ),
}


@dataclass(frozen=True)
class RefinerPlan:
    input_videos: tuple[Path, ...]
    native_output_dir: Path
    submission_output_dir: Path
    alpha: float = 0.7
    target_area: int = 720 * 1280
    submission_size: tuple[int, int] = (640, 480)
    seed: int = 666
    sample_steps: int = 1


def official_stage1_contract() -> dict[str, object]:
    """Return the exact arguments from the released WorldArena shell runner."""
    return {
        "contract": "flowwam-worldarena-official-stage1/1",
        "num_output_frames": 121,
        "flow_max_magnitude": 20.0,
        "max_stride": 3,
        "max_rollouts": 2,
        "seed": 1,
        "sigma_shift": 5.0,
        "num_inference_steps": 50,
        "width": 640,
        "height": 480,
        "flow_width": 320,
        "flow_height": 240,
        "text_cfg": None,
    }


def _validated_timestamps(values: Iterable[float], *, name: str) -> tuple[float, ...]:
    timestamps = tuple(float(value) for value in values)
    if not timestamps:
        raise ValueError(f"{name} timestamps must not be empty")
    if not all(math.isfinite(value) for value in timestamps):
        raise ValueError(f"{name} timestamps must be finite")
    if any(right <= left for left, right in zip(timestamps, timestamps[1:])):
        raise ValueError(f"{name} timestamps must be strictly increasing")
    return timestamps


def action_timestamp_nearest_indices(
    generated_timestamps: Iterable[float],
    action_timestamps: Iterable[float],
) -> tuple[int, ...]:
    """Map each action time to the nearest generated frame, preferring earlier ties."""
    generated = _validated_timestamps(
        generated_timestamps,
        name="generated",
    )
    actions = _validated_timestamps(action_timestamps, name="action")
    result = []
    for target in actions:
        distances = tuple(abs(timestamp - target) for timestamp in generated)
        minimum = min(distances)
        result.append(
            next(
                index
                for index, distance in enumerate(distances)
                if math.isclose(distance, minimum, rel_tol=1e-12, abs_tol=1e-12)
            )
        )
    return tuple(result)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_seedvr_weights(checkpoint_dir: Path) -> dict[str, str]:
    """Require the two exact files from the pinned SeedVR2 revision."""
    root = Path(checkpoint_dir)
    result: dict[str, str] = {}
    for name, (expected_size, expected_hash) in SEEDVR2_WEIGHTS.items():
        path = root / name
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"SeedVR2 weight must be a regular file: {path}")
        if path.stat().st_size != expected_size:
            raise ValueError(f"SeedVR2 weight size mismatch: {path}")
        actual_hash = _sha256(path)
        if actual_hash != expected_hash:
            raise ValueError(f"SeedVR2 weight hash mismatch: {path}")
        result[name] = actual_hash
    return result


def validate_seedvr_4090_config(
    reference_config: Path | str,
    compatible_config: Path | str,
) -> str:
    """Require the 4090 config to differ only by its four norm backends."""
    reference = Path(reference_config)
    compatible = Path(compatible_config)
    for path in (reference, compatible):
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"SeedVR2 config must be a regular file: {path}")
    reference_payload = yaml.safe_load(reference.read_text(encoding="utf-8"))
    compatible_payload = yaml.safe_load(compatible.read_text(encoding="utf-8"))
    expected = copy.deepcopy(reference_payload)
    model = expected["dit"]["model"]
    model["vid_out_norm"] = "rms"
    model["txt_in_norm"] = "layer"
    model["norm"] = "rms"
    model["qk_norm"] = "rms"
    if compatible_payload != expected:
        raise ValueError("SeedVR2 4090 config must change only fusedrms/fusedln norms")
    return _sha256(compatible)


@contextmanager
def seedvr_config_override(config_module, compatible_config: Path | str):
    """Redirect only SeedVR2's hard-coded default 3B config load."""
    compatible = Path(compatible_config).resolve(strict=True)
    original_load_config = config_module.load_config

    def load_config(path, argv=None):
        normalized = Path(str(path)).as_posix().lstrip("./")
        if normalized == "configs_3b/main.yaml":
            return original_load_config(str(compatible), argv)
        return original_load_config(path, argv)

    config_module.load_config = load_config
    try:
        yield
    finally:
        config_module.load_config = original_load_config


@contextmanager
def seedvr_low_memory_rmsnorm_override(normalization_module, replacement_class):
    """Replace SeedVR's RMSNorm constructor only while building the DiT."""
    original_rmsnorm = normalization_module.RMSNorm
    normalization_module.RMSNorm = replacement_class
    try:
        yield
    finally:
        normalization_module.RMSNorm = original_rmsnorm


def build_seedvr_chunked_rmsnorm(torch_module, *, chunk_rows: int = 1024):
    """Build an inference-only RMSNorm that bounds the temporary FP32 tensor.

    Diffusers RMSNorm converts the complete activation to FP32. SeedVR2's first
    attention norm therefore requests hundreds of MiB at once on a 24 GiB GPU.
    This implementation preserves the same FP32 variance computation but does
    it over rows in bounded chunks and stores the result in the input dtype.
    """
    if chunk_rows <= 0:
        raise ValueError("chunk_rows must be positive")

    class ChunkedRMSNorm(torch_module.nn.Module):
        def __init__(self, dim, eps: float, elementwise_affine: bool = True):
            super().__init__()
            self.eps = eps
            self.dim = torch_module.Size((dim,) if isinstance(dim, int) else dim)
            self.weight = (
                torch_module.nn.Parameter(torch_module.ones(self.dim))
                if elementwise_affine
                else None
            )

        def forward(self, hidden_states):
            if torch_module.is_grad_enabled() and hidden_states.requires_grad:
                raise RuntimeError("chunked SeedVR RMSNorm is inference-only")
            input_dtype = hidden_states.dtype
            flat = hidden_states.reshape(-1, hidden_states.shape[-1])
            output = torch_module.empty_like(flat)
            for start in range(0, flat.shape[0], chunk_rows):
                stop = min(start + chunk_rows, flat.shape[0])
                chunk = flat[start:stop]
                variance = chunk.to(torch_module.float32).pow(2).mean(-1, keepdim=True)
                normalized = chunk * torch_module.rsqrt(variance + self.eps)
                if self.weight is not None:
                    if self.weight.dtype in (
                        torch_module.float16,
                        torch_module.bfloat16,
                    ):
                        normalized = normalized.to(self.weight.dtype)
                    normalized = normalized * self.weight
                else:
                    normalized = normalized.to(input_dtype)
                output[start:stop].copy_(normalized)
            return output.reshape_as(hidden_states)

    ChunkedRMSNorm.__name__ = "SeedVRChunkedRMSNorm"
    return ChunkedRMSNorm


def seedvr_required_rope_shape(vid_shapes, txt_lengths):
    """Return the exact frequency extents needed by a SeedVR batch."""
    rows = [tuple(int(value) for value in row) for row in vid_shapes]
    lengths = [int(value) for value in txt_lengths]
    if not rows or len(rows) != len(lengths):
        raise ValueError("video shapes and text lengths must be non-empty and aligned")
    if any(len(row) != 3 or min(row) <= 0 for row in rows):
        raise ValueError("each video shape must contain positive F/H/W")
    if any(length < 0 for length in lengths):
        raise ValueError("text lengths must be non-negative")
    return (
        max(length + frames for (frames, _, _), length in zip(rows, lengths)),
        max(height for _, height, _ in rows),
        max(width for _, _, width in rows),
        max(lengths),
    )


def install_seedvr_shape_bounded_rope(
    refiner_runner,
    *,
    torch_module,
    apply_rotary_emb_fn,
    chunk_rows: int = 1024,
):
    """Avoid SeedVR's fixed 1024x128x128 GPU RoPE allocation.

    The multilingual RoPE uses integer positions, so generating only the
    observed extents and then applying the original slices is numerically
    identical to generating the fixed upper-bound tensor first.
    """
    if chunk_rows <= 0:
        raise ValueError("chunk_rows must be positive")
    engine = refiner_runner.runner
    patched = 0

    def bounded_get_freqs(self, vid_shape, txt_shape):
        rows = vid_shape.tolist()
        lengths = txt_shape[:, 0].tolist()
        time_extent, height_extent, width_extent, text_extent = (
            seedvr_required_rope_shape(rows, lengths)
        )
        vid_freqs = self.get_axial_freqs(time_extent, height_extent, width_extent)
        txt_freqs = self.get_axial_freqs(max(text_extent, 1))
        vid_freq_list, txt_freq_list = [], []
        for (frames, height, width), length in zip(rows, lengths):
            vid_freq = vid_freqs[length : length + frames, :height, :width].reshape(
                -1, vid_freqs.size(-1)
            )
            txt_freq = txt_freqs[:length].repeat(1, 3).reshape(-1, vid_freqs.size(-1))
            vid_freq_list.append(vid_freq)
            txt_freq_list.append(txt_freq)
        return (
            torch_module.cat(vid_freq_list, dim=0),
            torch_module.cat(txt_freq_list, dim=0),
        )

    def apply_chunked(freqs, values):
        values_hld = values.permute(1, 0, 2)
        output = torch_module.empty_like(values_hld)
        for start in range(0, values_hld.shape[1], chunk_rows):
            stop = min(start + chunk_rows, values_hld.shape[1])
            rotated = apply_rotary_emb_fn(
                freqs[start:stop],
                values_hld[:, start:stop].to(torch_module.float32),
            ).to(values.dtype)
            output[:, start:stop].copy_(rotated)
        return output.permute(1, 0, 2)

    def bounded_forward(
        self,
        vid_q,
        vid_k,
        vid_shape,
        txt_q,
        txt_k,
        txt_shape,
        cache,
    ):
        vid_freqs, txt_freqs = cache(
            "mmrope_freqs_3d",
            lambda: self.get_freqs(vid_shape, txt_shape),
        )
        return (
            apply_chunked(vid_freqs, vid_q),
            apply_chunked(vid_freqs, vid_k),
            apply_chunked(txt_freqs, txt_q),
            apply_chunked(txt_freqs, txt_k),
        )

    for module in engine.dit.modules():
        if module.__class__.__name__ == "NaMMRotaryEmbedding3d":
            module.get_freqs = MethodType(bounded_get_freqs, module)
            module.forward = MethodType(bounded_forward, module)
            patched += 1
    if patched == 0:
        raise RuntimeError("SeedVR2 shape-bounded RoPE found no target modules")
    return {
        "contract": "seedvr2-shape-bounded-chunked-rope/2",
        "patched_modules": patched,
        "fp32_chunk_rows": chunk_rows,
    }


def seedvr_memory_safe_contract() -> dict[str, object]:
    return {
        "contract": "flowwam-seedvr2-memory-safe-4090/1",
        "conv_max_mem": 0.5,
        "norm_max_mem": 0.5,
        "dit_offload": True,
    }


def install_seedvr_memory_safe_offload(
    refiner_runner,
    *,
    torch_module,
    device: str,
) -> dict[str, object]:
    """Serialize SeedVR2 DiT/VAE GPU residency without changing its math."""
    engine = refiner_runner.runner
    contract = seedvr_memory_safe_contract()
    engine.vae.set_memory_limit(
        conv_max_mem=contract["conv_max_mem"],
        norm_max_mem=contract["norm_max_mem"],
    )
    engine.vae.cpu()
    torch_module.cuda.empty_cache()

    original_vae_encode = engine.vae_encode
    original_inference = engine.inference

    def memory_safe_vae_encode(*args, **kwargs):
        engine.dit.cpu()
        torch_module.cuda.empty_cache()
        engine.vae.to(device=device, dtype=torch_module.bfloat16)
        try:
            return original_vae_encode(*args, **kwargs)
        finally:
            engine.vae.cpu()
            engine.dit.to(device)
            torch_module.cuda.empty_cache()

    def memory_safe_inference(*args, **kwargs):
        kwargs["dit_offload"] = True
        try:
            return original_inference(*args, **kwargs)
        finally:
            engine.vae.cpu()
            torch_module.cuda.empty_cache()

    engine.vae_encode = memory_safe_vae_encode
    engine.inference = memory_safe_inference
    return contract


def _require_under_root(path: Path, root: Path) -> Path:
    resolved = path.resolve(strict=False)
    try:
        resolved.relative_to(root)
    except ValueError as error:
        raise ValueError(f"path must stay under artifact root: {path}") from error
    return resolved


def build_refiner_plan(
    *,
    artifact_root: Path,
    input_dir: Path,
    native_output_dir: Path,
    submission_output_dir: Path,
    expected_count: int,
    official_commit: str,
    seedvr_revision: str,
) -> RefinerPlan:
    """Validate immutable release IDs and a bounded regular-video input set."""
    if official_commit != OFFICIAL_FLOWWAM_COMMIT:
        raise ValueError("official commit mismatch")
    if seedvr_revision != SEEDVR2_REVISION:
        raise ValueError("SeedVR2 revision mismatch")
    if expected_count <= 0:
        raise ValueError("expected count must be positive")

    root = Path(artifact_root).resolve(strict=True)
    source = Path(input_dir)
    if source.is_symlink() or not source.is_dir():
        raise ValueError("input directory must be a regular non-symlink directory")
    source = _require_under_root(source, root)
    native = _require_under_root(Path(native_output_dir), root)
    submission = _require_under_root(Path(submission_output_dir), root)
    for output in (native, submission):
        if output.is_symlink():
            raise ValueError("output directory must not be a symlink")

    videos = tuple(sorted(source.glob("*.mp4")))
    if len(videos) != expected_count:
        raise ValueError(f"expected {expected_count} input videos, found {len(videos)}")
    for video in videos:
        if video.is_symlink() or not video.is_file():
            raise ValueError(f"input video must be a regular non-symlink file: {video}")

    return RefinerPlan(
        input_videos=videos,
        native_output_dir=native,
        submission_output_dir=submission,
    )
