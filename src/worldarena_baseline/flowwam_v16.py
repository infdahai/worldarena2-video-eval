"""Fail-closed contracts for the bounded Official FlowWAM v16 LoRA run."""

from __future__ import annotations

import math
import re
import statistics
from collections.abc import Iterable, Mapping, Sequence

V16_CONTRACT = "flowwam-official-v16-qv-lora/1"
_EVAL_NAME = re.compile(r"^(?P<task>.+)_episode_(?P<episode>\d+)\.mp4$")
_LORA_NAME = re.compile(
    r"^pipe\.dit\..*\.(?:self_attn|cross_attn)\.(?P<target>q|v)"
    r"\.lora_(?:A|B)(?:\.[^.]+)?\.weight$"
)
_INFERENCE_LORA_NAME = re.compile(
    r"^blocks\.\d+\.(?P<attention>self_attn|cross_attn)\."
    r"(?P<target>q|v)\.lora_(?P<side>A|B)\.default\.weight$"
)


class FlowWAMV16ContractError(RuntimeError):
    """Raised when a v16 experiment would violate its frozen contract."""


def evaluate_prompt_triplets(
    *,
    baseline: Mapping[int, Sequence[float]],
    candidate: Mapping[int, Sequence[float]],
    expected_per_variant: int,
) -> dict[str, object]:
    """Require Instruction Following to be preserved for all three prompts."""
    expected_variants = {0, 1, 2}
    if set(baseline) != expected_variants or set(candidate) != expected_variants:
        raise FlowWAMV16ContractError("prompt triplet variants must be exactly 0, 1, 2")
    if expected_per_variant <= 0:
        raise FlowWAMV16ContractError("expected prompt rows must be positive")

    normalized: dict[str, dict[int, list[float]]] = {"baseline": {}, "candidate": {}}
    for label, source in (("baseline", baseline), ("candidate", candidate)):
        for variant in sorted(expected_variants):
            values = [float(value) for value in source[variant]]
            if len(values) != expected_per_variant or any(
                not math.isfinite(value) or not 0.0 <= value <= 1.0 for value in values
            ):
                raise FlowWAMV16ContractError(
                    f"invalid {label} instruction scores for variant {variant}"
                )
            normalized[label][variant] = values

    baseline_means = {
        str(variant): statistics.fmean(normalized["baseline"][variant])
        for variant in sorted(expected_variants)
    }
    candidate_means = {
        str(variant): statistics.fmean(normalized["candidate"][variant])
        for variant in sorted(expected_variants)
    }
    variant_preserved = {
        str(variant): candidate_means[str(variant)] >= baseline_means[str(variant)]
        for variant in sorted(expected_variants)
    }
    baseline_overall = statistics.fmean(baseline_means.values())
    candidate_overall = statistics.fmean(candidate_means.values())
    return {
        "passed": all(variant_preserved.values())
        and candidate_overall >= baseline_overall,
        "expected_per_variant": expected_per_variant,
        "baseline_means": baseline_means,
        "candidate_means": candidate_means,
        "variant_delta": {
            key: candidate_means[key] - baseline_means[key] for key in baseline_means
        },
        "variant_preserved": variant_preserved,
        "baseline_overall": baseline_overall,
        "candidate_overall": candidate_overall,
        "overall_delta": candidate_overall - baseline_overall,
    }


def filter_v16_lora_state_for_inference(
    state: Mapping[str, object],
    *,
    attention_scope: str,
    rank: int,
    expected_source_pairs: int,
    require_self_only_source: bool = False,
) -> tuple[dict[str, object], dict[str, object]]:
    """Validate the frozen adapter and optionally retain self-attention only."""
    if attention_scope not in {"all", "self"}:
        raise FlowWAMV16ContractError(
            f"unsupported inference attention scope: {attention_scope}"
        )
    keys = set(state)
    matches = {key: _INFERENCE_LORA_NAME.fullmatch(key) for key in keys}
    invalid = sorted(key for key, match in matches.items() if match is None)
    if invalid:
        raise FlowWAMV16ContractError(
            "v16 inference LoRA key contract mismatch: " + ", ".join(invalid)
        )
    a_keys = sorted(key for key, match in matches.items() if match.group("side") == "A")
    if len(a_keys) != expected_source_pairs:
        raise FlowWAMV16ContractError(
            f"expected {expected_source_pairs} source LoRA pairs, got {len(a_keys)}"
        )
    if require_self_only_source:
        cross_keys = sorted(
            key for key in a_keys if matches[key].group("attention") == "cross_attn"
        )
        if attention_scope != "self" or cross_keys:
            raise FlowWAMV16ContractError(
                "self-only source layout contains cross-attention LoRA pairs"
            )

    retained: dict[str, object] = {}
    retained_pairs = 0
    filtered_cross_pairs = 0
    for a_key in a_keys:
        b_key = a_key.replace(".lora_A.default.weight", ".lora_B.default.weight")
        if b_key not in state:
            raise FlowWAMV16ContractError(f"missing LoRA B pair for {a_key}")
        if state[a_key].shape[0] != rank or state[b_key].shape[1] != rank:
            raise FlowWAMV16ContractError(f"LoRA rank mismatch for {a_key}")
        is_cross = matches[a_key].group("attention") == "cross_attn"
        if attention_scope == "self" and is_cross:
            filtered_cross_pairs += 1
            continue
        retained[a_key] = state[a_key]
        retained[b_key] = state[b_key]
        retained_pairs += 1

    if not retained_pairs:
        raise FlowWAMV16ContractError("inference LoRA filter retained no pairs")
    targets = (
        ["self_attn.q", "self_attn.v"] if attention_scope == "self" else ["q", "v"]
    )
    report = {
        "attention_scope": attention_scope,
        "source_pairs": len(a_keys),
        "loaded_pairs": retained_pairs,
        "filtered_cross_attention_pairs": filtered_cross_pairs,
        "target_modules": targets,
    }
    if require_self_only_source:
        report["source_layout"] = "self-qv-only"
    return retained, report


def _episode_key(row: Mapping[str, object]) -> tuple[str, int]:
    try:
        return str(row["task"]), int(row["episode_index"])
    except (KeyError, TypeError, ValueError) as exc:
        raise FlowWAMV16ContractError(
            f"invalid training episode identity: {row!r}"
        ) from exc


def _eval_key(name: str) -> tuple[str, int]:
    match = _EVAL_NAME.fullmatch(name)
    if match is None:
        raise FlowWAMV16ContractError(f"invalid evaluation video identity: {name}")
    return match.group("task"), int(match.group("episode"))


def validate_zero_eval_overlap(
    train_rows: Sequence[Mapping[str, object]],
    eval_video_names: Sequence[str],
    *,
    split_name: str,
) -> dict:
    """Reject any episode overlap using task + episode identity, not filenames."""
    train_keys = {_episode_key(row) for row in train_rows}
    eval_keys = {_eval_key(name) for name in eval_video_names}
    collisions = sorted(train_keys & eval_keys)
    if collisions:
        rendered = ", ".join(
            f"{task} episode {episode}" for task, episode in collisions
        )
        raise FlowWAMV16ContractError(f"{split_name} leakage: {rendered}")
    return {
        "split": split_name,
        "train_rows": len(train_rows),
        "eval_rows": len(eval_video_names),
        "collision_count": 0,
        "collisions": [],
        "passed": True,
    }


def validate_trainable_parameter_names(names: Iterable[str]) -> dict:
    """Allow only PEFT LoRA A/B parameters attached to DiT attention q/v."""
    materialized = sorted(set(names))
    invalid = [name for name in materialized if _LORA_NAME.fullmatch(name) is None]
    if not materialized or invalid:
        raise FlowWAMV16ContractError(
            "v16 trainable whitelist violation: "
            + (", ".join(invalid) if invalid else "no trainable parameters")
        )
    return {
        "passed": True,
        "trainable_count": len(materialized),
        "trainable_names": materialized,
        "targets": ["q", "v"],
    }


def enforce_qv_lora_only(model: object) -> dict:
    """Refreeze every parameter except DiT attention q/v LoRA A/B weights."""
    trainable = []
    for name, parameter in model.named_parameters():
        enabled = _LORA_NAME.fullmatch(name) is not None
        parameter.requires_grad_(enabled)
        if enabled:
            trainable.append(name)
    return validate_trainable_parameter_names(trainable)


def build_v16_contract(*, parent_sha256: str, manifest_sha256: str) -> dict:
    """Return the immutable experiment contract recorded in every receipt."""
    return {
        "contract": V16_CONTRACT,
        "parent_sha256": parent_sha256,
        "manifest_sha256": manifest_sha256,
        "lora": {
            "rank": 8,
            "alpha": 8,
            "targets": ["q", "v"],
            "learning_rate": 2e-6,
        },
        "production_shape": {
            "frames": 121,
            "width": 640,
            "height": 480,
            "batch_size": 1,
            "flow_max_magnitude": 20.0,
        },
        "gradient_checkpointing": True,
        "checkpoint_steps": [50, 100, 200],
        "losses": ["official_rgb_diffusion"],
        "forbidden_losses": ["sam", "depth", "jepa", "trajectory"],
    }
