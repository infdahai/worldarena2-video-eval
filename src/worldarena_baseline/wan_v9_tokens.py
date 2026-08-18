"""Shared four-modality action tokenizer for Wan v9."""

from __future__ import annotations

from collections.abc import Mapping
import re
from typing import Final

import torch
from torch import Tensor, nn


_FEATURE_WIDTHS: Final = {
    "translation": 4,
    "rotation": 4,
    "image": 6,
    "gripper": 3,
}
_MODALITIES: Final = tuple(_FEATURE_WIDTHS)
_SHA256: Final = re.compile(r"[0-9a-f]{64}")


class V9ActionTokenizer(nn.Module):
    """Map four physical modalities into four shared action tokens.

    Input feature tensors use ``(batch, arm, transition, feature)``.  Output
    uses ``(batch, transition, arm, modality, action_width)``.  The module has
    no time-slot or arm-identity embedding: those identities are structural in
    the downstream phase-locked attention.
    """

    def __init__(
        self,
        statistics: Mapping[str, object],
        *,
        action_width: int = 256,
        hidden_width: int = 128,
    ) -> None:
        super().__init__()
        if action_width <= 0 or hidden_width <= 0:
            raise ValueError("tokenizer widths must be positive")
        if statistics.get("contract") != "wan-v9-transition-normalization/1":
            raise ValueError("v9 normalization contract mismatch")
        receipt = statistics.get("receipt_sha256")
        if not isinstance(receipt, str) or _SHA256.fullmatch(receipt) is None:
            raise ValueError("v9 normalization receipt hash is invalid")
        self.normalization_receipt_sha256 = receipt
        self.action_width = int(action_width)
        self.hidden_width = int(hidden_width)

        for modality, width in _FEATURE_WIDTHS.items():
            entry = statistics.get(modality)
            if not isinstance(entry, Mapping):
                raise ValueError(f"{modality} statistics are missing")
            mean = torch.as_tensor(entry.get("mean"), dtype=torch.float32)
            std = torch.as_tensor(entry.get("std"), dtype=torch.float32)
            if mean.shape != (width,) or std.shape != (width,):
                raise ValueError(f"{modality} statistics have invalid shape")
            if not torch.isfinite(mean).all() or not torch.isfinite(std).all() or not torch.all(std > 0):
                raise ValueError(f"{modality} statistics must be finite with positive std")
            self.register_buffer(f"_{modality}_mean", mean, persistent=True)
            self.register_buffer(f"_{modality}_std", std, persistent=True)
            setattr(
                self,
                f"{modality}_mlp",
                nn.Sequential(
                    nn.Linear(width, hidden_width),
                    nn.SiLU(),
                    nn.Linear(hidden_width, action_width),
                ),
            )
        self.type_embedding = nn.Parameter(torch.zeros(4, action_width, dtype=torch.float32))
        nn.init.normal_(self.type_embedding, mean=0.0, std=0.02)

    def _tokenize(self, modality: str, value: Tensor) -> Tensor:
        width = _FEATURE_WIDTHS[modality]
        if value.ndim != 4 or tuple(value.shape[1:3]) != (2, 20) or value.shape[-1] != width:
            raise ValueError(f"{modality} must have shape (batch,2,20,{width})")
        if not torch.isfinite(value).all():
            raise ValueError(f"{modality} contains non-finite values")
        mean = getattr(self, f"_{modality}_mean").float()
        std = getattr(self, f"_{modality}_std").float()
        normalized = (value.float() - mean) / std
        mlp = getattr(self, f"{modality}_mlp")
        return mlp(normalized.to(dtype=mlp[0].weight.dtype))

    def forward(
        self,
        features: Mapping[str, Tensor],
        arm_present: Tensor,
    ) -> tuple[Tensor, Tensor]:
        if set(features) != set(_MODALITIES):
            raise ValueError("v9 tokenizer requires exactly four modalities")
        if arm_present.ndim != 3 or tuple(arm_present.shape[1:]) != (2, 20) or arm_present.dtype != torch.bool:
            raise ValueError("arm_present must have boolean shape (batch,2,20)")
        batch = arm_present.shape[0]
        encoded = []
        for index, modality in enumerate(_MODALITIES):
            value = features[modality]
            if value.shape[0] != batch or value.device != arm_present.device:
                raise ValueError(f"{modality} batch/device differs from arm_present")
            token = self._tokenize(modality, value)
            embedding = self.type_embedding[index].to(device=token.device, dtype=token.dtype)
            encoded.append(token + embedding)
        # B,A,T,M,W -> B,T,A,M,W
        tokens = torch.stack(encoded, dim=3)
        tokens = tokens * arm_present[..., None, None].to(dtype=tokens.dtype)
        return tokens.permute(0, 2, 1, 3, 4).contiguous(), arm_present.permute(0, 2, 1).contiguous()

