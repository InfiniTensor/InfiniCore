# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

from enum import Enum

import torch
import torch.nn.functional as F


class MoEActivation(Enum):
    SILU = "silu"
    GELU = "gelu"
    RELU2 = "relu2"
    SWIGLUOAI = "swigluoai"
    SWIGLUSTEP = "swiglustep"
    SILU_NO_MUL = "silu_no_mul"
    GELU_NO_MUL = "gelu_no_mul"
    RELU2_NO_MUL = "relu2_no_mul"

    @property
    def is_gated(self) -> bool:
        return not self.value.endswith("_no_mul")

    @classmethod
    def from_str(cls, s: str) -> MoEActivation:
        for member in cls:
            if member.value == s:
                return member
        valid = [m.value for m in cls]
        raise ValueError(f"Unknown MoE activation: {s!r}. Valid activations: {valid}")


def apply_moe_activation(
    activation: MoEActivation,
    output: torch.Tensor,
    input: torch.Tensor,
) -> torch.Tensor:
    assert input.dim() == 2, "Input must be 2D"
    assert output.dim() == 2, "Output must be 2D"
    if activation.is_gated:
        assert output.size(-1) * 2 == input.size(-1), (
            f"{activation.value} expects 2x ratio: "
            f"{output.size(-1) * 2} vs {input.size(-1)}"
        )
    else:
        assert output.size(-1) == input.size(-1), (
            f"{activation.value} expects equal sizes: "
            f"{output.size(-1)} vs {input.size(-1)}"
        )

    if activation == MoEActivation.SILU:
        gate, up = input.chunk(2, dim=-1)
        torch.mul(F.silu(gate), up, out=output)
    elif activation == MoEActivation.GELU:
        gate, up = input.chunk(2, dim=-1)
        torch.mul(F.gelu(gate), up, out=output)
    elif activation in (MoEActivation.SWIGLUOAI, MoEActivation.SWIGLUSTEP):
        raise NotImplementedError(f"{activation} requires vLLM Triton ops in upstream.")
    elif activation == MoEActivation.SILU_NO_MUL:
        output.copy_(F.silu(input))
    elif activation == MoEActivation.GELU_NO_MUL:
        output.copy_(F.gelu(input))
    elif activation == MoEActivation.RELU2_NO_MUL:
        tmp = F.relu(input)
        torch.square(tmp, out=output)
    else:
        raise ValueError(f"Unsupported FusedMoe activation: {activation}")

    return output
