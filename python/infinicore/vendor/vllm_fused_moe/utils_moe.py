# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import functools
from math import prod

import torch
from packaging import version


def _resize_cache(x: torch.Tensor, v: tuple[int, ...]) -> torch.Tensor:
    assert prod(v) <= x.numel(), f"{v} ({prod(v)}) <= {x.shape} ({x.numel()})"
    return x.flatten()[: prod(v)].view(*v)


def moe_kernel_quantize_input(
    A: torch.Tensor,
    A_scale: torch.Tensor | None,
    quant_dtype: None | torch.dtype | str,
    per_act_token_quant: bool,
    block_shape: list[int] | None = None,
    is_fp4_scale_swizzled: bool = True,
    ocp_mx_scheme: str | None = None,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    del per_act_token_quant, block_shape, is_fp4_scale_swizzled
    if ocp_mx_scheme is not None:
        raise NotImplementedError("OCP MX fused MoE path is not vendored.")
    if quant_dtype is not None:
        raise NotImplementedError(
            "Quantized fused MoE activations are not supported in the InfiniLM vendor build."
        )
    return A, A_scale


@functools.cache
def disable_inplace() -> bool:
    try:
        return version.parse(torch.__version__.split("+", 1)[0]) >= version.parse("2.9")
    except Exception:
        return False
