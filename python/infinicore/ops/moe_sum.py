# Copyright (c) 2025, InfiniCore
"""MoE top-k accumulate: [M, topk, H] -> [M, H]."""

from infinicore.lib import _infinicore
from infinicore.tensor import Tensor


def moe_sum(input, out=None):
    """Sum expert outputs over the top-k axis (vLLM ``moe_sum`` semantics)."""
    if out is None:
        return Tensor(_infinicore.moe_sum(input._underlying))
    _infinicore.moe_sum_(out._underlying, input._underlying)
    return out
