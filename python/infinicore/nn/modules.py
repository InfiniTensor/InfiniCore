from __future__ import annotations

from typing import Any

from infinicore.lib import _infinicore
from infinicore.tensor import Tensor as TensorWrapper

__all__ = ["RoPE"]


def _unwrap_tensor(tensor: Any):
    if isinstance(tensor, TensorWrapper):
        return tensor._underlying
    return tensor


def _wrap_tensor(tensor: Any) -> TensorWrapper:
    if isinstance(tensor, TensorWrapper):
        return tensor
    return TensorWrapper(tensor)


class RoPE:
    """Python-friendly wrapper for ``_infinicore.RoPE``."""

    def __init__(
        self,
        head_dim: int,
        max_seq_len: int,
        theta: float = 10000.0,
        freq_gen: _infinicore.RoPEAlgo = _infinicore.RoPEAlgo.GPT_J,
        algo: _infinicore.RoPEAlgo = _infinicore.RoPEAlgo.GPT_J,
        dtype: _infinicore.DataType = _infinicore.DataType.F32,
        device=None,
    ) -> None:
        self._module = _infinicore.RoPE(
            head_dim,
            max_seq_len,
            theta,
            freq_gen,
            algo,
            dtype,
            getattr(device, "_underlying", device),
        )

    def forward(self, x, pos):
        output = self._module.forward(_unwrap_tensor(x), _unwrap_tensor(pos))
        return _wrap_tensor(output)

    def __call__(self, x, pos):
        return self.forward(x, pos)
