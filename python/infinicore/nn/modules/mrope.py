import numpy as np

import infinicore
from infinicore.ops.mrope import mrope

from ...tensor import Tensor
from .module import InfiniCoreModule as Module


def create_sin_cos_table_numpy(max_position, rotary_dim, theta=10000.0):
    if rotary_dim % 2 != 0:
        raise ValueError("rotary_dim must be even")
    pos = np.arange(0, max_position)
    freqs = 1.0 / (
        theta
        ** (np.arange(0, rotary_dim, 2)[: (rotary_dim // 2)].astype(float) / rotary_dim)
    )
    angles = np.outer(pos, freqs)
    sin_table = np.sin(angles, dtype=np.float32)
    cos_table = np.cos(angles, dtype=np.float32)
    return sin_table, cos_table


def create_sin_cos_table(
    max_position, rotary_dim, theta=10000.0, device=None, dtype=None
):
    sin_table_np, cos_table_np = create_sin_cos_table_numpy(
        max_position, rotary_dim, theta
    )
    return (
        infinicore.from_numpy(sin_table_np, dtype=dtype, device=device),
        infinicore.from_numpy(cos_table_np, dtype=dtype, device=device),
    )


class MRoPE(Module):
    r"""Multimodal rotary position embedding with vLLM-style 2D sin/cos cache."""

    __constants__ = [
        "max_position_embeddings",
        "rope_theta",
        "head_dim",
        "rotary_dim",
        "section",
        "interleaved",
    ]

    def __init__(
        self,
        max_position_embeddings: int,
        rope_theta: float,
        head_dim: int,
        rotary_dim: int,
        section: tuple[int, int, int],
        interleaved: bool = False,
        device=None,
        dtype=None,
    ):
        super().__init__()
        if rotary_dim <= 0 or rotary_dim > head_dim or rotary_dim % 2 != 0:
            raise ValueError("rotary_dim must be positive, even, and <= head_dim")
        if len(section) != 3 or 2 * sum(section) != rotary_dim:
            raise ValueError("section must contain 3 values and sum to rotary_dim / 2")

        factory_kwargs = {
            "device": infinicore.device("cpu", 0) if device is None else device,
            "dtype": infinicore.float32 if dtype is None else dtype,
        }

        self.max_position_embeddings = max_position_embeddings
        self.rope_theta = rope_theta
        self.head_dim = head_dim
        self.rotary_dim = rotary_dim
        self.section = tuple(section)
        self.interleaved = interleaved

        self._sin_table, self._cos_table = create_sin_cos_table(
            self.max_position_embeddings,
            self.rotary_dim,
            self.rope_theta,
            **factory_kwargs,
        )

    def forward(
        self,
        q: Tensor,
        k: Tensor,
        positions: Tensor,
        *,
        out: tuple[Tensor, Tensor] | None = None,
    ) -> tuple[Tensor, Tensor]:
        return mrope(
            q,
            k,
            self._cos_table,
            self._sin_table,
            positions,
            self.head_dim,
            self.rotary_dim,
            self.section[0],
            self.section[1],
            self.section[2],
            self.interleaved,
            out=out,
        )
