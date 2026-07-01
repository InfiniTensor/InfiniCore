import numpy as np

import infinicore
from infinicore.nn import functional as F
from infinicore.ops.mrope import mrope

from ...tensor import Tensor
from ..functional import RopeAlgo
from .module import InfiniCoreModule as Module


def create_sin_cos_table_numpy(max_position, rotary_dim, theta=10000.0):
    assert rotary_dim % 2 == 0, "Embedding dimension must be even."
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
    max_position,
    rotary_dim,
    theta=10000.0,
    device=None,
    dtype=None,
):
    sin_table_np, cos_table_np = create_sin_cos_table_numpy(
        max_position, rotary_dim, theta
    )

    sin_table_infini = infinicore.from_numpy(sin_table_np, dtype=dtype, device=device)
    cos_table_infini = infinicore.from_numpy(cos_table_np, dtype=dtype, device=device)

    return sin_table_infini, cos_table_infini


class RoPE(Module):
    r"""Rotary Position Embedding(RoPE).

    Standard RoPE is used when ``mrope_section`` is None. MRoPE is enabled by passing
    ``mrope_section=[t, h, w]`` and then calling the module as ``rope(q, k, positions)``.
    """

    __constants__ = [
        "max_position_embeddings",
        "rope_theta",
        "head_dim",
        "rotary_dim",
        "mrope_section",
        "mrope_interleaved",
    ]
    max_position_embeddings: int
    rope_theta: float
    head_dim: int
    rotary_dim: int

    def __init__(
        self,
        max_position_embeddings: int,
        rope_theta: float,
        head_dim: int,
        device=None,
        dtype=None,
        rotary_dim: int | None = None,
        mrope_section: list[int] | tuple[int, int, int] | None = None,
        mrope_interleaved: bool = False,
    ):
        factory_kwargs = {
            "device": infinicore.device("cpu", 0) if device is None else device,
            "dtype": infinicore.float32 if dtype is None else dtype,
        }
        super().__init__()

        self.max_position_embeddings = max_position_embeddings
        self.rope_theta = rope_theta
        self.head_dim = head_dim
        self.rotary_dim = head_dim if rotary_dim is None else rotary_dim
        self.mrope_section = None if mrope_section is None else list(mrope_section)
        self.mrope_interleaved = mrope_interleaved

        if (
            self.rotary_dim <= 0
            or self.rotary_dim > self.head_dim
            or self.rotary_dim % 2 != 0
        ):
            raise ValueError("rotary_dim must be positive, even, and <= head_dim")
        if self.mrope_section is not None and (
            len(self.mrope_section) != 3
            or 2 * sum(self.mrope_section) != self.rotary_dim
        ):
            raise ValueError(
                "mrope_section must contain 3 values and sum to rotary_dim / 2"
            )

        self._sin_table, self._cos_table = create_sin_cos_table(
            self.max_position_embeddings,
            rotary_dim=self.rotary_dim,
            theta=self.rope_theta,
            **factory_kwargs,
        )

    def forward(
        self,
        states: Tensor,
        position_ids: Tensor,
        *args,
        algo=RopeAlgo.GPT_NEOX,
        out=None,
    ):
        if args:
            q = states
            k = position_ids
            positions = args[0]
            if self.mrope_section is not None:
                return mrope(
                    q,
                    k,
                    self._cos_table,
                    self._sin_table,
                    positions,
                    self.head_dim,
                    self.rotary_dim,
                    self.mrope_section[0],
                    self.mrope_section[1],
                    self.mrope_section[2],
                    self.mrope_interleaved,
                    out=out,
                )

            if out is None:
                q_out = infinicore.empty(q.shape, dtype=q.dtype, device=q.device)
                k_out = infinicore.empty(k.shape, dtype=k.dtype, device=k.device)
            else:
                q_out, k_out = out
            F.rope(q, positions, self._sin_table, self._cos_table, algo=algo, out=q_out)
            F.rope(k, positions, self._sin_table, self._cos_table, algo=algo, out=k_out)
            return q_out, k_out

        if self.mrope_section is not None:
            raise NotImplementedError(
                "MRoPE single-tensor forward is not implemented; use fused forward(q, k, positions) instead"
            )
        target = states if out is None else out
        F.rope(
            states,
            position_ids,
            self._sin_table,
            self._cos_table,
            algo=algo,
            out=target,
        )
        return target
