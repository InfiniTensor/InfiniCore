from typing import Optional

from infinicore.lib import _infinicore
from infinicore.tensor import Tensor


def self_attention(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    scale: Optional[float] = None,
    *,
    out=None,
) -> Tensor:
    r"""Computes scaled dot product attention on query, key and value tensors."""

    seq_len = query.shape[-2]
    total_seq_len = key.shape[-2]

    assert (1 == seq_len and total_seq_len > 1) or (seq_len == total_seq_len), (
        "Incorrect parameter value."
    )

    if out is None:
        return Tensor(
            _infinicore.self_attention(
                query._underlying, key._underlying, value._underlying, scale
            )
        )

    _infinicore.self_attention_(
        out._underlying,
        query._underlying,
        key._underlying,
        value._underlying,
        scale,
    )

    return out
