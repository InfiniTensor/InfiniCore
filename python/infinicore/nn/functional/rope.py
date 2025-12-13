from infinicore.lib import _infinicore
from infinicore.tensor import Tensor


class RopeAlgo:
    r"""Different types of RoPE algorithms."""

    GPT_J = _infinicore.RoPEAlgo.GPT_J
    GPT_NEOX = _infinicore.RoPEAlgo.GPT_NEOX


def rope(
    x: Tensor,
    pos_ids: Tensor,
    sin_table: Tensor,
    cos_table: Tensor,
    algo: RopeAlgo = RopeAlgo.GPT_NEOX,
    *,
    out=None,
) -> Tensor:
    r"""Rotary Position Embedding(RoPE)."""

    bs, seq_len, num_heads, head_dim = x.shape
    x_stride = x.stride()
    assert seq_len * x_stride[1] == x_stride[0], (
        "x need to be continuous in dim=0 and dim=1"
    )

    x = x.view((bs * seq_len, num_heads, head_dim))
    bs, num = pos_ids.shape
    pos_ids = pos_ids.view((bs * num,))

    if out is None:
        return Tensor(
            _infinicore.rope(
                x._underlying,
                pos_ids._underlying,
                sin_table._underlying,
                cos_table._underlying,
                algo,
            )
        ).view((bs, seq_len, num_heads, head_dim))

    out = out.view((bs * seq_len, num_heads, head_dim))
    _infinicore.rope_(
        out._underlying,
        x._underlying,
        pos_ids._underlying,
        sin_table._underlying,
        cos_table._underlying,
        algo,
    )
    return out.view((bs, seq_len, num_heads, head_dim))
