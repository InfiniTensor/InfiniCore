from infinicore.lib import _infinicore
from infinicore.tensor import Tensor


def mamba2_scan(
    x: Tensor,
    dt: Tensor,
    b: Tensor,
    c: Tensor,
    a: Tensor,
    d: Tensor,
    dt_bias: Tensor,
    state: Tensor,
    offsets: Tensor,
    initial_indices: Tensor,
    final_indices: Tensor,
) -> Tensor:
    """Run a packed Mamba-2 scan with FP32 recurrent state.

    Inputs are contiguous device tensors. ``x`` is [tokens, heads, head_dim],
    ``dt`` is [tokens, heads], and ``b``/``c`` are [tokens, groups, state_size].
    ``a``, ``d`` and ``dt_bias`` are FP32 [heads]; ``a`` contains -exp(A_log).
    ``state`` is FP32 [pool, heads, head_dim, state_size]. Offsets and indices
    are int32. Offsets delimit nonempty sequences and span all input tokens.
    Final state indices must be distinct, nonzero valid pool rows. Initial
    rows may be zero or the same request's final row; cross-request read/write
    aliasing is unsupported. State row zero is never modified.
    """
    return Tensor(
        _infinicore.mamba2_scan(
            x._underlying,
            dt._underlying,
            b._underlying,
            c._underlying,
            a._underlying,
            d._underlying,
            dt_bias._underlying,
            state._underlying,
            offsets._underlying,
            initial_indices._underlying,
            final_indices._underlying,
        )
    )
