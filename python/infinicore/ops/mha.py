from infinicore.lib import _infinicore
from infinicore.tensor import Tensor


def mha(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    alibi_slopes: Tensor | None = None,
    scale: float = 1.0,
    is_causal: bool = False,
    *,
    out: Tensor | None = None,
):
    if out is None:
        return Tensor(
            _infinicore.mha(
                q._underlying,
                k._underlying,
                v._underlying,
                alibi_slopes._underlying if alibi_slopes is not None else None,
                scale,
                is_causal,
            )
        )

    _infinicore.mha_(
        out._underlying,
        q._underlying,
        k._underlying,
        v._underlying,
        alibi_slopes._underlying if alibi_slopes is not None else None,
        scale,
        is_causal,
    )

    return out
