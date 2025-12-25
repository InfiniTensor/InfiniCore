from infinicore.lib import _infinicore
from infinicore.tensor import Tensor


def bi_attention(q, k, v, k_cache, v_cache, pos, *, out=None):
    if out is None:
        return Tensor(
            _infinicore.attention(
                q._underlying,
                k._underlying,
                v._underlying,
                k_cache._underlying,
                v_cache._underlying,
                pos,
            )
        )

    _infinicore.bi_attention_(
        out._underlying,
        q._underlying,
        k._underlying,
        v._underlying,
        k_cache._underlying,
        v_cache._underlying,
        pos,
    )

    return out
