from infinicore.lib import _infinicore
from infinicore.tensor import Tensor


def paged_attention(
    q: Tensor,
    k_cache: Tensor,
    v_cache: Tensor,
    block_tables: Tensor,
    cache_lens: Tensor,
    alibi_slopes: Tensor | None = None,
    scale: float = 1.0,
    k_scale: Tensor | None = None,
    v_scale: Tensor | None = None,
    *,
    out: Tensor | None = None,
):
    if out is None:
        return Tensor(
            _infinicore.paged_attention(
                q._underlying,
                k_cache._underlying,
                v_cache._underlying,
                block_tables._underlying,
                cache_lens._underlying,
                alibi_slopes._underlying if alibi_slopes is not None else None,
                scale,
                k_scale._underlying if k_scale is not None else None,
                v_scale._underlying if v_scale is not None else None,
            )
        )

    _infinicore.paged_attention_(
        out._underlying,
        q._underlying,
        k_cache._underlying,
        v_cache._underlying,
        block_tables._underlying,
        cache_lens._underlying,
        alibi_slopes._underlying if alibi_slopes is not None else None,
        scale,
        k_scale._underlying if k_scale is not None else None,
        v_scale._underlying if v_scale is not None else None,
    )

    return out
