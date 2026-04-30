from infinicore.lib import _infinicore
from infinicore.tensor import Tensor


def _ensure_head_dim_contiguous(tensor: Tensor) -> Tensor:
    if tensor.ndim > 0 and tensor.stride(tensor.ndim - 1) != 1:
        return tensor.contiguous()
    return tensor


def paged_attention_prefill(
    q: Tensor,
    k_cache: Tensor,
    v_cache: Tensor,
    block_tables: Tensor,
    history_lens: Tensor,
    cu_seqlens_q: Tensor,
    alibi_slopes: Tensor | None = None,
    scale: float = 1.0,
    *,
    out: Tensor | None = None,
):
    k_cache = _ensure_head_dim_contiguous(k_cache)
    v_cache = _ensure_head_dim_contiguous(v_cache)
    alibi_ptr = alibi_slopes._underlying if alibi_slopes is not None else None

    if out is None:
        return Tensor(
            _infinicore.paged_attention_prefill(
                q._underlying,
                k_cache._underlying,
                v_cache._underlying,
                block_tables._underlying,
                history_lens._underlying,
                cu_seqlens_q._underlying,
                alibi_ptr,
                scale,
            )
        )

    _infinicore.paged_attention_prefill_(
        out._underlying,
        q._underlying,
        k_cache._underlying,
        v_cache._underlying,
        block_tables._underlying,
        history_lens._underlying,
        cu_seqlens_q._underlying,
        alibi_ptr,
        scale,
    )

    return out
