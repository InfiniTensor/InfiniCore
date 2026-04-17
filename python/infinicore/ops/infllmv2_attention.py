"""
InfLLM-V2 attention ops (varlen and kvcache).
Available only when InfiniCore is built with ENABLE_INFLLMV2 and linked to infllmv2 .so.
"""

from infinicore.lib import _infinicore
from infinicore.tensor import Tensor

_native_infllmv2_attention_varlen = getattr(
    _infinicore, "infllmv2_attention_varlen", None
)
_native_infllmv2_attention_kvcache = getattr(
    _infinicore, "infllmv2_attention_kvcache", None
)

_MISSING_MSG = (
    "infllmv2_attention_varlen / infllmv2_attention_kvcache not found in _infinicore. "
    "Build InfiniCore with: xmake f --aten=y --infllmv2=y (auto-detect under third_party/infllmv2_cuda_impl) "
    "or --infllmv2=/abs/path/to/libinfllm_v2.so (recommended), then xmake build/install."
)


def infllmv2_attention_varlen(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    cu_seqlens_q: Tensor,
    cu_seqlens_k: Tensor,
    max_seqlen_q: int,
    max_seqlen_k: int,
    scale: float = 1.0,
    causal: bool = True,
    window_size_left: int = -1,
    window_size_right: int = -1,
):
    """InfLLM-V2 varlen attention. q,k,v unpadded; cu_seqlens_q/k [batch+1]. Returns [total_q, nheads, head_dim]."""
    if _native_infllmv2_attention_varlen is None:
        raise NotImplementedError(_MISSING_MSG)
    return Tensor(
        _native_infllmv2_attention_varlen(
            q._underlying,
            k._underlying,
            v._underlying,
            cu_seqlens_q._underlying,
            cu_seqlens_k._underlying,
            max_seqlen_q,
            max_seqlen_k,
            scale,
            causal,
            window_size_left,
            window_size_right,
        )
    )


def infllmv2_attention_kvcache(
    q: Tensor,
    k_cache: Tensor,
    v_cache: Tensor,
    cache_lens: Tensor,
    scale: float = 1.0,
    causal: bool = True,
    window_size_left: int = -1,
    window_size_right: int = -1,
):
    """InfLLM-V2 KV-cache (decode) attention. Returns [batch, seqlen_q, nheads, head_dim]."""
    if _native_infllmv2_attention_kvcache is None:
        raise NotImplementedError(_MISSING_MSG)
    return Tensor(
        _native_infllmv2_attention_kvcache(
            q._underlying,
            k_cache._underlying,
            v_cache._underlying,
            cache_lens._underlying,
            scale,
            causal,
            window_size_left,
            window_size_right,
        )
    )
