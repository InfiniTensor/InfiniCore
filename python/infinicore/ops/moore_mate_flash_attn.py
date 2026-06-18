"""
Paged Flash-Attention wrapper backed by MooreThreads mate (flash_attn).

Runtime requirements:
    - torch (with MUSA)
    - mate (repo : https://github.com/MooreThreads/mate)

Provides three entry points:
    - moore_mate_flash_attn_decode: decode with layout (num_blocks, block_size, num_kv_heads, head_size)
    - moore_mate_flash_attn_prefill: variable-length prefill (used by mha_varlen)
"""

import torch

try:
    from flash_attn import flash_attn_with_kvcache, get_scheduler_metadata

    _MATE_AVAILABLE = True
except ImportError:
    _MATE_AVAILABLE = False


def is_available() -> bool:
    """Return True if mate / flash_attn is installed and importable."""
    return _MATE_AVAILABLE


def _check_mate_available():
    """Raise a clear error if mate is not installed."""
    if not _MATE_AVAILABLE:
        raise RuntimeError(
            "flash_attn (mate) is not installed. "
            "Please build and install MooreThreads/mate first."
        )


# =============================================================================
# Decode kernels
# =============================================================================


@torch.inference_mode()
def moore_mate_flash_attn_decode(
    q: torch.Tensor,  # (num_seqs, num_heads, head_size)
    k_cache: torch.Tensor,  # (num_blocks, block_size, num_kv_heads, head_size)
    v_cache: torch.Tensor,  # (num_blocks, block_size, num_kv_heads, head_size)
    block_tables: torch.Tensor,  # (num_seqs, max_blocks_per_seq)
    seq_lens: torch.Tensor,  # (num_seqs,)
    scale: float,
    block_size: int,
    max_seq_len: int,
) -> torch.Tensor:
    """
    Decode entry point with native flash_attn KV cache layout (B, P, H, D).
    No layout conversion is performed.
    """
    _check_mate_available()

    num_seqs, num_heads, head_size = q.shape
    num_kv_heads = k_cache.shape[2]
    device = q.device

    cache_seqlens = seq_lens.to(torch.int32)
    page_table = block_tables.to(torch.int32)
    cu_seqlens_q = torch.arange(0, num_seqs + 1, dtype=torch.int32, device=device)
    pack_gqa = num_heads != num_kv_heads

    metadata = get_scheduler_metadata(
        batch_size=num_seqs,
        max_seqlen_q=1,
        max_seqlen_k=max_seq_len,
        num_heads_q=num_heads,
        num_heads_kv=num_kv_heads,
        headdim=head_size,
        cache_seqlens=cache_seqlens,
        qkv_dtype=q.dtype,
        headdim_v=head_size,
        cu_seqlens_q=cu_seqlens_q,
        page_size=block_size,
        causal=False,
        window_size=(None, None),
        pack_gqa=pack_gqa,
    )

    out, *_ = flash_attn_with_kvcache(
        q=q,
        k_cache=k_cache,
        v_cache=v_cache,
        cache_seqlens=cache_seqlens,
        page_table=page_table,
        cu_seqlens_q=cu_seqlens_q,
        max_seqlen_q=1,
        softmax_scale=scale,
        causal=False,
        scheduler_metadata=metadata,
        pack_gqa=pack_gqa,
        return_softmax_lse=True,
    )
    return out


# =============================================================================
# Prefill kernel (variable-length)
# =============================================================================


@torch.inference_mode()
def moore_mate_flash_attn_prefill(
    q: torch.Tensor,  # (total_q, num_heads, head_size) -- varlen unpad
    k_cache: torch.Tensor,  # (num_blocks, block_size, num_kv_heads, head_size)
    v_cache: torch.Tensor,  # (num_blocks, block_size, num_kv_heads, head_size)
    cu_seqlens_q: torch.Tensor,  # (batch+1,) int32
    cu_seqlens_k: torch.Tensor,  # (batch+1,) int32
    block_tables: torch.Tensor,  # (batch, max_blocks_per_seq)
    scale: float,
    max_seqlen_q: int,
    max_seqlen_k: int,
    block_size: int,
    causal: bool = True,  # prefill is typically causal
) -> torch.Tensor:
    """
    Variable-length prefill entry point. Layout follows flash_attn (B, P, H, D).
    Intended to be called from the C++ mha_varlen Moore branch.
    """
    _check_mate_available()

    cu_seqlens_q = cu_seqlens_q.to(torch.int32)
    cu_seqlens_k = cu_seqlens_k.to(torch.int32)
    page_table = block_tables.to(torch.int32)

    # mate uses cache_seqlens (per-batch KV length), derived from cu_seqlens_k
    cache_seqlens = (cu_seqlens_k[1:] - cu_seqlens_k[:-1]).to(torch.int32).contiguous()

    batch_size = cache_seqlens.shape[0]
    num_heads = q.shape[1]
    head_size = q.shape[2]
    num_kv_heads = k_cache.shape[2]
    pack_gqa = num_heads != num_kv_heads

    metadata = get_scheduler_metadata(
        batch_size=batch_size,
        max_seqlen_q=int(max_seqlen_q),
        max_seqlen_k=int(max_seqlen_k),
        num_heads_q=num_heads,
        num_heads_kv=num_kv_heads,
        headdim=head_size,
        cache_seqlens=cache_seqlens,
        qkv_dtype=q.dtype,
        headdim_v=head_size,
        cu_seqlens_q=cu_seqlens_q,
        page_size=block_size,
        causal=causal,
        window_size=(None, None),
        pack_gqa=pack_gqa,
    )

    out, *_ = flash_attn_with_kvcache(
        q=q,
        k_cache=k_cache,
        v_cache=v_cache,
        cache_seqlens=cache_seqlens,
        page_table=page_table,
        cu_seqlens_q=cu_seqlens_q,
        max_seqlen_q=int(max_seqlen_q),
        softmax_scale=scale,
        causal=causal,
        scheduler_metadata=metadata,
        pack_gqa=pack_gqa,
        return_softmax_lse=True,
    )
    return out


# =============================================================================
# Self tests
# =============================================================================


def _test_moore_mate_flash_attn_decode():
    """Test moore_mate_flash_attn_decode with flash_attn layout (B, P, H, D)."""
    print("\n=== Test 1: moore_mate_flash_attn_decode (decode, flash_attn layout) ===")
    device = torch.device("musa")

    num_seqs, num_heads, num_kv_heads = 2, 8, 2
    head_size, block_size, max_seq_len = 128, 16, 64
    num_blocks = 32

    q = torch.randn(num_seqs, num_heads, head_size, dtype=torch.float16, device=device)
    k_cache = torch.randn(
        num_blocks,
        block_size,
        num_kv_heads,
        head_size,
        dtype=torch.float16,
        device=device,
    )
    v_cache = torch.randn(
        num_blocks,
        block_size,
        num_kv_heads,
        head_size,
        dtype=torch.float16,
        device=device,
    )
    block_tables = torch.zeros(num_seqs, 4, dtype=torch.int32, device=device)
    block_tables[0, 0] = 0
    block_tables[1, 0] = 1
    seq_lens = torch.tensor([32, 48], dtype=torch.int32, device=device)

    out = moore_mate_flash_attn_decode(
        q=q,
        k_cache=k_cache,
        v_cache=v_cache,
        block_tables=block_tables,
        seq_lens=seq_lens,
        scale=head_size**-0.5,
        block_size=block_size,
        max_seq_len=max_seq_len,
    )
    torch.musa.synchronize()
    print(f"output shape = {tuple(out.shape)}")
    assert out.shape == q.shape
    print("moore_mate_flash_attn_decode passed")


def _test_moore_mate_flash_attn_prefill():
    """Test moore_mate_flash_attn_prefill with variable-length input."""
    print("\n=== Test 2: moore_mate_flash_attn_prefill (varlen prefill) ===")
    device = torch.device("musa")
    torch.manual_seed(666)
    torch.musa.manual_seed(666)

    batch_size = 2
    seqlens_q = [55, 222]
    seqlens_kv = [55, 222]  # prefill: q_len == k_len
    num_heads, num_kv_heads = 8, 2
    head_size, block_size = 128, 16

    total_q = sum(seqlens_q)
    max_q = max(seqlens_q)
    max_k = max(seqlens_kv)
    num_blocks_per_seq = (max_k + block_size - 1) // block_size
    num_blocks = batch_size * num_blocks_per_seq

    q_unpad = torch.randn(
        total_q, num_heads, head_size, dtype=torch.float16, device=device
    )
    k_cache = torch.randn(
        num_blocks,
        block_size,
        num_kv_heads,
        head_size,
        dtype=torch.float16,
        device=device,
    )
    v_cache = torch.randn(
        num_blocks,
        block_size,
        num_kv_heads,
        head_size,
        dtype=torch.float16,
        device=device,
    )

    cu_seqlens_q = torch.tensor(
        [0] + list(torch.cumsum(torch.tensor(seqlens_q), 0).tolist()),
        dtype=torch.int32,
        device=device,
    )
    cu_seqlens_k = torch.tensor(
        [0] + list(torch.cumsum(torch.tensor(seqlens_kv), 0).tolist()),
        dtype=torch.int32,
        device=device,
    )
    block_tables = torch.arange(num_blocks, dtype=torch.int32, device=device).view(
        batch_size, num_blocks_per_seq
    )

    out = moore_mate_flash_attn_prefill(
        q=q_unpad,
        k_cache=k_cache,
        v_cache=v_cache,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_k=cu_seqlens_k,
        block_tables=block_tables,
        scale=head_size**-0.5,
        max_seqlen_q=max_q,
        max_seqlen_k=max_k,
        block_size=block_size,
        causal=True,
    )
    torch.musa.synchronize()
    print(f"output shape = {tuple(out.shape)}")
    assert out.shape == q_unpad.shape
    print("moore_mate_flash_attn_prefill passed")


if __name__ == "__main__":
    if not is_available():
        raise SystemExit("mate / flash_attn not available, please build mate first.")

    _test_moore_mate_flash_attn_decode()
    _test_moore_mate_flash_attn_prefill()
    print("\nAll tests passed.")
