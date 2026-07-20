"""
Paged Flash-Attention wrapper backed by Ascend CANN aclnnFusedInferAttentionScore.

Runtime requirements:
    - torch_npu (Ascend PyTorch adapter)
    - CANN toolkit (provides libopapi.so, libascendcl.so, etc.)

Provides two entry points:
    - ascend_flash_attn_decode: decode with layout (num_blocks, block_size, num_kv_heads, head_size)
    - ascend_flash_attn_prefill: variable-length prefill (used by mha_varlen)

NOTE: The actual compute is done by the C++ *._ascend.cc files that call
aclnnFusedInferAttentionScore directly.  This Python module exists as a
convenience wrapper for users who prefer a pure-Python calling convention
(e.g. for testing or benchmarking) and mirrors the moore_mate_flash_attn.py
interface.
"""

import torch

try:
    import torch_npu  # noqa: F401 - register NPU backend

    _NPU_AVAILABLE = True
except ImportError:
    _NPU_AVAILABLE = False


def is_available() -> bool:
    return _NPU_AVAILABLE


def _check_available():
    if not _NPU_AVAILABLE:
        raise RuntimeError(
            "torch_npu is not installed. "
            "Please install torch_npu and CANN toolkit first."
        )


@torch.inference_mode()
def ascend_flash_attn_decode(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    block_tables: torch.Tensor,
    seq_lens: torch.Tensor,
    scale: float,
    block_size: int,
    max_seq_len: int,
) -> torch.Tensor:
    """
    Decode entry point for Ascend NPU.

    This is a thin wrapper that validates inputs and delegates to
    infinicore's C++ mha_kvcache op (which calls
    aclnnFusedInferAttentionScore internally when ENABLE_ASCEND_FLASH_ATTN
    is defined at build time).

    Args:
        q: (num_seqs, num_heads, head_size) on NPU
        k_cache: (num_blocks, block_size, num_kv_heads, head_size) on NPU
        v_cache: (num_blocks, block_size, num_kv_heads, head_size) on NPU
        block_tables: (num_seqs, max_blocks_per_seq) int32 on NPU
        seq_lens: (num_seqs,) int32 on NPU
        scale: softmax scale (typically 1/sqrt(head_size))
        block_size: paged KV block size
        max_seq_len: maximum sequence length (reserved, not used by aclnn)

    Returns:
        out: (num_seqs, num_heads, head_size) on NPU
    """
    _check_available()

    import infinicore

    num_seqs, num_heads, head_size = q.shape

    q_4d = q.unsqueeze(1)  # (num_seqs, 1, num_heads, head_size)

    out_4d = infinicore.mha_kvcache(
        q=infinicore.from_torch(q_4d),
        k_cache=infinicore.from_torch(k_cache),
        v_cache=infinicore.from_torch(v_cache),
        seqlens_k=infinicore.from_torch(seq_lens),
        block_table=infinicore.from_torch(block_tables),
        alibi_slopes=None,
        scale=scale,
    )

    return out_4d.squeeze(1)  # (num_seqs, num_heads, head_size)


@torch.inference_mode()
def ascend_flash_attn_prefill(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    block_tables: torch.Tensor,
    scale: float,
    max_seqlen_q: int,
    max_seqlen_k: int,
    block_size: int,
    causal: bool = True,
) -> torch.Tensor:
    """
    Variable-length prefill entry point for Ascend NPU.

    Delegates to infinicore's C++ mha_varlen op (which calls
    aclnnFusedInferAttentionScore internally).

    Args:
        q: (total_q, num_heads, head_size) on NPU
        k_cache: (num_blocks, block_size, num_kv_heads, head_size) on NPU
        v_cache: (num_blocks, block_size, num_kv_heads, head_size) on NPU
        cu_seqlens_q: (batch_size + 1,) int32 on NPU
        cu_seqlens_k: (batch_size + 1,) int32 on NPU
        block_tables: (batch_size, max_blocks_per_seq) int32 on NPU
        scale: softmax scale
        max_seqlen_q: max query sequence length in the batch
        max_seqlen_k: max key sequence length in the batch
        block_size: paged KV block size
        causal: whether to apply causal mask

    Returns:
        out: (total_q, num_heads, head_size) on NPU
    """
    _check_available()

    import infinicore

    out = infinicore.mha_varlen(
        q=infinicore.from_torch(q),
        k=infinicore.from_torch(k_cache),
        v=infinicore.from_torch(v_cache),
        cum_seqlens_q=infinicore.from_torch(cu_seqlens_q),
        cum_seqlens_k=infinicore.from_torch(cu_seqlens_k),
        block_table=infinicore.from_torch(block_tables),
        max_seqlen_q=max_seqlen_q,
        max_seqlen_k=max_seqlen_k,
        alibi_slopes=None,
        scale=scale,
    )

    return out


if __name__ == "__main__":
    if not is_available():
        raise SystemExit("torch_npu not available, please install torch_npu first.")

    device = torch.device("npu:0")

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

    out = ascend_flash_attn_decode(
        q=q,
        k_cache=k_cache,
        v_cache=v_cache,
        block_tables=block_tables,
        seq_lens=seq_lens,
        scale=head_size**-0.5,
        block_size=block_size,
        max_seq_len=max_seq_len,
    )
    torch.npu.synchronize()
    assert list(out.shape) == list(q.shape), f"Expected {q.shape}, got {out.shape}"
    print("ascend_flash_attn_decode test passed")

    # --- Prefill test ---
    total_q = 64  # sum of sequence lengths
    cu_seqlens_q = torch.tensor([0, 32, 64], dtype=torch.int32, device=device)
    cu_seqlens_k = torch.tensor([0, 32, 64], dtype=torch.int32, device=device)
    q_prefill = torch.randn(
        total_q, num_heads, head_size, dtype=torch.float16, device=device
    )
    out_prefill = ascend_flash_attn_prefill(
        q=q_prefill,
        k_cache=k_cache,
        v_cache=v_cache,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_k=cu_seqlens_k,
        block_tables=block_tables,
        scale=head_size**-0.5,
        max_seqlen_q=32,
        max_seqlen_k=32,
        block_size=block_size,
        causal=True,
    )
    torch.npu.synchronize()
    assert list(out_prefill.shape) == list(q_prefill.shape), (
        f"Expected {q_prefill.shape}, got {out_prefill.shape}"
    )
    print("ascend_flash_attn_prefill test passed")
