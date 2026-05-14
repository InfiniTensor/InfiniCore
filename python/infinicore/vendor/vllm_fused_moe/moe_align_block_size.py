# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import torch


def _cdiv(a: int, b: int) -> int:
    return (a + b - 1) // b


def moe_align_block_size(
    topk_ids: torch.Tensor,
    block_size: int,
    num_experts: int,
    expert_map: torch.Tensor | None = None,
    pad_sorted_ids: bool = False,
    ignore_invalid_experts: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Pure-PyTorch replacement for vLLM `_moe_C.moe_align_block_size`.

    Tokens are grouped by ascending expert id; within each expert, flat indices follow
    row-major appearance order (same as ascending flat index). Padding uses sentinel ``numel``.
    """
    del pad_sorted_ids  # unused; vLLM JSON tuning may set this in upstream only.

    if expert_map is not None:
        raise NotImplementedError(
            "infinilm vendor MoE: expert_map is not implemented (need EP routing semantics)."
        )

    device = topk_ids.device
    numel = topk_ids.numel()
    flat_e = topk_ids.flatten().to(torch.int64)
    flat_pos = torch.arange(numel, device=device, dtype=torch.int64)

    max_num_tokens_padded = numel + num_experts * (block_size - 1)

    pieces: list[torch.Tensor] = []
    block_experts: list[torch.Tensor] = []
    for e in range(num_experts):
        mask = flat_e == e
        idx = flat_pos[mask].to(torch.int32)
        c = int(idx.numel())
        pad_len = _cdiv(c, block_size) * block_size
        if pad_len > c:
            pad = torch.full((pad_len - c,), numel, device=device, dtype=torch.int32)
            idx = torch.cat([idx, pad])
        pieces.append(idx)
        n_blocks = pad_len // block_size
        block_experts.append(torch.full((n_blocks,), e, device=device, dtype=torch.int32))

    cat_sorted = torch.cat(pieces)
    cat_exp = torch.cat(block_experts)

    max_num_m_blocks = _cdiv(max_num_tokens_padded, block_size)
    sorted_ids = torch.full((max_num_tokens_padded,), numel, device=device, dtype=torch.int32)
    sorted_ids[: cat_sorted.numel()] = cat_sorted

    expert_ids = torch.full((max_num_m_blocks,), -1, device=device, dtype=torch.int32)
    expert_ids[: cat_exp.numel()] = cat_exp

    num_tokens_post_pad = torch.tensor([cat_sorted.numel()], device=device, dtype=torch.int32)

    _ = ignore_invalid_experts  # EP remap unsupported; signature matches vLLM.

    return sorted_ids, expert_ids, num_tokens_post_pad
