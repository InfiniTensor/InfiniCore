# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#
# Grouped sigmoid MoE routing aligned with vLLM's ``grouped_topk`` native path
# (``vllm/model_executor/layers/fused_moe/router/grouped_topk_router.py``).
# Default implementation is a vendored pure-Torch copy (no vLLM import).
# ``INFINILM_MOE_FUSED_STACK=upstream`` calls upstream ``grouped_topk`` for validation.
# Optional: ``INFINILM_USE_VLLM_GROUPED_TOPK_KERNEL=1`` uses ``vllm._custom_ops.grouped_topk`` on CUDA
# (vendor stack only; skipped when ``INFINILM_MOE_FUSED_STACK=upstream``).

from __future__ import annotations

import os

import torch

from infinicore.moe_fused_stack import resolve_moe_fused_stack

from .torch_register import direct_register_custom_op, infinilm_fused_lib


def _use_sorted_topk() -> bool:
    return os.environ.get("INFINILM_MOE_ROUTER_SORTED_TOPK", "0") == "1"


def grouped_sigmoid_topk_torch_reference(
    gating_output: torch.Tensor,
    topk: int,
    renormalize: bool,
    num_expert_group: int,
    topk_group: int,
    routed_scaling_factor: float,
    e_score_correction_bias: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pure Torch: sigmoid scores + bias for selection; weights from unbiased sigmoid (vLLM semantics)."""
    scores = gating_output.sigmoid()
    num_token = scores.size(0)
    original_scores = scores
    scores = scores + e_score_correction_bias.unsqueeze(0).to(dtype=scores.dtype)
    use_sorted = _use_sorted_topk()
    group_scores = (
        scores.view(num_token, num_expert_group, -1)
        .topk(2, dim=-1, largest=True, sorted=use_sorted)[0]
        .sum(dim=-1)
    )
    group_idx = torch.topk(group_scores, k=topk_group, dim=-1, largest=True, sorted=use_sorted)[1]
    group_mask = torch.zeros_like(group_scores)
    group_mask.scatter_(1, group_idx, 1)
    score_mask = (
        group_mask.unsqueeze(-1)
        .expand(num_token, num_expert_group, scores.size(-1) // num_expert_group)
        .reshape(num_token, -1)
    )
    tmp_scores = scores.masked_fill(~score_mask.bool(), float("-inf"))
    topk_ids = torch.topk(tmp_scores, k=topk, dim=-1, largest=True, sorted=use_sorted)[1]
    topk_weights = original_scores.gather(1, topk_ids)
    if renormalize:
        topk_weights = topk_weights / (topk_weights.sum(dim=-1, keepdim=True) + 1e-20)
    topk_weights = topk_weights * routed_scaling_factor
    return topk_weights.to(torch.float32), topk_ids.to(torch.int32)


def _grouped_topk_via_vllm_poc(
    gating_output: torch.Tensor,
    topk: int,
    renormalize: bool,
    num_expert_group: int,
    topk_group: int,
    routed_scaling_factor: float,
    e_score_correction_bias: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    from vllm.model_executor.layers.fused_moe.router.grouped_topk_router import (  # type: ignore[import-not-found]
        grouped_topk,
    )

    hidden_states = torch.zeros(
        (gating_output.size(0), 1),
        device=gating_output.device,
        dtype=gating_output.dtype,
    )
    return grouped_topk(
        hidden_states,
        gating_output,
        topk,
        renormalize,
        num_expert_group,
        topk_group,
        "sigmoid",
        routed_scaling_factor,
        e_score_correction_bias,
    )


def _grouped_topk_via_vllm_cuda_kernel(
    gating_output: torch.Tensor,
    topk: int,
    renormalize: bool,
    num_expert_group: int,
    topk_group: int,
    routed_scaling_factor: float,
    e_score_correction_bias: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor] | None:
    if not gating_output.is_cuda:
        return None
    if num_expert_group > 32 or topk > 32:
        return None
    try:
        import vllm._custom_ops as vops  # type: ignore[import-not-found]
    except ImportError:
        return None
    bias = e_score_correction_bias.to(device=gating_output.device, dtype=gating_output.dtype).contiguous()
    try:
        return vops.grouped_topk(
            gating_output.contiguous(),
            num_expert_group,
            topk_group,
            topk,
            renormalize,
            float(routed_scaling_factor),
            bias,
            1,
        )
    except (NotImplementedError, RuntimeError):
        return None


def minicpm5_grouped_sigmoid_topk(
    router_logits: torch.Tensor,
    e_score_correction_bias: torch.Tensor,
    topk: int,
    renormalize: bool,
    num_expert_group: int,
    topk_group: int,
    routed_scaling_factor: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    router_logits = router_logits.contiguous()
    e_score_correction_bias = e_score_correction_bias.contiguous()

    if resolve_moe_fused_stack() == "upstream":
        return _grouped_topk_via_vllm_poc(
            router_logits,
            topk,
            renormalize,
            num_expert_group,
            topk_group,
            routed_scaling_factor,
            e_score_correction_bias,
        )

    if os.environ.get("INFINILM_USE_VLLM_GROUPED_TOPK_KERNEL", "0") == "1":
        fused = _grouped_topk_via_vllm_cuda_kernel(
            router_logits,
            topk,
            renormalize,
            num_expert_group,
            topk_group,
            routed_scaling_factor,
            e_score_correction_bias,
        )
        if fused is not None:
            return fused[0], fused[1]

    return grouped_sigmoid_topk_torch_reference(
        router_logits,
        topk,
        renormalize,
        num_expert_group,
        topk_group,
        routed_scaling_factor,
        e_score_correction_bias,
    )


def minicpm5_grouped_sigmoid_topk_fake(
    router_logits: torch.Tensor,
    e_score_correction_bias: torch.Tensor,
    topk: int,
    renormalize: bool,
    num_expert_group: int,
    topk_group: int,
    routed_scaling_factor: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    n = router_logits.shape[0]
    dev = router_logits.device
    w = torch.empty((n, topk), dtype=torch.float32, device=dev)
    ids = torch.empty((n, topk), dtype=torch.int32, device=dev)
    return w, ids


direct_register_custom_op(
    op_name="minicpm5_grouped_sigmoid_topk",
    target_lib=infinilm_fused_lib,
    op_func=minicpm5_grouped_sigmoid_topk,
    fake_impl=minicpm5_grouped_sigmoid_topk_fake,
)
