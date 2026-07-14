from infinicore.dtype import float32, int32
from infinicore.lib import _infinicore
from infinicore.tensor import empty


def _alloc_outputs(gating_output, topk, topk_ids_dtype):
    shape = (gating_output.shape[0], topk)
    # vLLM perf allows fp32 weights for all gating dtypes; use fp32 by default for stable comparisons.
    topk_weights = empty(shape, dtype=float32, device=gating_output.device)
    topk_ids = empty(shape, dtype=topk_ids_dtype, device=gating_output.device)
    token_expert_indices = empty(shape, dtype=int32, device=gating_output.device)
    return topk_weights, topk_ids, token_expert_indices


def moe_topk_softmax_vllm(
    gating_output,
    topk,
    renormalize=False,
    correction_bias=None,
    *,
    out=None,
    topk_ids_dtype=int32,
):
    if out is None:
        out = _alloc_outputs(gating_output, topk, topk_ids_dtype)
    topk_weights, topk_ids, token_expert_indices = out
    _infinicore.moe_topk_softmax_vllm_(
        topk_weights._underlying,
        topk_ids._underlying,
        token_expert_indices._underlying,
        gating_output._underlying,
        renormalize,
        None if correction_bias is None else correction_bias._underlying,
    )
    return out


def moe_topk_sigmoid_vllm(
    gating_output,
    topk,
    renormalize=False,
    correction_bias=None,
    *,
    out=None,
    topk_ids_dtype=int32,
):
    if out is None:
        out = _alloc_outputs(gating_output, topk, topk_ids_dtype)
    topk_weights, topk_ids, token_expert_indices = out
    _infinicore.moe_topk_sigmoid_vllm_(
        topk_weights._underlying,
        topk_ids._underlying,
        token_expert_indices._underlying,
        gating_output._underlying,
        renormalize,
        None if correction_bias is None else correction_bias._underlying,
    )
    return out
