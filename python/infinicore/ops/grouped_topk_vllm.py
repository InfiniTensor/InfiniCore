from infinicore.dtype import float32, int32
from infinicore.lib import _infinicore
from infinicore.tensor import empty


def grouped_topk_vllm(
    scores,
    num_expert_group,
    topk_group,
    topk,
    renormalize,
    routed_scaling_factor=1.0,
    bias=None,
    scoring_func="softmax",
    *,
    out=None,
    topk_ids_dtype=int32,
):
    if bias is None:
        raise RuntimeError(
            "grouped_topk_vllm currently requires correction bias; vllm_iluvatar perf no-bias path mismatches reference"
        )
    if out is None:
        shape = (scores.shape[0], topk)
        out = (
            empty(shape, dtype=float32, device=scores.device),
            empty(shape, dtype=topk_ids_dtype, device=scores.device),
        )
    topk_weights, topk_ids = out
    _infinicore.grouped_topk_vllm_(
        topk_weights._underlying,
        topk_ids._underlying,
        scores._underlying,
        num_expert_group,
        topk_group,
        renormalize,
        float(routed_scaling_factor),
        None if bias is None else bias._underlying,
        scoring_func,
    )
    return out
