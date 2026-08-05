from infinicore.lib import _infinicore


def moe_argsort_bincount_with_inv_pos_(
    tokens_per_experts, sorted_indices, inv_pos, topk_ids, num_experts: int
):
    _infinicore.moe_argsort_bincount_with_inv_pos_(
        tokens_per_experts._underlying,
        sorted_indices._underlying,
        inv_pos._underlying,
        topk_ids._underlying,
        num_experts,
    )
    return tokens_per_experts, sorted_indices, inv_pos
