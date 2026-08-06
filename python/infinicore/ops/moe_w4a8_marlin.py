from infinicore.lib import _infinicore


def prepare_w4a8_marlin_weight_(output, input):
    _infinicore.prepare_w4a8_marlin_weight_(output._underlying, input._underlying)
    return output


def moe_align_block_size_from_counts_(
    padded_sorted_token_ids,
    expert_ids,
    num_tokens_post_pad,
    sorted_token_ids,
    tokens_per_expert,
    block_size: int,
    routing_topk: int,
):
    _infinicore.moe_align_block_size_from_counts_(
        padded_sorted_token_ids._underlying,
        expert_ids._underlying,
        num_tokens_post_pad._underlying,
        sorted_token_ids._underlying,
        tokens_per_expert._underlying,
        block_size,
        routing_topk,
    )
    return padded_sorted_token_ids, expert_ids, num_tokens_post_pad


def moe_w4a8_marlin_(
    output,
    input,
    marlin_weight,
    input_scale,
    weight_scale,
    topk_weights,
    padded_sorted_token_ids,
    expert_ids,
    num_tokens_post_pad,
    topk: int,
    routing_topk: int,
):
    _infinicore.moe_w4a8_marlin_(
        output._underlying,
        input._underlying,
        marlin_weight._underlying,
        input_scale._underlying,
        weight_scale._underlying,
        None if topk_weights is None else topk_weights._underlying,
        padded_sorted_token_ids._underlying,
        expert_ids._underlying,
        num_tokens_post_pad._underlying,
        topk,
        routing_topk,
    )
    return output
