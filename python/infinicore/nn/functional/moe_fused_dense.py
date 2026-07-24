from infinicore.lib import _infinicore
from infinicore.tensor import Tensor


def moe_fused_dense(
    hidden_states: Tensor,
    w13: Tensor,
    w2: Tensor,
    topk_weights: Tensor,
    topk_ids: Tensor,
    sorted_token_ids: Tensor,
    expert_ids: Tensor,
    num_tokens_post_padded: Tensor,
) -> Tensor:
    """Run the fused MoE dense block with aligned routing metadata."""
    return Tensor(
        _infinicore.moe_fused_dense(
            hidden_states._underlying,
            w13._underlying,
            w2._underlying,
            topk_weights._underlying,
            topk_ids._underlying,
            sorted_token_ids._underlying,
            expert_ids._underlying,
            num_tokens_post_padded._underlying,
        )
    )
