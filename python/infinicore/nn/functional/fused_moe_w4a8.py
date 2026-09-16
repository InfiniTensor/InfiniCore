from typing import Optional

from infinicore.lib import _infinicore
from infinicore.tensor import Tensor


def fused_moe_w4a8(
    input: Tensor,
    selected_experts: Tensor,
    routing_weights: Tensor,
    w13_packed: Tensor,
    w13_scale: Tensor,
    w2_packed: Tensor,
    w2_scale: Tensor,
    activation: int = 1,
    weights_are_aiter_shuffled: bool = False,
    *,
    out: Optional[Tensor] = None,
) -> Tensor:
    args = (
        input._underlying,
        selected_experts._underlying,
        routing_weights._underlying,
        w13_packed._underlying,
        w13_scale._underlying,
        w2_packed._underlying,
        w2_scale._underlying,
        activation,
        weights_are_aiter_shuffled,
    )
    if out is None:
        return Tensor(_infinicore.fused_moe_w4a8(*args))
    _infinicore.fused_moe_w4a8_(out._underlying, *args)
    return out
