from infinicore.lib import _infinicore
from infinicore.tensor import Tensor


def fused_moe_mxfp4(
    input: Tensor,
    selected_experts: Tensor,
    routing_weights: Tensor,
    w13_packed: Tensor,
    w13_scale: Tensor,
    w2_packed: Tensor,
    w2_scale: Tensor,
    activation: int = 1,
    out=None,
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
    )
    if out is None:
        return Tensor(_infinicore.fused_moe_mxfp4(*args))

    _infinicore.fused_moe_mxfp4_(out._underlying, *args)
    return out
