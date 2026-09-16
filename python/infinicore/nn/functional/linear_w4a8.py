from typing import Optional

from infinicore.lib import _infinicore
from infinicore.tensor import Tensor


def linear_w4a8(
    input: Tensor,
    packed_weight: Tensor,
    weight_scale: Tensor,
    bias: Optional[Tensor] = None,
    alpha: float = 1.0,
    *,
    out: Optional[Tensor] = None,
) -> Tensor:
    args = (
        input._underlying,
        packed_weight._underlying,
        weight_scale._underlying,
        None if bias is None else bias._underlying,
        alpha,
    )
    if out is None:
        return Tensor(_infinicore.linear_w4a8(*args))
    _infinicore.linear_w4a8_(out._underlying, *args)
    return out
