from typing import Optional

from infinicore.lib import _infinicore
from infinicore.tensor import Tensor


def w4a8_moe_shuffle(
    input: Tensor,
    *,
    out: Optional[Tensor] = None,
) -> Tensor:
    if out is None:
        return Tensor(_infinicore.w4a8_moe_shuffle(input._underlying))
    _infinicore.w4a8_moe_shuffle_(out._underlying, input._underlying)
    return out
