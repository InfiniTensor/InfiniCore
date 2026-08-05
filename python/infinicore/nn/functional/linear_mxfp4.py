from infinicore.lib import _infinicore
from infinicore.tensor import Tensor


def linear_mxfp4(
    input: Tensor,
    packed_weight: Tensor,
    weight_scale: Tensor,
    bias=None,
    alpha: float = 1.0,
    out=None,
) -> Tensor:
    if out is None:
        return Tensor(
            _infinicore.linear_mxfp4(
                input._underlying,
                packed_weight._underlying,
                weight_scale._underlying,
                None if bias is None else bias._underlying,
                alpha,
            )
        )

    _infinicore.linear_mxfp4_(
        out._underlying,
        input._underlying,
        packed_weight._underlying,
        weight_scale._underlying,
        None if bias is None else bias._underlying,
        alpha,
    )
    return out
