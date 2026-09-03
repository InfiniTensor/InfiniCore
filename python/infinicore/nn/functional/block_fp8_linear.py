from infinicore.lib import _infinicore
from infinicore.tensor import Tensor


def block_fp8_linear(
    input: Tensor,
    weight: Tensor,
    weight_scale: Tensor,
    out=None,
) -> Tensor:
    if out is None:
        return Tensor(
            _infinicore.block_fp8_linear(
                input._underlying,
                weight._underlying,
                weight_scale._underlying,
            )
        )

    _infinicore.block_fp8_linear_(
        out._underlying,
        input._underlying,
        weight._underlying,
        weight_scale._underlying,
    )
    return out
