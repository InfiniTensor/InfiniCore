from infinicore.lib import _infinicore
from infinicore.tensor import Tensor


def conv1d(
    input: Tensor,
    weight: Tensor,
    bias: Tensor | None = None,
    stride: int = 1,
    padding: int = 0,
    dilation: int = 1,
    groups: int = 1,
    *,
    out: Tensor | None = None,
) -> Tensor:
    bias_tensor = bias._underlying if bias is not None else None

    if out is None:
        return Tensor(
            _infinicore.conv1d(
                input._underlying,
                weight._underlying,
                bias_tensor,
                stride,
                padding,
                dilation,
                groups,
            )
        )

    _infinicore.conv1d_(
        out._underlying,
        input._underlying,
        weight._underlying,
        bias_tensor,
        stride,
        padding,
        dilation,
        groups,
    )
    return out
