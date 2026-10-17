from infinicore.lib import _infinicore
from infinicore.tensor import Tensor

__all__ = ["linear_w8a16"]


def linear_w8a16(input, weight, scale, bias=None, *, out=None):
    if out is None:
        return Tensor(
            _infinicore.linear_w8a16(
                input._underlying,
                weight._underlying,
                scale._underlying,
                None if bias is None else bias._underlying,
            )
        )

    _infinicore.linear_w8a16_(
        out._underlying,
        input._underlying,
        weight._underlying,
        scale._underlying,
        None if bias is None else bias._underlying,
    )
    return out
