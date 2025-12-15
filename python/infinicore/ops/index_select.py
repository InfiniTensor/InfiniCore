import infinicore
from infinicore.lib import _infinicore
from infinicore.tensor import Tensor


def index_select(input: Tensor, dim: int, index: Tensor, *, out=None) -> Tensor:
    r"""Selects elements from input along a specific dimension."""

    if infinicore.use_ntops and input.device.type in ("cuda", "musa"):
        return infinicore.ntops.torch.index_select(input, dim, index, out=out)
    if out is None:
        return Tensor(
            _infinicore.index_select(input._underlying, dim, index._underlying)
        )

    _infinicore.index_select_(
        out._underlying, input._underlying, dim, index._underlying
    )

    return out
