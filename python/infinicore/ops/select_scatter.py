import infinicore
from infinicore.lib import _infinicore
from infinicore.tensor import Tensor


def select_scatter(input: Tensor, src: Tensor, dim: int, index: int) -> Tensor:
    if infinicore.use_ntops and input.device.type in ("cuda", "musa"):
        return infinicore.ntops.torch.select_scatter(input, src, dim, index)

    return Tensor(
        _infinicore.select_scatter(input._underlying, src._underlying, dim, index)
    )
