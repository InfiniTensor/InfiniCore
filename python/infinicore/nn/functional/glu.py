import infinicore
from infinicore.lib import _infinicore
from infinicore.tensor import Tensor


def glu(input: Tensor, dim: int = -1) -> Tensor:
    if infinicore.use_ntops and input.device.type in ("cuda", "musa"):
        return infinicore.ntops.torch.glu(input, dim)

    return Tensor(_infinicore.glu(input._underlying, dim))
