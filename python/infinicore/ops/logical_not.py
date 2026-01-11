import infinicore
from infinicore.lib import _infinicore
from infinicore.tensor import Tensor

def logical_not(input: Tensor, *, out=None) -> Tensor:
    r"""Computes the element-wise logical NOT of the given input tensors."""
    # 1. 如果启用了 ntops 且设备支持，调用 ntops 实现
    if infinicore.use_ntops and input.device.type in ("cuda", "musa"):
        return infinicore.ntops.torch.logical_not(input, out=out)

    # 2. 如果没有提供 out，创建一个新的 Tensor 并返回
    if out is None:
        return Tensor(_infinicore.logical_not(input._underlying))

    # 3. 如果提供了 out，进行原地操作 (In-place operation)
    _infinicore.logical_not_(out._underlying, input._underlying)
    return out