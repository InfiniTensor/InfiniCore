import infinicore
from infinicore.lib import _infinicore
from infinicore.tensor import Tensor


def logical_not(input: Tensor, *, out=None) -> Tensor:
    r"""Computes the element-wise logical NOT of the given input tensors."""
    if input.device.type != "cpu":
        assert infinicore.use_ntops
        input_zero = infinicore.zeros(
            input.shape, dtype=input.dtype, device=input.device
        )
        result = infinicore.empty(
            input.shape, dtype=infinicore.bool, device=input.device
        )

        infinicore.ntops.torch.eq(input, input_zero, out=result)

        if out is None:
            return result

        infinicore.ntops.torch.bitwise_and(result, result, out=out)
        return out

    # 2. 如果没有提供 out，创建一个新的 Tensor 并返回
    if out is None:
        return Tensor(_infinicore.logical_not(input._underlying))

    # 3. 如果提供了 out，进行原地操作 (In-place operation)
    _infinicore.logical_not_(out._underlying, input._underlying)
    return out
