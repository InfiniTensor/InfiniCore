import infinicore
from infinicore.lib import _infinicore
from infinicore.tensor import Tensor


def logical_and(input: Tensor, other: Tensor, *, out=None) -> Tensor:
    r"""Computes the element-wise logical AND of the given input tensors."""
    if input.device.type != "cpu":
        assert infinicore.use_ntops
        lhs = infinicore.empty(input.shape, dtype=infinicore.bool, device=input.device)
        rhs = infinicore.empty(other.shape, dtype=infinicore.bool, device=other.device)
        input_zero = infinicore.zeros(
            input.shape, dtype=input.dtype, device=input.device
        )
        other_zero = infinicore.zeros(
            other.shape, dtype=other.dtype, device=other.device
        )

        infinicore.ntops.torch.ne(input, input_zero, out=lhs)
        infinicore.ntops.torch.ne(other, other_zero, out=rhs)

        if out is None:
            out = infinicore.empty(
                input.shape, dtype=infinicore.bool, device=input.device
            )

        infinicore.ntops.torch.bitwise_and(lhs, rhs, out=out)
        return out

    if out is None:
        return Tensor(_infinicore.logical_and(input._underlying, other._underlying))

    _infinicore.logical_and_(out._underlying, input._underlying, other._underlying)
    return out
