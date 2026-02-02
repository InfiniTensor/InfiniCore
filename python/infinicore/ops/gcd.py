import infinicore
from infinicore.lib import _infinicore
from infinicore.tensor import Tensor


def gcd(input: Tensor, other: Tensor, *, out=None) -> Tensor:
    r"""Computes the element-wise greatest common divisor (GCD)."""

    if infinicore.use_ntops and input.device.type in ("cuda", "musa"):
        return infinicore.ntops.torch.gcd(input, other, out=out)

    if out is None:
        return Tensor(_infinicore.gcd(input._underlying, other._underlying))

    _infinicore.gcd_(input._underlying, other._underlying, out._underlying)
    return out
