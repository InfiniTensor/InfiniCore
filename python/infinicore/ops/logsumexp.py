import infinicore
from infinicore.lib import _infinicore
from infinicore.tensor import Tensor


def logsumexp(
    input: Tensor, dim: int | None = None, keepdim=False, *, out=None
) -> Tensor:
    r"""Apply the logsumexp function."""

    if infinicore.use_ntops and input.device.type in ("cuda", "musa"):
        return infinicore.ntops.torch.logsumexp(input, dim, keepdim=keepdim, out=out)

    if out is None:
        return Tensor(_infinicore.logsumexp(input._underlying, dim, keepdim))

    _infinicore.logsumexp_(input._underlying, dim, keepdim, out._underlying)

    return out
