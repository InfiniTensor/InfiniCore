import infinicore
from infinicore.lib import _infinicore
from infinicore.tensor import Tensor


def bucketize(
    input: Tensor,
    boundaries: Tensor,
    *,
    out=None,
    right=False,
) -> Tensor:
    r"""Apply the bucketize function."""

    if infinicore.use_ntops and input.device.type in ("cuda", "musa"):
        return infinicore.ntops.torch.bucketize(input, boundaries, out=out, right=right)

    if out is None:
        return Tensor(
            _infinicore.bucketize(
                input._underlying, boundaries._underlying, bool(right)
            )
        )

    _infinicore.bucketize_(
        input._underlying, boundaries._underlying, out._underlying, bool(right)
    )
    return out
