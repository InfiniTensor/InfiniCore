import infinicore
from infinicore.lib import _infinicore
from infinicore.tensor import Tensor


def sum(
    input: Tensor,
    dim: int | tuple[int] | list[int] | None = None,
    keepdim=False,
    *,
    dtype=None,
    out=None,
) -> Tensor:
    r"""Apply the sum function."""

    if infinicore.use_ntops and input.device.type in ("cuda", "musa"):
        return infinicore.ntops.torch.sum(
            input, dim, keepdim=keepdim, dtype=dtype, out=out
        )

    if dim is None:
        if out is None:
            return Tensor(_infinicore.sum_global(input._underlying))
        _infinicore.sum_global_(input._underlying, out._underlying)
        return out

    else:
        target_dim = dim
        if isinstance(target_dim, (tuple, list)):
            if len(target_dim) == 1:
                target_dim = target_dim[0]
        if out is None:
            return Tensor(
                _infinicore.sum_reduce(input._underlying, target_dim, keepdim)
            )

        _infinicore.sum_reduce_(input._underlying, out._underlying, target_dim, keepdim)
        return out
