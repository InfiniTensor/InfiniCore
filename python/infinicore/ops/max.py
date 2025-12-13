import infinicore
from infinicore.lib import _infinicore
from infinicore.tensor import Tensor


def max(
    input: Tensor, dim: int | None = None, keepdim=False, *, out=None
) -> Tensor | tuple[Tensor, Tensor]:
    r"""Apply the max function."""

    if dim is None:
        if infinicore.use_ntops and input.device.type in ("cuda", "musa"):
            return infinicore.ntops.torch.max(input, out=out)

        if out is None:
            return Tensor(_infinicore.max_global(input._underlying))

        _infinicore.max_global_(input._underlying, out._underlying)

        return out
    else:
        if infinicore.use_ntops and input.device.type in ("cuda", "musa"):
            return infinicore.ntops.torch.max(input, dim, keepdim=keepdim, out=out)

        if out is None:
            res, res_idx = _infinicore.max_reduce(input._underlying, dim, keepdim)
            return Tensor(res), Tensor(res_idx)

        _infinicore.max_reduce_(
            input._underlying, out[0]._underlying, out[1]._underlying, dim, keepdim
        )

        return out
