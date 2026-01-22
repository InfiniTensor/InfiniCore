import infinicore
from infinicore.lib import _infinicore
from infinicore.tensor import Tensor


def topk(
    input: Tensor,
    k: int,
    dim: int = -1,
    largest: bool = True,
    sorted: bool = True,
    *,
    out=None,
):
    r"""Returns the k largest elements of the given input tensor along a given dimension."""

    if infinicore.use_ntops and input.device.type in ("cuda", "musa"):
        return infinicore.ntops.torch.topk(input, k, dim, largest, sorted, out=out)

    if out is None:
        res_values, res_indices = _infinicore.topk(
            input._underlying, k, dim, largest, sorted
        )
        return Tensor(res_values), Tensor(res_indices)
    else:
        if not isinstance(out, (tuple, list)) or len(out) != 2:
            raise ValueError("out argument must be a tuple of (values, indices)")

        out_values, out_indices = out
        _infinicore.topk_(
            input._underlying,
            out_values._underlying,
            out_indices._underlying,
            k,
            dim,
            largest,
            sorted,
        )
        return out_values, out_indices
