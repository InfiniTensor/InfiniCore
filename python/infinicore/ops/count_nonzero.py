import infinicore
from infinicore.tensor import Tensor


def _normalize_dims(dim, ndim):
    if dim is None:
        return tuple(range(ndim))

    if isinstance(dim, int):
        dims = (dim,)
    else:
        dims = tuple(dim)

    normalized = []
    for d in dims:
        d = int(d)
        if d < 0:
            d += ndim
        if d < 0 or d >= ndim:
            raise IndexError("dim out of range")
        if d in normalized:
            raise ValueError("dim contains duplicate values")
        normalized.append(d)

    return tuple(normalized)


def _output_rank(ndim, reduce_dims):
    return ndim - len(reduce_dims)


def count_nonzero(input: Tensor, dim=None) -> Tensor:
    r"""Count the number of non-zero values in the tensor."""

    if infinicore.use_ntops and input.device.type in ("cuda", "musa"):
        result = infinicore.ntops.torch.count_nonzero(input, dim=dim)

        reduce_dims = _normalize_dims(dim, input.ndim)

        # scalar 输出在 ntops kernel 中是 shape (1,)，这里 squeeze 成 shape ()。
        if _output_rank(input.ndim, reduce_dims) == 0:
            return infinicore.squeeze(result, 0)

        return result
