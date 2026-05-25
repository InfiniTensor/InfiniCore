import infinicore
from infinicore.lib import _infinicore
from infinicore.tensor import Tensor


def _normalize_dim(dim: int, ndim: int) -> int:
    if dim < 0:
        dim += ndim

    if dim < 0 or dim >= ndim:
        raise IndexError("dim out of range")

    return dim


def _normalize_start_end(start, end, size: int):
    if start is None:
        start = 0

    if end is None:
        end = size

    if start < 0:
        start += size

    if end < 0:
        end += size

    start = max(0, min(start, size))
    end = max(0, min(end, size))

    return start, end


def slice_scatter(
    input: Tensor,
    src: Tensor,
    dim: int = 0,
    start: int | None = None,
    end: int | None = None,
    step: int = 1,
    *,
    out=None,
) -> Tensor:
    r"""Embed the values of src into input at the given slice."""

    if input.ndim == 0:
        raise RuntimeError("slice_scatter does not support zero-dimensional input")

    if step is None:
        step = 1

    if step <= 0:
        raise ValueError("slice_scatter only supports step > 0")

    dim = _normalize_dim(dim, input.ndim)
    start, end = _normalize_start_end(start, end, input.shape[dim])
    if infinicore.use_ntops and input.device.type in ("cuda", "musa") and out is None:
        return infinicore.ntops.torch.slice_scatter(
            input,
            src,
            dim=dim,
            start=start,
            end=end,
            step=step,
        )
    if out is None:
        return Tensor(
            _infinicore.slice_scatter(
                input._underlying,
                src._underlying,
                dim,
                start,
                end,
                step,
            )
        )
    _infinicore.slice_scatter_(
        out._underlying,
        input._underlying,
        src._underlying,
        dim,
        start,
        end,
        step,
    )

    return out