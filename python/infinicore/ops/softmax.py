import infinicore
from infinicore.tensor import Tensor


def softmax(input: Tensor, dim: int, dtype=None, *, out=None) -> Tensor:
    r"""Apply the softmax function over a given dimension."""

    if dim is None:
        raise TypeError("softmax() missing required argument: 'dim'")

    if not infinicore.use_ntops or input.device.type not in ("cuda", "musa"):
        raise RuntimeError("softmax is currently only available with ntops on CUDA/MUSA devices")

    if out is None:
        target_dtype = dtype if dtype is not None else input.dtype
        return infinicore.ntops.torch.softmax(input, dim, dtype=target_dtype)

    if not isinstance(out, Tensor):
        raise TypeError(f"out must be a Tensor, got {type(out).__name__}")

    if out.shape != input.shape:
        raise ValueError("out tensor must have the same shape as input")

    if out.device != input.device:
        raise ValueError("out tensor must be on the same device as input")

    target_dtype = dtype if dtype is not None else out.dtype

    if dtype is not None and out.dtype != target_dtype:
        raise TypeError("out tensor dtype must match the dtype argument")

    # Reuse the cached ntops kernel to write directly into the provided output tensor.
    from infinicore.ntops.torch.utils import _cached_make

    kernel = _cached_make(infinicore.ntops.kernels.softmax.premake, input.ndim, dim)
    kernel(input, out)

    return out
