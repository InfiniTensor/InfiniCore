import infinicore
from infinicore.lib import _infinicore
from infinicore.tensor import Tensor


def fold(
    input: Tensor,
    output_size: int | tuple[int, int],
    kernel_size: int | tuple[int, int],
    dilation: int | tuple[int, int] = 1,
    padding: int | tuple[int, int] = 0,
    stride: int | tuple[int, int] = 1,
) -> Tensor:
    r"""Combines an array of sliding local blocks into a large containing tensor. Currently, only unbatched (3D) or batched (4D) image-like output tensors are supported."""

    if isinstance(output_size, int):
        output_size = (output_size, output_size)

    if isinstance(kernel_size, int):
        kernel_size = (kernel_size, kernel_size)

    if isinstance(dilation, int):
        dilation = (dilation, dilation)

    if isinstance(padding, int):
        padding = (padding, padding)

    if isinstance(stride, int):
        stride = (stride, stride)

    assert input.ndim in (3, 4), "only 3D or 4D input tensors are supported"
    assert len(output_size) == 2, "output_size must be a tuple of two integers (H, W)"
    assert len(kernel_size) == 2, "kernel_size must be a tuple of two integers (kH, kW)"
    assert len(dilation) == 2, "dilation must be a tuple of two integers (dH, dW)"
    assert len(padding) == 2, "padding must be a tuple of two integers (pH, pW)"
    assert len(stride) == 2, "stride must be a tuple of two integers (sH, sW)"

    if infinicore.use_ntops and input.device.type in ("cuda", "musa"):
        return infinicore.ntops.torch.fold(
            input, output_size, kernel_size, dilation, padding, stride
        )

    return Tensor(
        _infinicore.fold(
            input._underlying, output_size, kernel_size, dilation, padding, stride
        )
    )
