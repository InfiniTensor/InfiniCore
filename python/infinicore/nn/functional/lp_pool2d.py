import infinicore
from infinicore.lib import _infinicore
from infinicore.tensor import Tensor


def lp_pool2d(
    input: Tensor,
    norm_type: float,
    kernel_size: int | tuple[int, int],
    stride: int | tuple[int, int] | None = None,
    ceil_mode: bool = False,
):
    r"""Applies a 2D power-average pooling over an input signal composed of several input planes."""
    if isinstance(kernel_size, int):
        kernel_size = (kernel_size, kernel_size)

    if isinstance(stride, int):
        stride = (stride, stride)

    if stride is None:
        stride = kernel_size

    if infinicore.use_ntops and input.device.type in ("cuda", "musa"):
        return infinicore.ntops.torch.lp_pool2d(
            input, norm_type, kernel_size, stride, ceil_mode
        )

    return Tensor(
        _infinicore.lp_pool2d(
            input._underlying, norm_type, kernel_size, stride, ceil_mode
        )
    )
