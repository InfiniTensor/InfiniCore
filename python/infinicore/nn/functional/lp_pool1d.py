import infinicore
from infinicore.lib import _infinicore
from infinicore.tensor import Tensor


def lp_pool1d(
    input: Tensor,
    norm_type: float,
    kernel_size: int,
    stride: int | None = None,
    ceil_mode: bool = False,
):
    r"""Applies a 1D power-average pooling over an input signal composed of several input planes."""

    if stride is None:
        stride = kernel_size

    if infinicore.use_ntops and input.device.type in ("cuda", "musa"):
        return infinicore.ntops.torch.lp_pool1d(
            input, norm_type, kernel_size, stride, ceil_mode
        )

    return Tensor(
        _infinicore.lp_pool1d(
            input._underlying, norm_type, kernel_size, stride, ceil_mode
        )
    )
