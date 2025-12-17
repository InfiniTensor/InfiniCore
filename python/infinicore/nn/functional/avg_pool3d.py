import infinicore
from infinicore.lib import _infinicore
from infinicore.tensor import Tensor


def _zeros_pad(input: Tensor, padding: tuple[int, ...]) -> Tensor:
    r"""Pad a tensor.

    Args:
        input (Tensor): The input tensor.
        padding (tuple[int, ...]): The padding sizes.

    Returns:
        Tensor: The padded tensor.
    """
    output_shape = []
    for i in range(input.ndim):
        output_shape.append(input.size(i) + 2 * padding[i])

    output = infinicore.empty(output_shape, dtype=input.dtype, device=input.device)
    output = infinicore.nn.init.zeros_(output)

    # 使用 narrow 函数获取对应的位置，然后复制数据
    # 需要逐维度进行 narrow 操作
    output_view = output
    for dim in range(len(input.size())):
        output_view = infinicore.narrow(output_view, dim, padding[dim], input.size(dim))

    # 将输入数据复制到输出张量的对应位置
    infinicore.add(input, output_view, out=output_view)

    return output


def avg_pool3d(
    input: Tensor,
    kernel_size: tuple[int, int, int] | int,
    stride: tuple[int, int, int] | int | None = None,
    padding: tuple[int, int, int] | int = 0,
    ceil_mode: bool = False,
):
    r"""Applies a 3D average pooling over an input signal composed of several input
    planes."""
    if isinstance(kernel_size, int):
        kernel_size = (kernel_size, kernel_size, kernel_size)

    if isinstance(stride, int):
        stride = (stride, stride, stride)

    if stride is None:
        stride = kernel_size

    if isinstance(padding, int):
        padding = [padding, padding, padding]

    if infinicore.use_ntops and input.device.type in ("cuda", "musa"):
        padding = [0, 0] + list(padding)

        if any(p > 0 for p in padding):
            input = _zeros_pad(input, padding)
        return infinicore.ntops.torch.avg_pool3d(input, kernel_size, stride, ceil_mode)

    # cpu infer
    return Tensor(
        _infinicore.avg_pool3d(
            input._underlying, kernel_size, stride, padding, ceil_mode
        )
    )
