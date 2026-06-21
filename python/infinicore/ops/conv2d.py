from infinicore.lib import _infinicore
from infinicore.tensor import Tensor, zeros


def _pair(value):
    if isinstance(value, int):
        return [value, value]
    return list(value)


def conv2d(
    input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1, *, out=None
):
    if groups != 1:
        raise NotImplementedError(
            "infinicore.ops.conv2d currently supports groups=1 only"
        )

    stride = _pair(stride)
    padding = _pair(padding)
    dilation = _pair(dilation)
    if bias is None:
        bias = zeros((weight.shape[0],), dtype=weight.dtype, device=weight.device)
    bias_raw = bias._underlying

    if out is None:
        return Tensor(
            _infinicore.conv2d(
                input._underlying,
                weight._underlying,
                bias_raw,
                padding,
                stride,
                dilation,
            )
        )

    _infinicore.conv2d_(
        out._underlying,
        input._underlying,
        weight._underlying,
        bias_raw,
        padding,
        stride,
        dilation,
    )
    return out
