import infinicore
from infinicore.lib import _infinicore
from infinicore.tensor import Tensor


def pixel_unshuffle(input: Tensor, downscale_factor: int, *, out=None) -> Tensor:
    r"""Rearrange elements in a tensor of shape (*, C, H * r, W * r)
    to a tensor of shape (*, C * r * r, H, W), where r is downscale_factor.
    """

    assert isinstance(downscale_factor, int), "`downscale_factor` must be int."
    assert downscale_factor > 0, "`downscale_factor` must be positive."
    assert input.ndim == 4, "`pixel_unshuffle` only supports 4D NCHW input."

    n, c, h, w = input.shape
    r = downscale_factor

    assert h % r == 0, "input height must be divisible by downscale_factor."
    assert w % r == 0, "input width must be divisible by downscale_factor."

    if infinicore.use_ntops and input.device.type in ("cuda", "musa") and out is None:
        return infinicore.ntops.torch.pixel_unshuffle(input, downscale_factor)

    if out is None:
        return Tensor(_infinicore.pixel_unshuffle(input._underlying, downscale_factor))

    _infinicore.pixel_unshuffle_(
        out._underlying,
        input._underlying,
        downscale_factor,
    )

    return out