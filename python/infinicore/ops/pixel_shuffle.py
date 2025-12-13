import infinicore
from infinicore.lib import _infinicore
from infinicore.tensor import Tensor


def pixel_shuffle(input, upscale_factor, *, out=None):
    """
    Rearranges elements in a tensor of shape (*, C*r^2, H, W) to a tensor of shape (*, C, H*r, W*r).
    
    Args:
        input: Input tensor
        upscale_factor: Factor to increase spatial resolution by
        out: Optional output tensor
    """
    # 1. 优先尝试使用 ntops 加速 (仅限 CUDA/MUSA 设备且无 out 参数时)
    if infinicore.use_ntops and input.device.type in ("cuda", "musa") and out is None:
        return infinicore.ntops.torch.pixel_shuffle(input, upscale_factor)

    # 2. 使用 infinicore C++ 实现（通过 infiniop）
    result = _infinicore.pixel_shuffle(input, upscale_factor)

    # 3. 处理 out 参数
    if out is not None:
        out.copy_(result)
        return out
    
    return result

