from infinicore.lib import _infinicore
from infinicore.tensor import Tensor


def block_fp8_linear(input, weight, weight_scale):
    """Block-FP8 linear: BF16 input x F8 weight + block scale -> BF16 output.

    Args:
        input: BF16 tensor [M, K]
        weight: F8 tensor [N, K]
        weight_scale: F32 tensor [ceil(N/128), ceil(K/128)]

    Returns:
        BF16 tensor [M, N]
    """
    return Tensor(_infinicore.block_fp8_linear(
        input._underlying, weight._underlying, weight_scale._underlying))


def block_fp8_linear_(output, input, weight, weight_scale):
    """In-place block-FP8 linear.

    Args:
        output: pre-allocated BF16 tensor [M, N]
        input: BF16 tensor [M, K]
        weight: F8 tensor [N, K]
        weight_scale: F32 tensor [ceil(N/128), ceil(K/128)]
    """
    _infinicore.block_fp8_linear_(
        output._underlying, input._underlying,
        weight._underlying, weight_scale._underlying)
