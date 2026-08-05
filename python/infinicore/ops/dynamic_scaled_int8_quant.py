from infinicore.dtype import int8
from infinicore.lib import _infinicore
from infinicore.tensor import empty


def dynamic_scaled_int8_quant(input, input_scales, *, out=None):
    """Per-token dynamic scaled int8 quantization.

    Mutates/writes:
      out: int8 tensor with the same shape as input
      input_scales: float32 tensor with numel == input.numel / input.shape[-1]
    """
    if out is None:
        out = empty(input.shape, dtype=int8, device=input.device)
    _infinicore.dynamic_scaled_int8_quant_(
        out._underlying, input._underlying, input_scales._underlying
    )
    return out
