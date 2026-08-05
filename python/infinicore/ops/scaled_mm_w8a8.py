from infinicore.lib import _infinicore
from infinicore.tensor import Tensor


def scaled_mm_w8a8(a, b, a_scales, b_scales, bias=None, trans_weight=True, *, out=None):
    args = (
        a._underlying,
        b._underlying,
        a_scales._underlying,
        b_scales._underlying,
        None if bias is None else bias._underlying,
        trans_weight,
    )
    if out is None:
        return Tensor(_infinicore.scaled_mm_w8a8(*args))
    _infinicore.scaled_mm_w8a8_(out._underlying, *args)
    return out
