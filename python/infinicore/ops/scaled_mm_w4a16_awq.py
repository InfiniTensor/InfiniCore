from infinicore.lib import _infinicore
from infinicore.tensor import Tensor


def scaled_mm_w4a16_awq(input, qweight, qzeros, scales, bias=None, *, out=None):
    if out is None:
        return Tensor(
            _infinicore.scaled_mm_w4a16_awq(
                input._underlying,
                qweight._underlying,
                qzeros._underlying,
                scales._underlying,
                None if bias is None else bias._underlying,
            )
        )
    _infinicore.scaled_mm_w4a16_awq_(
        out._underlying,
        input._underlying,
        qweight._underlying,
        qzeros._underlying,
        scales._underlying,
        None if bias is None else bias._underlying,
    )
    return out
