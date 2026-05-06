from infinicore.lib import _infinicore
from infinicore.tensor import Tensor


def rotmg(d1: Tensor, d2: Tensor, x1: Tensor, y1: Tensor, param: Tensor):
    _infinicore.rotmg_(
        d1._underlying,
        d2._underlying,
        x1._underlying,
        y1._underlying,
        param._underlying,
    )
    return d1, d2, x1, param
