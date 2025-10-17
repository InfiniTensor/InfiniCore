from infinicore.lib import _infinicore
from infinicore.tensor import Tensor


def rms_norm(input, weight, epsilon=1e-5, *, out=None):
    if out is None:
        return Tensor(
            _infinicore.rms_norm(input._underlying, weight._underlying, epsilon)
        )

    _infinicore.rms_norm_(
        out._underlying, input._underlying, weight._underlying, epsilon
    )
