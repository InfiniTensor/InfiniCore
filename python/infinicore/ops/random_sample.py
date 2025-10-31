from infinicore.lib import _infinicore
from infinicore.tensor import Tensor


def random_sample(logits, random_val, topp, topk, temperature, *, out=None):
    if out is None:
        return Tensor(
            _infinicore.random_sample(
                logits._underlying, random_val, topp, topk, temperature
            )
        )

    _infinicore.random_sample_(
        out._underlying,
        logits._underlying,
        random_val,
        topp,
        topk,
        temperature,
    )


