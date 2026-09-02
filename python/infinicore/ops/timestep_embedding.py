from infinicore.lib import _infinicore
from infinicore.tensor import Tensor


def timestep_embedding(timestep, embedding_dim=256, max_period=10000.0, *, out=None):
    max_period = float(max_period)
    if out is None:
        return Tensor(
            _infinicore.timestep_embedding(
                timestep._underlying,
                int(embedding_dim),
                max_period,
            )
        )

    _infinicore.timestep_embedding_(
        out._underlying,
        timestep._underlying,
        max_period,
    )
    return out
