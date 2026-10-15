from infinicore.lib import _infinicore
from infinicore.tensor import Tensor


def kimi_attention_residual(
    values: Tensor,
    norm_weight: Tensor,
    projection_weight: Tensor,
    epsilon: float = 1e-6,
) -> Tensor:
    return Tensor(
        _infinicore.kimi_attention_residual(
            values._underlying,
            norm_weight._underlying,
            projection_weight._underlying,
            epsilon,
        )
    )
