from infinicore.lib import _infinicore
from infinicore.tensor import Tensor


def lightning_attention(q, k, v, slope, initial_state, initial_state_indices, final_state_indices):
    """Indexed-pool lightning attention (MiniMax-01 style).

    Returns out [B, T, H, D]; `initial_state` is updated in place at the
    `final_state_indices` rows.
    """
    return Tensor(
        _infinicore.lightning_attention(
            q._underlying,
            k._underlying,
            v._underlying,
            slope._underlying,
            initial_state._underlying,
            initial_state_indices._underlying,
            final_state_indices._underlying,
        )
    )
