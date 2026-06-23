from infinicore.lib import _infinicore
from infinicore.tensor import Tensor


def recurrent_gated_delta_rule(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    g: Tensor,
    beta: Tensor,
    initial_state: Tensor,
    *,
    initial_state_indices: Tensor | None = None,
    final_state_indices: Tensor | None = None,
    use_qk_l2norm: bool = False,
) -> Tensor:
    if initial_state_indices is None and final_state_indices is None:
        return Tensor(
            _infinicore.recurrent_gated_delta_rule(
                q._underlying,
                k._underlying,
                v._underlying,
                g._underlying,
                beta._underlying,
                initial_state._underlying,
                use_qk_l2norm,
            )
        )

    if initial_state_indices is None or final_state_indices is None:
        raise ValueError(
            "initial_state_indices and final_state_indices must be provided together"
        )

    return Tensor(
        _infinicore.recurrent_gated_delta_rule_indexed(
            q._underlying,
            k._underlying,
            v._underlying,
            g._underlying,
            beta._underlying,
            initial_state._underlying,
            initial_state_indices._underlying,
            final_state_indices._underlying,
            use_qk_l2norm,
        )
    )
