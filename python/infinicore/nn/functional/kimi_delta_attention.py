from infinicore.lib import _infinicore
from infinicore.tensor import Tensor


def kimi_delta_attention(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    g: Tensor,
    beta: Tensor,
    A_log: Tensor,
    dt_bias: Tensor,
    initial_state: Tensor,
    *,
    cu_seqlens: Tensor | None = None,
    initial_state_indices: Tensor | None = None,
    final_state_indices: Tensor | None = None,
    scale: float = 1.0,
    lower_bound: float = -5.0,
    use_qk_l2norm: bool = True,
) -> Tensor:
    """Run Kimi Delta Attention and return only ``out``.

    Padded mode:
        q/k/v/g/out: ``[B, T, H, D]``
        beta: ``[B, T, H]``
        initial_state: ``[B, H, D, D]``

    Continuous-batch mode:
        Pass ``cu_seqlens`` with shape ``[num_requests + 1]``.
        q/k/v/g/out: ``[1, total_tokens, H, D]``
        beta: ``[1, total_tokens, H]``

    Indexed state-pool mode:
        initial_state is ``[pool_size, H, D, D]``.
        ``initial_state_indices`` and ``final_state_indices`` are both
        ``[num_requests]`` int32/int64 tensors. The final state is written
        in-place to ``initial_state[final_state_indices]``.
    """
    if (initial_state_indices is None) != (final_state_indices is None):
        raise ValueError(
            "initial_state_indices and final_state_indices must be provided together"
        )

    return Tensor(
        _infinicore.kimi_delta_attention(
            q._underlying,
            k._underlying,
            v._underlying,
            g._underlying,
            beta._underlying,
            A_log._underlying,
            dt_bias._underlying,
            initial_state._underlying,
            None if cu_seqlens is None else cu_seqlens._underlying,
            (
                None
                if initial_state_indices is None
                else initial_state_indices._underlying
            ),
            None if final_state_indices is None else final_state_indices._underlying,
            scale,
            lower_bound,
            use_qk_l2norm,
        )
    )
