from infinicore.lib import _infinicore
from infinicore.tensor import Tensor


def chunk_gated_delta_rule(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    g: Tensor,
    beta: Tensor,
    initial_state: Tensor,
    *,
    cu_seqlens: Tensor | None = None,
    initial_state_indices: Tensor | None = None,
    final_state_indices: Tensor | None = None,
    use_qk_l2norm: bool = False,
    chunk_size: int = 64,
) -> Tensor:
    """Run chunk gated delta rule and return only ``out``.

    Padded mode shapes:
        q, k: ``[B, T, Hk, Dk]``
        v, out: ``[B, T, Hv, Dv]``
        g, beta: ``[B, T, Hv]``
        initial_state: ``[B, Hv, Dv, Dk]``

    Continuous-batch mode shapes:
        Pass ``cu_seqlens`` with shape ``[B + 1]`` and dtype int32/int64.
        q, k: ``[1, total_tokens, Hk, Dk]``
        v, out: ``[1, total_tokens, Hv, Dv]``
        g, beta: ``[1, total_tokens, Hv]``

    Indexed state-pool mode:
        initial_state is ``[pool_size, Hv, Dv, Dk]``.
        ``initial_state_indices`` and ``final_state_indices`` are both ``[B]``
        int32/int64 tensors. The final state is written in-place into
        ``initial_state[final_state_indices]`` and no final state tensor is
        returned.

    Notes:
        ``Hv`` must be a multiple of ``Hk``. q/k/v/out may be strided in the
        first three dimensions, but the last dimension must be contiguous.
        g and beta may use a different floating dtype from q/k/v/state.
    """
    if (initial_state_indices is None) != (final_state_indices is None):
        raise ValueError(
            "initial_state_indices and final_state_indices must be provided together"
        )

    return Tensor(
        _infinicore.chunk_gated_delta_rule(
            q._underlying,
            k._underlying,
            v._underlying,
            g._underlying,
            beta._underlying,
            initial_state._underlying,
            None if cu_seqlens is None else cu_seqlens._underlying,
            (
                None
                if initial_state_indices is None
                else initial_state_indices._underlying
            ),
            None if final_state_indices is None else final_state_indices._underlying,
            use_qk_l2norm,
            chunk_size,
        )
    )
