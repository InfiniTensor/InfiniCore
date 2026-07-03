from infinicore.lib import _infinicore
from infinicore.tensor import Tensor


def causal_conv1d(
    qkv: Tensor,
    conv_state: Tensor,
    weight: Tensor,
    bias: Tensor | None = None,
    *,
    cu_seqlens: Tensor | None = None,
    initial_state_indices: Tensor | None = None,
    final_state_indices: Tensor | None = None,
) -> Tensor:
    """Run causal depthwise Conv1d and return only ``out``.

    Padded mode:
        qkv/out: ``[B, T, C]``
        conv_state: ``[B, C, state_len]``

    Continuous-batch mode:
        Pass ``cu_seqlens`` with shape ``[num_requests + 1]``.
        qkv/out: ``[1, total_tokens, C]``

    Indexed state-pool mode:
        conv_state is ``[pool_size, C, state_len]``.
        ``initial_state_indices`` is optional and selects read slots.
        If ``final_state_indices`` is provided, final states are written
        in-place to ``conv_state[final_state_indices]``. Otherwise an internal
        final-state tensor is allocated by the C++ wrapper and discarded.

    Notes:
        The current infiniop backend supports ``K == 4`` only, where ``weight``
        has shape ``[C, 1, K]`` and ``conv_state`` has shape ``[*, C, K - 1]``.
        No activation is applied; call ``silu`` separately when needed.
    """
    return Tensor(
        _infinicore.causal_conv1d(
            qkv._underlying,
            conv_state._underlying,
            weight._underlying,
            None if bias is None else bias._underlying,
            None if cu_seqlens is None else cu_seqlens._underlying,
            (
                None
                if initial_state_indices is None
                else initial_state_indices._underlying
            ),
            None if final_state_indices is None else final_state_indices._underlying,
        )
    )
