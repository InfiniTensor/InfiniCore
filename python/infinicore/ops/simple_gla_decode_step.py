from infinicore.lib import _infinicore
from infinicore.tensor import Tensor

_native = getattr(_infinicore, "simple_gla_decode_step", None)
if _native is None:
    _MISSING_MSG = (
        "simple_gla_decode_step not found in _infinicore. Rebuild InfiniCore extension: "
        "cd InfiniCore && xmake build _infinicore"
    )


def simple_gla_decode_step(q, k, v, state, g_gamma, *, scale):
    """One Simple GLA decode step.

    q, k, v: [B, 1, H, D]. state: [B, H, D, D] float32, updated in-place (must be contiguous).
    g_gamma: [H]. Returns output [B, 1, H, D].
    """
    if _native is None:
        raise NotImplementedError(_MISSING_MSG)
    return Tensor(
        _native(
            q._underlying,
            k._underlying,
            v._underlying,
            state._underlying,
            g_gamma._underlying,
            float(scale),
        )
    )
