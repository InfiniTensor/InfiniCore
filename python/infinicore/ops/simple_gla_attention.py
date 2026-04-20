from infinicore.lib import _infinicore
from infinicore.tensor import Tensor

_native_simple_gla_attention = getattr(_infinicore, "simple_gla_attention", None)
if _native_simple_gla_attention is None:
    _MISSING_MSG = (
        "simple_gla_attention not found in _infinicore. Rebuild InfiniCore extension: "
        "cd InfiniCore && xmake build _infinicore"
    )


def simple_gla_attention(q, k, v, g_gamma, *, scale):
    """Simple GLA (recurrent linear) attention. q, k, v [B, T, H, D], g_gamma [H]. Returns [B, T, H, D]."""
    if _native_simple_gla_attention is None:
        raise NotImplementedError(_MISSING_MSG)
    return Tensor(
        _native_simple_gla_attention(
            q._underlying,
            k._underlying,
            v._underlying,
            g_gamma._underlying,
            float(scale),
        )
    )
