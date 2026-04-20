from infinicore.lib import _infinicore
from infinicore.tensor import Tensor

_native_simple_gla_prefill = getattr(_infinicore, "simple_gla_prefill", None)
if _native_simple_gla_prefill is None:
    _MISSING_MSG = (
        "simple_gla_prefill not found in _infinicore. Rebuild InfiniCore extension: "
        "cd InfiniCore && xmake build _infinicore"
    )


def simple_gla_prefill(q, k, v, g_gamma, *, scale):
    """Simple GLA prefill fused kernel. q, k, v [B, T, H, D], g_gamma [H] (F32). Returns [B, T, H, D]."""
    if _native_simple_gla_prefill is None:
        raise NotImplementedError(_MISSING_MSG)
    return Tensor(
        _native_simple_gla_prefill(
            q._underlying,
            k._underlying,
            v._underlying,
            g_gamma._underlying,
            float(scale),
        )
    )
