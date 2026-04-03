from infinicore.lib import _infinicore
from infinicore.tensor import Tensor

_native_gla_attention = getattr(_infinicore, "gla_attention", None)
if _native_gla_attention is None:
    _MISSING_MSG = (
        "gla_attention not found in _infinicore. Rebuild InfiniCore extension: "
        "cd InfiniCore && xmake build _infinicore"
    )


def gla_attention(q, k_total, v_total, scale, *, causal=True):
    """GLA-style attention. q, k_total, v_total are [B, n_q/n_kv, S, D]. Returns [B, n_q, S_q, D]."""
    if _native_gla_attention is None:
        raise NotImplementedError(_MISSING_MSG)
    return Tensor(
        _native_gla_attention(
            q._underlying,
            k_total._underlying,
            v_total._underlying,
            float(scale),
            causal,
        )
    )
