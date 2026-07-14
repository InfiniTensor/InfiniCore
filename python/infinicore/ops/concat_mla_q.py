from infinicore.lib import _infinicore
from infinicore.tensor import empty


def concat_mla_q(ql_nope, q_pe, *, out=None):
    """Concatenate MLA ql_nope and q_pe along the last dimension into out."""
    if out is None:
        shape = list(ql_nope.shape)
        shape[-1] = ql_nope.shape[-1] + q_pe.shape[-1]
        out = empty(tuple(shape), dtype=ql_nope.dtype, device=ql_nope.device)
    _infinicore.concat_mla_q_(ql_nope._underlying, q_pe._underlying, out._underlying)
    return out
