from infinicore.lib import _infinicore
from infinicore.tensor import Tensor


def mrope(
    q: Tensor,
    k: Tensor,
    cos: Tensor,
    sin: Tensor,
    positions: Tensor,
    head_size: int,
    rotary_dim: int,
    section_t: int,
    section_h: int,
    section_w: int,
    interleaved: bool,
    *,
    out=None,
) -> tuple[Tensor, Tensor]:
    if out is None:
        q_out, k_out = _infinicore.mrope(
            q._underlying,
            k._underlying,
            cos._underlying,
            sin._underlying,
            positions._underlying,
            head_size,
            rotary_dim,
            section_t,
            section_h,
            section_w,
            interleaved,
        )
        return Tensor(q_out), Tensor(k_out)

    q_out, k_out = out
    _infinicore.mrope_(
        q_out._underlying,
        k_out._underlying,
        q._underlying,
        k._underlying,
        cos._underlying,
        sin._underlying,
        positions._underlying,
        head_size,
        rotary_dim,
        section_t,
        section_h,
        section_w,
        interleaved,
    )
    return q_out, k_out
