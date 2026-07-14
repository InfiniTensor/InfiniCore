from infinicore.lib import _infinicore


def scaled_mm_w4a8(
    a, b, a_scales, b_scales, bias=None, trans_weight=False, *, out=None
):
    if out is None:
        return _infinicore.scaled_mm_w4a8(
            a._underlying,
            b._underlying,
            a_scales._underlying,
            b_scales._underlying,
            None if bias is None else bias._underlying,
            trans_weight,
        )
    _infinicore.scaled_mm_w4a8_(
        out._underlying,
        a._underlying,
        b._underlying,
        a_scales._underlying,
        b_scales._underlying,
        None if bias is None else bias._underlying,
        trans_weight,
    )
    return out
