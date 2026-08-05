from infinicore.lib import _infinicore


def moe_expand_input_with_inv_pos_(
    expand_states,
    expand_scales,
    hidden_states,
    inv_pos,
    top_k: int,
    group_size: int = 128,
    format: int = 0,
):
    _infinicore.moe_expand_input_with_inv_pos_(
        expand_states._underlying,
        None if expand_scales is None else expand_scales._underlying,
        hidden_states._underlying,
        inv_pos._underlying,
        top_k,
        group_size,
        format,
    )
    return expand_states if expand_scales is None else (expand_states, expand_scales)
