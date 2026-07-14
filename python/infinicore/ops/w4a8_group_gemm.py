from infinicore.lib import _infinicore


def w4a8_group_gemm_(
    out,
    input,
    weight,
    input_scale,
    weight_scale,
    tokens_per_experts,
    sorted_token_ids=None,
    bias=None,
    trans_weight=True,
    is_decode=False,
):
    _infinicore.w4a8_group_gemm_(
        out._underlying,
        input._underlying,
        weight._underlying,
        input_scale._underlying,
        weight_scale._underlying,
        tokens_per_experts._underlying,
        None if sorted_token_ids is None else sorted_token_ids._underlying,
        None if bias is None else bias._underlying,
        trans_weight,
        is_decode,
    )
    return out
