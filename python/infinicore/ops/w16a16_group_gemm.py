from infinicore.lib import _infinicore


def w16a16_group_gemm_(
    out,
    input,
    weight,
    tokens_per_experts,
    sorted_token_ids=None,
    bias=None,
    trans_weight=True,
    is_decode=False,
):
    if not trans_weight:
        raise RuntimeError(
            "w16a16_group_gemm currently supports only trans_weight=True (TN layout)"
        )
    _infinicore.w16a16_group_gemm_(
        out._underlying,
        input._underlying,
        weight._underlying,
        tokens_per_experts._underlying,
        None if sorted_token_ids is None else sorted_token_ids._underlying,
        None if bias is None else bias._underlying,
        trans_weight,
        is_decode,
    )
    return out
