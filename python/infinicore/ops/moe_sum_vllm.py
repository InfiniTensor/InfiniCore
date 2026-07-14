from infinicore.lib import _infinicore


def moe_sum_vllm_(
    output,
    input,
    topk_weights=None,
    extra_residual=None,
    routed_scale: float = 1.0,
    residual_scale: float = 1.0,
):
    _infinicore.moe_sum_vllm_(
        output._underlying,
        input._underlying,
        None if topk_weights is None else topk_weights._underlying,
        None if extra_residual is None else extra_residual._underlying,
        routed_scale,
        residual_scale,
    )
    return output
