from infinicore.lib import _infinicore


def prepare_glm_w4a16_awq_(qweight, qzeros, scales, checkpoint_weight, channel_scales):
    _infinicore.prepare_glm_w4a16_awq_(
        qweight._underlying,
        qzeros._underlying,
        scales._underlying,
        checkpoint_weight._underlying,
        channel_scales._underlying,
    )
