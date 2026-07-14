from infinicore.lib import _infinicore


def moe_silu_and_mul_quant_(output, output_scale, input, format: int = 0):
    _infinicore.moe_silu_and_mul_quant_(
        output._underlying,
        None if output_scale is None else output_scale._underlying,
        input._underlying,
        format,
    )
    return output if output_scale is None else (output, output_scale)
