import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
from framework import (
    BaseOperatorTest,
    GenericTestRunner,
    TensorInitializer,
    TensorSpec,
    TestCase,
)

import infinicore

_DTYPES = [infinicore.float16, infinicore.bfloat16, infinicore.float32]
_TOLERANCE = {
    infinicore.float16: {"atol": 8e-2, "rtol": 8e-2},
    infinicore.bfloat16: {"atol": 1.5e-1, "rtol": 1.5e-1},
    infinicore.float32: {"atol": 2e-4, "rtol": 2e-4},
}


def unpack_int4(packed):
    values = packed.to(torch.uint8)
    unpacked = torch.stack((values >> 4, values & 0x0F), dim=-1).flatten(-2)
    unpacked = unpacked.to(torch.int8)
    return torch.where(unpacked >= 8, unpacked - 16, unpacked)


def quantize_per_token(input):
    rows = input.float().reshape(-1, input.shape[-1])
    scale = rows.abs().amax(dim=-1, keepdim=True).clamp_min(1e-8) / 127.0
    quantized = torch.round(rows / scale).clamp(-127, 127).to(torch.int8)
    return quantized, scale


def torch_linear_w4a8(input, packed_weight, weight_scale, bias, alpha):
    input_q, input_scale = quantize_per_token(input)
    weight = unpack_int4(packed_weight).float()
    result = torch.matmul(input_q.float(), weight.t())
    result *= input_scale * weight_scale.float().t()
    result *= alpha
    if bias is not None:
        result += bias.float()
    return result.reshape(*input.shape[:-1], weight.shape[0]).to(input.dtype)


def make_cases():
    generator = torch.Generator(device="cpu").manual_seed(20260911)
    configs = [
        ((1, 64), 48, False, 1.0, "decode"),
        ((5, 64), 96, True, 0.75, "prefill"),
        ((2, 3, 128), 64, False, 1.0, "rank-3"),
    ]
    cases = []
    for input_shape, out_features, has_bias, alpha, description in configs:
        input_data = torch.randn(input_shape, generator=generator) * 0.25
        packed = torch.randint(
            -128,
            128,
            (out_features, input_shape[-1] // 2),
            generator=generator,
            dtype=torch.int8,
        )
        scales = torch.rand((out_features, 1), generator=generator) * 0.1
        bias_data = (
            torch.randn(out_features, generator=generator) * 0.1
            if has_bias
            else None
        )
        for dtype in _DTYPES:
            inputs = [
                TensorSpec.from_tensor(
                    input_shape,
                    None,
                    dtype,
                    init_mode=TensorInitializer.MANUAL,
                    set_tensor=input_data,
                    name="input",
                ),
                TensorSpec.from_tensor(
                    tuple(packed.shape),
                    None,
                    infinicore.int8,
                    init_mode=TensorInitializer.MANUAL,
                    set_tensor=packed,
                    name="packed_weight",
                ),
                TensorSpec.from_tensor(
                    tuple(scales.shape),
                    None,
                    infinicore.float32,
                    init_mode=TensorInitializer.MANUAL,
                    set_tensor=scales,
                    name="weight_scale",
                ),
            ]
            if bias_data is not None:
                inputs.append(
                    TensorSpec.from_tensor(
                        tuple(bias_data.shape),
                        None,
                        dtype,
                        init_mode=TensorInitializer.MANUAL,
                        set_tensor=bias_data,
                        name="bias",
                    )
                )
            cases.append(
                TestCase(
                    inputs=inputs,
                    kwargs={"alpha": alpha},
                    output_spec=None,
                    comparison_target=None,
                    tolerance=_TOLERANCE[dtype],
                    description=f"linear_w4a8 - {description} - dtype={dtype}",
                )
            )
    return cases


class OpTest(BaseOperatorTest):
    def __init__(self):
        super().__init__("linear_w4a8")

    def get_test_cases(self):
        return make_cases()

    def torch_operator(self, *args, **kwargs):
        if len(args) == 3:
            args = (*args, None)
        return torch_linear_w4a8(*args, **kwargs)

    def infinicore_operator(self, *args, **kwargs):
        if len(args) == 3:
            return infinicore.nn.functional.linear_w4a8(
                *args, bias=None, **kwargs
            )
        return infinicore.nn.functional.linear_w4a8(*args, **kwargs)


if __name__ == "__main__":
    GenericTestRunner(OpTest).run_and_exit()
