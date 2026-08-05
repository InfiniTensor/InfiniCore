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
    infinicore.float16: {"atol": 3e-2, "rtol": 3e-2},
    infinicore.bfloat16: {"atol": 6e-2, "rtol": 6e-2},
    infinicore.float32: {"atol": 2e-4, "rtol": 2e-4},
}


def dequantize_mxfp4(packed, scales):
    magnitudes = torch.tensor(
        [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0],
        dtype=torch.float32,
        device=packed.device,
    )
    codes = torch.stack((packed & 0x0F, packed >> 4), dim=-1).flatten(-2)
    values = magnitudes[(codes & 0x07).to(torch.int64)]
    values = torch.where((codes & 0x08) != 0, -values, values)
    exponents = scales.to(torch.int32).sub(127).repeat_interleave(32, dim=-1)
    return torch.ldexp(values, exponents)


def torch_linear_mxfp4(input, packed_weight, weight_scale, bias, alpha):
    weight = dequantize_mxfp4(packed_weight, weight_scale)
    if input.device.type == "cuda":
        old_allow_tf32 = torch.backends.cuda.matmul.allow_tf32
        torch.backends.cuda.matmul.allow_tf32 = False
        try:
            output = torch.matmul(input.float(), weight.transpose(0, 1)) * alpha
        finally:
            torch.backends.cuda.matmul.allow_tf32 = old_allow_tf32
    else:
        output = torch.matmul(input.float(), weight.transpose(0, 1)) * alpha
    if bias is not None:
        output = output + bias.float()
    return output.to(input.dtype)


def make_cases():
    generator = torch.Generator(device="cpu").manual_seed(20260730)
    shapes = [
        ((1, 64), 48, False, 1.0, "decode"),
        ((4, 64), 96, True, 0.75, "small batch with bias"),
        ((2, 5, 128), 64, False, 1.0, "prefill rank-3"),
    ]
    cases = []
    for input_shape, out_features, has_bias, alpha, description in shapes:
        in_features = input_shape[-1]
        input_data = torch.randn(input_shape, generator=generator) * 0.25
        packed = torch.randint(
            0,
            256,
            (out_features, in_features // 2),
            generator=generator,
            dtype=torch.uint8,
        )
        scales = torch.randint(
            123,
            130,
            (out_features, in_features // 32),
            generator=generator,
            dtype=torch.uint8,
        )
        bias_data = (
            torch.randn(out_features, generator=generator) * 0.1 if has_bias else None
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
                    infinicore.uint8,
                    init_mode=TensorInitializer.MANUAL,
                    set_tensor=packed,
                    name="packed_weight",
                ),
                TensorSpec.from_tensor(
                    tuple(scales.shape),
                    None,
                    infinicore.uint8,
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
                    description=f"linear_mxfp4 - {description} - dtype={dtype}",
                )
            )
    return cases


class OpTest(BaseOperatorTest):
    def __init__(self):
        super().__init__("linear_mxfp4")

    def get_test_cases(self):
        return make_cases()

    def torch_operator(self, *args, **kwargs):
        if len(args) == 3:
            args = (*args, None)
        return torch_linear_mxfp4(*args, **kwargs)

    def infinicore_operator(self, *args, **kwargs):
        if len(args) == 3:
            return infinicore.nn.functional.linear_mxfp4(*args, bias=None, **kwargs)
        return infinicore.nn.functional.linear_mxfp4(*args, **kwargs)


if __name__ == "__main__":
    GenericTestRunner(OpTest).run_and_exit()
