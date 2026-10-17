import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import infinicore
import torch
from framework import BaseOperatorTest, GenericTestRunner, TestCase, TensorSpec

_TEST_CASES_DATA = [
    (1, 128, 256, True),
    (4, 128, 256, False),
    (16, 4096, 6144, False),
    (16, 12288, 4096, False),
]

_TENSOR_DTYPES = [infinicore.float16, infinicore.bfloat16]
_TOLERANCE_MAP = {
    infinicore.float16: {"atol": 2e-3, "rtol": 2e-3},
    infinicore.bfloat16: {"atol": 2e-2, "rtol": 2e-2},
}


def parse_test_cases():
    cases = []
    for rows, in_features, out_features, has_bias in _TEST_CASES_DATA:
        for dtype in _TENSOR_DTYPES:
            input_spec = TensorSpec.from_tensor((rows, in_features), dtype=dtype)
            weight_spec = TensorSpec.from_tensor(
                (out_features, in_features), dtype=infinicore.int8
            )
            scale_spec = TensorSpec.from_tensor((1, out_features), dtype=dtype)
            bias_spec = (
                TensorSpec.from_tensor((out_features,), dtype=dtype)
                if has_bias
                else None
            )
            cases.append(
                TestCase(
                    inputs=[input_spec, weight_spec, scale_spec, bias_spec],
                    kwargs={},
                    output_spec=None,
                    comparison_target=None,
                    tolerance=_TOLERANCE_MAP[dtype],
                    description=f"LinearW8A16_{str(dtype).split('.')[-1]}_{rows}x{in_features}x{out_features}",
                )
            )
    return cases


class OpTest(BaseOperatorTest):
    def __init__(self):
        super().__init__("LinearW8A16")

    def get_test_cases(self):
        return parse_test_cases()

    def torch_operator(self, input, quantized_weight, scale, bias=None):
        weight = quantized_weight.to(input.dtype) * scale.t()
        return torch.nn.functional.linear(input, weight, bias)

    def infinicore_operator(self, input, quantized_weight, scale, bias=None):
        return infinicore.linear_w8a16(input, quantized_weight, scale, bias)


def main():
    GenericTestRunner(OpTest).run_and_exit()


if __name__ == "__main__":
    main()
