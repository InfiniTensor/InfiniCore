import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import infinicore
from framework.base import BaseOperatorTest, TensorSpec, TestCase
from framework.runner import GenericTestRunner

# Test cases format: (input_shape, weight_shape, input_strides_or_None, weight_strides_or_None, bias_present_bool)
# infinicore.nn.functional.linear(input, weight, bias=None)

_TEST_CASES_DATA = [
    ((4, 3), (2, 3), None, None, True),
    ((1, 6), (3, 6), None, None, False),
    ((8, 10), (5, 10), (80, 10), None, True),
    ((2, 4), (4, 4), None, (16, 4), True),
    ((16, 8), (8, 8), None, None, False),
    ((3, 1), (2, 1), None, None, True),
]

_TOLERANCE_MAP = {
    infinicore.float16: {"atol": 1e-3, "rtol": 1e-2},
    infinicore.float32: {"atol": 1e-5, "rtol": 1e-4},
    infinicore.bfloat16: {"atol": 1e-2, "rtol": 5e-2},
}

_TENSOR_DTYPES = [infinicore.float16, infinicore.bfloat16, infinicore.float32]


def parse_test_cases():
    test_cases = []
    for (
        input_shape,
        weight_shape,
        in_strides,
        w_strides,
        bias_present,
    ) in _TEST_CASES_DATA:
        for dtype in _TENSOR_DTYPES:
            tol = _TOLERANCE_MAP.get(dtype, {"atol": 1e-5, "rtol": 1e-4})
            inp = TensorSpec.from_tensor(input_shape, in_strides, dtype)
            weight = TensorSpec.from_tensor(weight_shape, w_strides, dtype)

            inputs = [inp, weight]
            if bias_present:
                bias_spec = TensorSpec.from_tensor((weight_shape[0],), None, dtype)
                inputs.append(bias_spec)

            test_cases.append(
                TestCase(
                    inputs=inputs,
                    kwargs={},
                    output_spec=None,
                    comparison_target=None,
                    tolerance=tol,
                    description="linear - OUT_OF_PLACE",
                )
            )

    return test_cases


class OpTest(BaseOperatorTest):
    """linear operator test with simplified implementation"""

    def __init__(self):
        super().__init__("linear")

    def get_test_cases(self):
        return parse_test_cases()

    def torch_operator(self, *args, **kwargs):
        return torch.nn.functional.linear(*args, **kwargs)

    # def infinicore_operator(self, *args, **kwargs):
    #     """InfiniCore implementation (operator not yet available)."""
    #     return infinicore.nn.functional.linear(*args, **kwargs)


def main():
    """Main entry point"""
    runner = GenericTestRunner(OpTest)
    runner.run_and_exit()


if __name__ == "__main__":
    main()
