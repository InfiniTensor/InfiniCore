import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import infinicore
from framework.base import BaseOperatorTest, TensorSpec, TestCase
from framework.runner import GenericTestRunner

# Test cases format: (in1_shape, in2_shape, weight_shape, in1_strides_or_None, in2_strides_or_None, weight_strides_or_None, bias_present_bool)

_TEST_CASES_DATA = [
    ((4, 3), (4, 5), (2, 3, 5), None, None, None, True),
    ((1, 6), (1, 7), (3, 6, 7), None, None, None, True),
    ((8, 2), (8, 4), (5, 2, 4), (16, 2), None, None, False),
    ((2, 3), (2, 3), (4, 3, 3), None, (0, 3), None, True),
    ((6, 10), (6, 12), (7, 10, 12), None, None, (840, 70, 1), False),
    ((3, 1), (3, 1), (2, 1, 1), None, None, None, True),
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
        in1_shape,
        in2_shape,
        weight_shape,
        in1_strides,
        in2_strides,
        weight_strides,
        bias_present,
    ) in _TEST_CASES_DATA:
        for dtype in _TENSOR_DTYPES:
            tol = _TOLERANCE_MAP.get(dtype, {"atol": 1e-5, "rtol": 1e-4})
            in1 = TensorSpec.from_tensor(in1_shape, in1_strides, dtype)
            in2 = TensorSpec.from_tensor(in2_shape, in2_strides, dtype)
            weight = TensorSpec.from_tensor(weight_shape, weight_strides, dtype)

            kwargs = {}

            test_cases.append(
                TestCase(
                    inputs=[in1, in2, weight],
                    kwargs=kwargs,
                    output_spec=None,
                    comparison_target=None,
                    tolerance=tol,
                    description="bilinear - OUT_OF_PLACE",
                )
            )

    return test_cases


class OpTest(BaseOperatorTest):
    """bilinear operator test with simplified implementation"""

    def __init__(self):
        super().__init__("bilinear")

    def get_test_cases(self):
        return parse_test_cases()

    def torch_operator(self, *args, **kwargs):
        return torch.nn.functional.bilinear(*args, **kwargs)

    # def infinicore_operator(self, *args, **kwargs):
    #     """InfiniCore implementation (operator not yet available)."""
    #     return infinicore.nn.functional.bilinear(*args, **kwargs)


def main():
    """Main entry point"""
    runner = GenericTestRunner(OpTest)
    runner.run_and_exit()


if __name__ == "__main__":
    main()
