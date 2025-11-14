import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import infinicore
from framework.base import BaseOperatorTest, TensorSpec, TestCase
from framework.runner import GenericTestRunner
from framework.utils import is_broadcast

# Test cases format: (shape, input_strides_or_None, out_strides_or_None)
# isneginf checks for -inf values

_TEST_CASES_DATA = [
    ((8, 8), None, None),
    ((8, 8), (16, 1), None),
    ((8, 8), None, (0, 1)),
    ((2, 3, 4), None, None),
    ((1, 8), None, None),
    ((16, 128), (256, 1), (256, 1)),
]

_TOLERANCE_MAP = {
    infinicore.float16: {"atol": 0, "rtol": 1e-2},
    infinicore.float32: {"atol": 0, "rtol": 1e-3},
    infinicore.bfloat16: {"atol": 0, "rtol": 5e-2},
}

_TENSOR_DTYPES = [infinicore.float16, infinicore.float32, infinicore.bfloat16]


def parse_test_cases():
    test_cases = []
    for data in _TEST_CASES_DATA:
        shape, in_strides, out_strides = data[0], data[1], data[2]

        input_supports_inplace = not is_broadcast(in_strides)
        out_supports_inplace = not is_broadcast(out_strides)

        for dtype in _TENSOR_DTYPES:
            tol = _TOLERANCE_MAP.get(dtype, {"atol": 0, "rtol": 1e-3})
            input_spec = TensorSpec.from_tensor(shape, in_strides, dtype)
            out_spec = TensorSpec.from_tensor(shape, out_strides, infinicore.bool)

            test_cases.append(
                TestCase(
                    inputs=[input_spec],
                    kwargs={},
                    output_spec=None,
                    comparison_target=None,
                    tolerance=tol,
                    description="IsNegInf - OUT_OF_PLACE",
                )
            )

            if out_supports_inplace:
                test_cases.append(
                    TestCase(
                        inputs=[input_spec],
                        kwargs=None,
                        output_spec=out_spec,
                        comparison_target="out",
                        tolerance=tol,
                        description="IsNegInf - INPLACE(out)",
                    )
                )

    return test_cases


class OpTest(BaseOperatorTest):
    """IsNegInf operator test with simplified implementation"""

    def __init__(self):
        super().__init__("IsNegInf")

    def get_test_cases(self):
        return parse_test_cases()

    def torch_operator(self, *args, **kwargs):
        return torch.isneginf(*args, **kwargs)

    # def infinicore_operator(self, *args, **kwargs):
    #     """InfiniCore implementation (operator not yet available)."""
    #     return infinicore.isneginf(*args, **kwargs)


def main():
    """Main entry point"""
    runner = GenericTestRunner(OpTest)
    runner.run_and_exit()


if __name__ == "__main__":
    main()
