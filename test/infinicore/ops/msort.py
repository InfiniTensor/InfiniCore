import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import infinicore
from framework.base import BaseOperatorTest, TensorSpec, TestCase
from framework.runner import GenericTestRunner
from framework.utils import is_broadcast

# Test cases format: (shape, input_strides)
_TEST_CASES_DATA = [
    ((6, 8), None),
    ((8, 4), (16, 1)),
    ((5, 5), None),
    ((3, 7), (14, 1)),
    ((10, 3), None),
    ((2, 16), (32, 1)),
]

_TOLERANCE_MAP = {
    infinicore.float16: {"atol": 0, "rtol": 1e-3},
    infinicore.float32: {"atol": 0, "rtol": 1e-5},
    infinicore.bfloat16: {"atol": 0, "rtol": 1e-2},
}

_TENSOR_DTYPES = [infinicore.float16, infinicore.bfloat16, infinicore.float32]


def parse_test_cases():
    test_cases = []
    for data in _TEST_CASES_DATA:
        shape, in_strides = data

        out_supports_inplace = not is_broadcast(in_strides)

        for dtype in _TENSOR_DTYPES:
            tol = _TOLERANCE_MAP.get(dtype, {"atol": 0, "rtol": 1e-5})

            input_spec = TensorSpec.from_tensor(shape, in_strides, dtype)

            # Out-of-place
            test_cases.append(
                TestCase(
                    inputs=[input_spec],
                    kwargs={},
                    output_spec=None,
                    comparison_target=None,
                    tolerance=tol,
                    description="msort - OUT_OF_PLACE",
                )
            )

            if out_supports_inplace:
                out_spec = TensorSpec.from_tensor(shape, None, dtype)
                test_cases.append(
                    TestCase(
                        inputs=[input_spec],
                        kwargs=None,
                        output_spec=out_spec,
                        comparison_target="out",
                        tolerance=tol,
                        description="msort - INPLACE(out)",
                    )
                )

    return test_cases


class OpTest(BaseOperatorTest):
    """MSort operator test with simplified implementation"""

    def __init__(self):
        super().__init__("MSort")

    def get_test_cases(self):
        return parse_test_cases()

    def torch_operator(self, *args, **kwargs):
        return torch.msort(*args, **kwargs)

    # def infinicore_operator(self, *args, **kwargs):
    #     """InfiniCore implementation (operator not yet available)."""
    #     return infinicore.msort(*args, **kwargs)


def main():
    """Main entry point"""
    runner = GenericTestRunner(OpTest)
    runner.run_and_exit()


if __name__ == "__main__":
    main()
