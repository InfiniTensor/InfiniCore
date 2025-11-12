import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import infinicore
from framework.base import BaseOperatorTest, TensorSpec, TestCase
from framework.runner import GenericTestRunner
from framework.utils import is_broadcast

# Test cases format: (in_shape, in_strides_or_None, dim_or_None)
# count_nonzero counts number of non-zero elements along dims or overall

_TEST_CASES_DATA = [
    ((8, 8), None, None),
    ((8, 8), (16, 1), 1),
    ((2, 3, 4), None, 0),
    ((1, 8), None, (0,)),
    ((16, 64), (128, 1), None),
    ((4, 5, 6), (60, 12, 2), 2),
]

_TOLERANCE_MAP = {infinicore.int64: {"atol": 0, "rtol": 0}}

_TENSOR_DTYPES = [infinicore.int32, infinicore.float32, infinicore.uint8]


def parse_test_cases():
    test_cases = []
    for data in _TEST_CASES_DATA:
        shape, strides, dim = data

        for dtype in _TENSOR_DTYPES:
            tol = _TOLERANCE_MAP.get(infinicore.int64, {"atol": 0, "rtol": 0})
            in_spec = TensorSpec.from_tensor(shape, strides, dtype)

            kwargs = {}
            if dim is not None:
                kwargs["dim"] = dim

            test_cases.append(
                TestCase(
                    inputs=[in_spec],
                    kwargs=kwargs,
                    output_spec=None,
                    comparison_target=None,
                    tolerance=tol,
                    description="CountNonZero - OUT_OF_PLACE",
                )
            )

    return test_cases


class OpTest(BaseOperatorTest):
    """CountNonZero operator test with simplified implementation"""

    def __init__(self):
        super().__init__("CountNonZero")

    def get_test_cases(self):
        return parse_test_cases()

    def torch_operator(self, *args, **kwargs):
        return torch.count_nonzero(*args, **kwargs)

    # def infinicore_operator(self, *args, **kwargs):
    #     """InfiniCore implementation (operator not yet available)."""
    #     return infinicore.count_nonzero(*args, **kwargs)


def main():
    """Main entry point"""
    runner = GenericTestRunner(OpTest)
    runner.run_and_exit()


if __name__ == "__main__":
    main()
