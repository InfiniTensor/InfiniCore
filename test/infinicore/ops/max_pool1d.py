import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import infinicore
from framework.base import BaseOperatorTest, TensorSpec, TestCase
from framework.runner import GenericTestRunner

# Test cases format: (in_shape, in_strides_or_None, kernel_size, stride_or_None, padding, dilation, ceil_mode)

_TEST_CASES_DATA = [
    ((2, 3, 16), None, 3, None, 0, 1, False),
    ((1, 4, 15), (60, 15, 1), 5, 1, 2, 1, True),
    ((2, 1, 32), None, 2, 2, 0, 1, False),
    ((3, 2, 7), (14, 7, 1), 3, None, 1, 1, True),
    ((4, 6, 31), None, 4, 2, 1, 1, False),
    ((2, 8, 9), (72, 9, 1), 3, 1, 0, 1, False),
]

_TOLERANCE_MAP = {
    infinicore.float16: {"atol": 0.0, "rtol": 0.0},
    infinicore.float32: {"atol": 0.0, "rtol": 0.0},
}
_TENSOR_DTYPES = [infinicore.float16, infinicore.float32]


def parse_test_cases():
    cases = []
    for in_shape, in_strides, k, s, p, d, ceil_mode in _TEST_CASES_DATA:
        for dtype in _TENSOR_DTYPES:
            tol = _TOLERANCE_MAP[dtype]
            in_spec = TensorSpec.from_tensor(in_shape, in_strides, dtype)
            kwargs = {"kernel_size": k, "dilation": d, "ceil_mode": ceil_mode}
            if s is not None:
                kwargs["stride"] = s
            if p is not None:
                kwargs["padding"] = p
            cases.append(
                TestCase(
                    inputs=[in_spec],
                    kwargs=kwargs,
                    output_spec=None,
                    comparison_target=None,
                    tolerance=tol,
                    description="MaxPool1d - OUT_OF_PLACE",
                )
            )

    return cases


class OpTest(BaseOperatorTest):
    """MaxPool1d operator test with simplified implementation"""

    def __init__(self):
        super().__init__("MaxPool1d")

    def get_test_cases(self):
        return parse_test_cases()

    def torch_operator(self, *args, **kwargs):
        return torch.nn.functional.max_pool1d(*args, **kwargs)

    # def infinicore_operator(self, *args, **kwargs):
    #     """InfiniCore implementation (operator not yet available)."""
    #     return infinicore.nn.functional.max_pool1d(*args, **kwargs)


def main():
    """Main entry point"""
    runner = GenericTestRunner(OpTest)
    runner.run_and_exit()


if __name__ == "__main__":
    main()
