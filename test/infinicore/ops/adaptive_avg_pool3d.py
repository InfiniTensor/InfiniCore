import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import infinicore
from framework.base import BaseOperatorTest, TensorSpec, TestCase
from framework.runner import GenericTestRunner
from framework.utils import is_broadcast

# Test cases format: (in_shape, in_strides_or_None, output_size_or_None)
# adaptive_avg_pool3d maps input D x H x W to target output size (d, h, w)

_TEST_CASES_DATA = [
    ((2, 3, 8, 8, 8), None, (1, 1, 1)),
    ((2, 4, 7, 9, 6), (2016, 504, 72, 8, 1), (3, 3, 2)),
    ((1, 8, 16, 16, 16), None, (4, 4, 4)),
    ((2, 2, 5, 7, 9), (1260, 630, 126, 18, 2), (2, 3, 4)),
    ((3, 3, 10, 9, 8), None, (5, 5, 4)),
    ((2, 6, 9, 11, 13), (5148, 858, 13, 1, 1), (3, 4, 5)),
]

_TOLERANCE_MAP = {
    infinicore.float16: {"atol": 1e-3, "rtol": 1e-2},
    infinicore.float32: {"atol": 1e-5, "rtol": 1e-4},
}

_TENSOR_DTYPES = [infinicore.float16, infinicore.float32]


def parse_test_cases():
    test_cases = []
    for data in _TEST_CASES_DATA:
        shape, strides, out_size = data
        for dtype in _TENSOR_DTYPES:
            tol = _TOLERANCE_MAP.get(dtype, {"atol": 1e-5, "rtol": 1e-3})
            in_spec = TensorSpec.from_tensor(shape, strides, dtype)

            kwargs = {"output_size": out_size}
            test_cases.append(
                TestCase(
                    inputs=[in_spec],
                    kwargs=kwargs,
                    output_spec=None,
                    comparison_target=None,
                    tolerance=tol,
                    description="AdaptiveAvgPool3d - OUT_OF_PLACE",
                )
            )

    return test_cases


class OpTest(BaseOperatorTest):
    """AdaptiveAvgPool3d operator test with simplified implementation"""

    def __init__(self):
        super().__init__("AdaptiveAvgPool3d")

    def get_test_cases(self):
        return parse_test_cases()

    def torch_operator(self, *args, **kwargs):
        return torch.nn.functional.adaptive_avg_pool3d(*args, **kwargs)

    # def infinicore_operator(self, *args, **kwargs):
    #     """InfiniCore implementation (operator not yet available)."""
    #     return infinicore.nn.functional.adaptive_avg_pool3d(*args, **kwargs)


def main():
    """Main entry point"""
    runner = GenericTestRunner(OpTest)
    runner.run_and_exit()


if __name__ == "__main__":
    main()
