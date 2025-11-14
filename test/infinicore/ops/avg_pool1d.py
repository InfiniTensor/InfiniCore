import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import infinicore
from framework.base import BaseOperatorTest, TensorSpec, TestCase
from framework.runner import GenericTestRunner
from framework.utils import is_broadcast

# Test cases format: (in_shape, in_strides_or_None, kernel_size, stride_or_None, padding)

_TEST_CASES_DATA = [
    ((2, 3, 16), None, 3, None, 0),
    ((1, 4, 15), (60, 15, 1), 5, 1, 2),
    ((2, 1, 32), None, 2, 2, 0),
    ((3, 2, 7), (14, 7, 1), 3, None, 1),
    ((4, 6, 31), None, 4, 2, 1),
    ((2, 8, 9), (72, 9, 1), 3, 1, 0),
]

_TOLERANCE_MAP = {
    infinicore.float16: {"atol": 1e-3, "rtol": 1e-2},
    infinicore.float32: {"atol": 1e-5, "rtol": 1e-4},
}

_TENSOR_DTYPES = [infinicore.float16, infinicore.float32]


def parse_test_cases():
    tests = []
    for shape, strides, k, s, p in _TEST_CASES_DATA:
        for dtype in _TENSOR_DTYPES:
            tol = _TOLERANCE_MAP.get(dtype, {"atol": 1e-5, "rtol": 1e-3})
            in_spec = TensorSpec.from_tensor(shape, strides, dtype)

            kwargs = {}
            kwargs["kernel_size"] = k
            if s is not None:
                kwargs["stride"] = s
            if p is not None:
                kwargs["padding"] = p

            tests.append(
                TestCase(
                    inputs=[in_spec],
                    kwargs=kwargs,
                    output_spec=None,
                    comparison_target=None,
                    tolerance=tol,
                    description="AvgPool1d - OUT_OF_PLACE",
                )
            )

            # In-place pooling isn't supported; only out-of-place expected

    return tests


class OpTest(BaseOperatorTest):
    """AvgPool1d operator test with simplified implementation"""

    def __init__(self):
        super().__init__("AvgPool1d")

    def get_test_cases(self):
        return parse_test_cases()

    def torch_operator(self, *args, **kwargs):
        return torch.nn.functional.avg_pool1d(*args, **kwargs)

    # def infinicore_operator(self, *args, **kwargs):
    #     """InfiniCore implementation (operator not yet available)."""
    #     return infinicore.nn.functional.avg_pool1d(*args, **kwargs)


def main():
    """Main entry point"""
    runner = GenericTestRunner(OpTest)
    runner.run_and_exit()


if __name__ == "__main__":
    main()
