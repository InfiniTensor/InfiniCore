import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import infinicore
from framework.base import BaseOperatorTest, TensorSpec, TestCase
from framework.runner import GenericTestRunner

# Test cases format: (in_shape, in_strides_or_None, kernel_size, output_size_or_None, return_indices)
# Note: fractional_max_pool may return values and indices; PyTorch accepts optional random samples. We avoid
# explicit _random_samples and focus on default behavior. Indices (if returned) form a separate output; tests
# here exercise the value-returning path.

_TEST_CASES_DATA = [
    ((2, 3, 15, 15), None, (3, 3), (5, 5), False),
    ((1, 4, 16, 14), (896, 224, 14, 1), (4, 3), (4, 5), False),
    ((2, 2, 17, 19), None, (5, 5), (7, 6), False),
    ((3, 6, 9, 11), None, (2, 2), (4, 5), False),
    ((1, 8, 20, 20), (3200, 400, 20, 1), (3, 3), (6, 6), False),
    ((2, 5, 12, 10), None, (4, 3), (3, 3), False),
]

_TOLERANCE_MAP = {
    infinicore.float16: {"atol": 1e-3, "rtol": 1e-2},
    infinicore.float32: {"atol": 1e-5, "rtol": 1e-4},
}
_TENSOR_DTYPES = [infinicore.float16, infinicore.float32]


def parse_test_cases():
    cases = []
    for in_shape, in_strides, kernel_size, out_size, return_indices in _TEST_CASES_DATA:
        for dtype in _TENSOR_DTYPES:
            tol = _TOLERANCE_MAP[dtype]
            in_spec = TensorSpec.from_tensor(in_shape, in_strides, dtype)
            kwargs = {
                "kernel_size": kernel_size,
                "output_size": out_size,
                "return_indices": return_indices,
            }
            cases.append(
                TestCase(
                    inputs=[in_spec],
                    kwargs=kwargs,
                    output_spec=None,
                    comparison_target=None,
                    tolerance=tol,
                    description="FractionalMaxPool2d - OUT_OF_PLACE",
                )
            )

    return cases


class OpTest(BaseOperatorTest):
    """FractionalMaxPool2d operator test with simplified implementation"""

    def __init__(self):
        super().__init__("FractionalMaxPool2d")

    def get_test_cases(self):
        return parse_test_cases()

    def torch_operator(self, *args, **kwargs):
        return torch.nn.functional.fractional_max_pool2d(*args, **kwargs)

    # def infinicore_operator(self, *args, **kwargs):
    #     """InfiniCore implementation (operator not yet available)."""
    #     return infinicore.nn.functional.fractional_max_pool2d(*args, **kwargs)


def main():
    """Main entry point"""
    runner = GenericTestRunner(OpTest)
    runner.run_and_exit()


if __name__ == "__main__":
    main()
