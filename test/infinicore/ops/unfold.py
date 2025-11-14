import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import infinicore
from framework.base import BaseOperatorTest, TensorSpec, TestCase
from framework.runner import GenericTestRunner

# Test cases format: (in_shape, in_strides_or_None, kernel_size, dilation, padding, stride)
# Unfold extracts sliding local blocks from a batched input tensor.

_TEST_CASES_DATA = [
    ((2, 3, 8, 8), None, (3, 3), 1, 0, (1, 1)),
    ((1, 4, 10, 12), None, (5, 3), 1, 1, (2, 1)),
    ((2, 2, 16, 16), (512, 256, 16, 1), (4, 4), 1, 0, (4, 4)),
    ((3, 6, 7, 9), None, (3, 2), 1, 0, (1, 1)),
    ((1, 8, 9, 11), None, (2, 3), 1, 1, (1, 2)),
    ((2, 5, 12, 6), (360, 72, 6, 1), (3, 3), 1, 0, (2, 1)),
]

_TOLERANCE_MAP = {
    infinicore.float16: {"atol": 1e-3, "rtol": 1e-2},
    infinicore.float32: {"atol": 1e-5, "rtol": 1e-4},
}

_TENSOR_DTYPES = [infinicore.float16, infinicore.float32]


def parse_test_cases():
    cases = []
    for (
        in_shape,
        in_strides,
        kernel_size,
        dilation,
        padding,
        stride,
    ) in _TEST_CASES_DATA:
        for dtype in _TENSOR_DTYPES:
            tol = _TOLERANCE_MAP.get(dtype)
            in_spec = TensorSpec.from_tensor(in_shape, in_strides, dtype)
            kwargs = {
                "kernel_size": kernel_size,
                "dilation": dilation,
                "padding": padding,
                "stride": stride,
            }
            cases.append(
                TestCase(
                    inputs=[in_spec],
                    kwargs=kwargs,
                    output_spec=None,
                    comparison_target=None,
                    tolerance=tol,
                    description="Unfold - OUT_OF_PLACE",
                )
            )

    return cases


class OpTest(BaseOperatorTest):
    """Unfold operator test with simplified implementation"""

    def __init__(self):
        super().__init__("Unfold")

    def get_test_cases(self):
        return parse_test_cases()

    def torch_operator(self, *args, **kwargs):
        return torch.nn.functional.unfold(*args, **kwargs)

    # def infinicore_operator(self, *args, **kwargs):
    #     """InfiniCore implementation (operator not yet available)."""
    #     return infinicore.nn.functional.unfold(*args, **kwargs)


def main():
    """Main entry point"""
    runner = GenericTestRunner(OpTest)
    runner.run_and_exit()


if __name__ == "__main__":
    main()
