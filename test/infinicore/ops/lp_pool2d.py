import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import infinicore
from framework.base import BaseOperatorTest, TensorSpec, TestCase
from framework.runner import GenericTestRunner

# Test cases format: (in_shape, in_strides_or_None, norm_type, kernel_size, stride_or_None, ceil_mode)

_TEST_CASES_DATA = [
    ((2, 3, 16, 16), None, 2.0, (3, 3), None, False),
    ((1, 4, 15, 17), (1020, 255, 17, 1), 1.0, (5, 4), (2, 2), True),
    ((2, 1, 32, 32), None, 3.0, (2, 2), (2, 2), False),
    ((3, 2, 7, 9), None, 2.0, (3, 3), None, True),
    ((4, 6, 31, 29), None, 1.5, (4, 4), (2, 2), False),
    ((2, 8, 9, 11), (1584, 198, 11, 1), 2.0, (3, 2), 1, False),
]

_TOLERANCE_MAP = {
    infinicore.float16: {"atol": 1e-3, "rtol": 1e-2},
    infinicore.float32: {"atol": 1e-5, "rtol": 1e-4},
}
_TENSOR_DTYPES = [infinicore.float16, infinicore.float32]


def parse_test_cases():
    tests = []
    for in_shape, in_strides, p, k, s, ceil_mode in _TEST_CASES_DATA:
        for dtype in _TENSOR_DTYPES:
            tol = _TOLERANCE_MAP[dtype]
            in_spec = TensorSpec.from_tensor(in_shape, in_strides, dtype)
            kwargs = {"norm_type": p, "kernel_size": k, "ceil_mode": ceil_mode}
            if s is not None:
                kwargs["stride"] = s
            tests.append(
                TestCase(
                    inputs=[in_spec],
                    kwargs=kwargs,
                    output_spec=None,
                    comparison_target=None,
                    tolerance=tol,
                    description="LpPool2d - OUT_OF_PLACE",
                )
            )

    return tests


class OpTest(BaseOperatorTest):
    """LpPool2d operator test with simplified implementation"""

    def __init__(self):
        super().__init__("LpPool2d")

    def get_test_cases(self):
        return parse_test_cases()

    def torch_operator(self, *args, **kwargs):
        return torch.nn.functional.lp_pool2d(*args, **kwargs)

    # def infinicore_operator(self, *args, **kwargs):
    #     """InfiniCore implementation (operator not yet available)."""
    #     return infinicore.nn.functional.lp_pool2d(*args, **kwargs)


def main():
    """Main entry point"""
    runner = GenericTestRunner(OpTest)
    runner.run_and_exit()


if __name__ == "__main__":
    main()
