import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import infinicore
from framework.base import BaseOperatorTest, TensorSpec, TestCase
from framework.runner import GenericTestRunner
from framework.utils import is_broadcast

# Test cases format: (shape, input_strides_or_None)
# infinicore.flipud(input) flips up/down (dim=-2 for 2D-like tensors)

_TEST_CASES_DATA = [
    ((13, 4), None),
    ((8, 16), (128, 1)),
    ((2, 3, 4), None),
    ((4, 5), None),
    ((16, 64), None),
    ((3, 4, 5), (60, 20, 4)),
]

_TOLERANCE_MAP = {
    infinicore.float16: {"atol": 0, "rtol": 1e-2},
    infinicore.float32: {"atol": 0, "rtol": 1e-4},
    infinicore.bfloat16: {"atol": 0, "rtol": 5e-2},
}

_TENSOR_DTYPES = [infinicore.float16, infinicore.bfloat16, infinicore.float32]


def parse_test_cases():
    test_cases = []
    for data in _TEST_CASES_DATA:
        shape = data[0]
        in_strides = data[1] if len(data) > 1 else None

        supports_inplace = not is_broadcast(in_strides)

        for dtype in _TENSOR_DTYPES:
            tol = _TOLERANCE_MAP.get(dtype, {"atol": 0, "rtol": 1e-4})
            in_spec = TensorSpec.from_tensor(shape, in_strides, dtype)

            test_cases.append(
                TestCase(
                    inputs=[in_spec],
                    kwargs={},
                    output_spec=None,
                    comparison_target=None,
                    tolerance=tol,
                    description="flipud - OUT_OF_PLACE",
                )
            )

    return test_cases


class OpTest(BaseOperatorTest):
    """FlipUD operator test with simplified implementation"""

    def __init__(self):
        super().__init__("FlipUD")

    def get_test_cases(self):
        return parse_test_cases()

    def torch_operator(self, *args, **kwargs):
        return torch.flipud(*args, **kwargs)

    # def infinicore_operator(self, *args, **kwargs):
    #     """InfiniCore implementation (operator not yet available)."""
    #     return infinicore.flipud(*args, **kwargs)


def main():
    """Main entry point"""
    runner = GenericTestRunner(OpTest)
    runner.run_and_exit()


if __name__ == "__main__":
    main()
