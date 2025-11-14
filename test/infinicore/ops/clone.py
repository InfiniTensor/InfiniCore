import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import infinicore
from framework.base import BaseOperatorTest, TensorSpec, TestCase
from framework.runner import GenericTestRunner

# Test cases format: (shape, input_strides_or_None)
_TEST_CASES_DATA = [
    ((3, 4), None),
    ((6, 2), (12, 1)),
    ((5, 5), None),
    ((1, 7), None),
    ((8, 3), (24, 1)),
    ((2, 2, 2), None),
]

_TOLERANCE_MAP = {infinicore.float32: {"atol": 1e-5, "rtol": 1e-4}}
_TENSOR_DTYPES = [infinicore.float32]


def parse_test_cases():
    test_cases = []
    for data in _TEST_CASES_DATA:
        shape, strides = data

        for dtype in _TENSOR_DTYPES:
            tol = _TOLERANCE_MAP.get(dtype)
            input_spec = TensorSpec.from_tensor(shape, strides, dtype)

            # clone is out-of-place; also test clone with memory_format not None if supported
            test_cases.append(
                TestCase(
                    inputs=[input_spec],
                    kwargs={},
                    output_spec=None,
                    comparison_target=None,
                    tolerance=tol,
                    description=f"clone - OUT_OF_PLACE",
                )
            )

    return test_cases


class OpTest(BaseOperatorTest):
    """Clone operator test with simplified implementation"""

    def __init__(self):
        super().__init__("Clone")

    def get_test_cases(self):
        return parse_test_cases()

    def torch_operator(self, *args, **kwargs):
        return torch.clone(*args, **kwargs)

    # def infinicore_operator(self, *args, **kwargs):
    #     """InfiniCore implementation (operator not yet available)."""
    #     return infinicore.clone(*args, **kwargs)


def main():
    """Main entry point"""
    runner = GenericTestRunner(OpTest)
    runner.run_and_exit()


if __name__ == "__main__":
    main()
