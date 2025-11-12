import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import infinicore
from framework.base import BaseOperatorTest, TensorSpec, TestCase
from framework.runner import GenericTestRunner

# Test cases format: (input_shape, input_strides_or_None)
_TEST_CASES_DATA = [
    ((3, 4), None),
    ((5,), None),
    ((2, 2, 3), (12, 6, 2)),
    ((1, 6), None),
    ((4, 4), None),
    ((2, 3, 2), None),
]

_TOLERANCE_MAP = {infinicore.int64: {"atol": 0, "rtol": 0}}
_TENSOR_DTYPES = [infinicore.float32]


def parse_test_cases():
    test_cases = []
    for shape, strides in _TEST_CASES_DATA:
        input_spec = TensorSpec.from_tensor(shape, strides, infinicore.float32)

        test_cases.append(
            TestCase(
                inputs=[input_spec],
                kwargs={},
                output_spec=None,
                comparison_target=None,
                tolerance=_TOLERANCE_MAP[infinicore.int64],
                description=f"argwhere - OUT_OF_PLACE",
            )
        )

    return test_cases


class OpTest(BaseOperatorTest):
    """ArgWhere operator test with simplified implementation"""

    def __init__(self):
        super().__init__("ArgWhere")

    def get_test_cases(self):
        return parse_test_cases()

    def torch_operator(self, *args, **kwargs):
        return torch.argwhere(*args, **kwargs)

    # def infinicore_operator(self, *args, **kwargs):
    #     """InfiniCore implementation (operator not yet available)."""
    #     return infinicore.argwhere(*args, **kwargs)


def main():
    """Main entry point"""
    runner = GenericTestRunner(OpTest)
    runner.run_and_exit()


if __name__ == "__main__":
    main()
