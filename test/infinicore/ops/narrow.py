import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import infinicore
from framework.base import BaseOperatorTest, TensorSpec, TestCase
from framework.runner import GenericTestRunner

# Test cases format: (input_shape, input_strides_or_None, dim, start, length)
_TEST_CASES_DATA = [
    ((5, 6), None, 1, 0, 3),
    ((4, 4), (16, 1), 0, 1, 2),
    ((3, 5), None, 1, 1, 2),
    ((2, 6), None, 1, 0, 1),
    ((6, 3), None, 0, 2, 2),
    ((4, 7), (28, 1), 1, 1, 3),
]

_TOLERANCE_MAP = {infinicore.float32: {"atol": 1e-5, "rtol": 1e-4}}
_TENSOR_DTYPES = [infinicore.float32]


def parse_test_cases():
    test_cases = []
    for shape, strides, dim, start, length in _TEST_CASES_DATA:
        input_spec = TensorSpec.from_tensor(shape, strides, infinicore.float32)

        kwargs = {"dim": dim, "start": start, "length": length}

        test_cases.append(
            TestCase(
                inputs=[input_spec],
                kwargs=kwargs,
                output_spec=None,
                comparison_target=None,
                tolerance=_TOLERANCE_MAP[infinicore.float32],
                description=f"narrow - OUT_OF_PLACE",
            )
        )

    return test_cases


class OpTest(BaseOperatorTest):
    """Narrow operator test with simplified implementation"""

    def __init__(self):
        super().__init__("Narrow")

    def get_test_cases(self):
        return parse_test_cases()

    def torch_operator(self, *args, **kwargs):
        return torch.narrow(*args, **kwargs)

    # def infinicore_operator(self, *args, **kwargs):
    #     """InfiniCore implementation (operator not yet available)."""
    #     return infinicore.narrow(*args, **kwargs)


def main():
    """Main entry point"""
    runner = GenericTestRunner(OpTest)
    runner.run_and_exit()


if __name__ == "__main__":
    main()
