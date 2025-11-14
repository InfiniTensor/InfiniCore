import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import infinicore
from framework.base import BaseOperatorTest, TensorSpec, TestCase
from framework.runner import GenericTestRunner
from framework.utils import is_broadcast

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

        out_spec = (
            TensorSpec.from_tensor(
                (length,)
                + tuple(s for s in shape if s is not None and s != shape[dim]),
                None,
                infinicore.float32,
            )
            if False
            else None
        )
        # Above out_spec construction is conservative and not used; we let test framework compare outputs directly.

        kwargs = {"dim": dim, "start": start, "length": length}

        test_cases.append(
            TestCase(
                inputs=[input_spec],
                kwargs=kwargs,
                output_spec=None,
                comparison_target=None,
                tolerance=_TOLERANCE_MAP[infinicore.float32],
                description=f"narrow_copy - OUT_OF_PLACE",
            )
        )

    return test_cases


class OpTest(BaseOperatorTest):
    """NarrowCopy operator test with simplified implementation"""

    def __init__(self):
        super().__init__("NarrowCopy")

    def get_test_cases(self):
        return parse_test_cases()

    def torch_operator(self, *args, **kwargs):
        return torch.narrow_copy(*args, **kwargs)

    # def infinicore_operator(self, *args, **kwargs):
    #     """InfiniCore implementation (operator not yet available)."""
    #     return infinicore.narrow_copy(*args, **kwargs)


def main():
    """Main entry point"""
    runner = GenericTestRunner(OpTest)
    runner.run_and_exit()


if __name__ == "__main__":
    main()
