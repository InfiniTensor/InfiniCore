import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import infinicore
from framework.base import BaseOperatorTest, TensorSpec, TestCase
from framework.runner import GenericTestRunner

# Test cases format: (input_shape, input_strides_or_None, mask_shape)
_TEST_CASES_DATA = [
    ((3, 4), None, (3, 4)),
    ((5,), None, (5,)),
    ((2, 2, 3), (12, 6, 2), (2, 2, 3)),
    ((1, 6), None, (1, 6)),
    ((4, 4), None, (4, 4)),
    ((2, 3, 2), None, (2, 3, 2)),
]

_TOLERANCE_MAP = {infinicore.float32: {"atol": 1e-5, "rtol": 1e-4}}
_TENSOR_DTYPES = [infinicore.float32]


def parse_test_cases():
    test_cases = []
    for shape, strides, mask_shape in _TEST_CASES_DATA:
        input_spec = TensorSpec.from_tensor(shape, strides, infinicore.float32)
        mask_spec = TensorSpec.from_tensor(mask_shape, None, infinicore.bool)

        test_cases.append(
            TestCase(
                inputs=[input_spec, mask_spec],
                kwargs={},
                output_spec=None,
                comparison_target=None,
                tolerance=_TOLERANCE_MAP[infinicore.float32],
                description=f"masked_select - OUT_OF_PLACE",
            )
        )

    return test_cases


class OpTest(BaseOperatorTest):
    """MaskedSelect operator test with simplified implementation"""

    def __init__(self):
        super().__init__("MaskedSelect")

    def get_test_cases(self):
        return parse_test_cases()

    def torch_operator(self, *args, **kwargs):
        return torch.masked_select(*args, **kwargs)

    # def infinicore_operator(self, *args, **kwargs):
    #     """InfiniCore implementation (operator not yet available)."""
    #     return infinicore.masked_select(*args, **kwargs)


def main():
    """Main entry point"""
    runner = GenericTestRunner(OpTest)
    runner.run_and_exit()


if __name__ == "__main__":
    main()
