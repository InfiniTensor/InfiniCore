import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import infinicore
from framework.base import BaseOperatorTest, TensorSpec, TestCase
from framework.tensor import TensorInitializer
from framework.runner import GenericTestRunner

# Test cases format: (input_shape, input_strides_or_None, indices_shape)
_TEST_CASES_DATA = [
    ((3, 4), None, (6,)),
    ((5,), None, (3,)),
    ((2, 3, 4), (24, 8, 2), (4,)),
    ((1, 6), None, (2,)),
    ((4, 4), None, (8,)),
    ((2, 3, 2), None, (3,)),
]

_TOLERANCE_MAP = {infinicore.float32: {"atol": 1e-5, "rtol": 1e-4}}
_TENSOR_DTYPES = [infinicore.float32]


def parse_test_cases():
    test_cases = []
    for shape, strides, idx_shape in _TEST_CASES_DATA:
        input_spec = TensorSpec.from_tensor(shape, strides, infinicore.float32)

        # indices for infinicore.take are flattened indices in [0, input.numel())
        prod = 1
        for s in shape:
            prod *= s

        indices_spec = TensorSpec.from_tensor(
            idx_shape,
            None,
            infinicore.int64,
            init_mode=TensorInitializer.RANDINT,
            low=0,
            high=max(1, prod),
        )

        test_cases.append(
            TestCase(
                inputs=[input_spec, indices_spec],
                kwargs={},
                output_spec=None,
                comparison_target=None,
                tolerance=_TOLERANCE_MAP[infinicore.float32],
                description=f"take - OUT_OF_PLACE",
            )
        )

    return test_cases


class OpTest(BaseOperatorTest):
    """Take operator test with simplified implementation"""

    def __init__(self):
        super().__init__("Take")

    def get_test_cases(self):
        return parse_test_cases()

    def torch_operator(self, *args, **kwargs):
        return torch.take(*args, **kwargs)

    # def infinicore_operator(self, *args, **kwargs):
    #     """InfiniCore implementation (operator not yet available)."""
    #     return infinicore.take(*args, **kwargs)


def main():
    """Main entry point"""
    runner = GenericTestRunner(OpTest)
    runner.run_and_exit()


if __name__ == "__main__":
    main()
