import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import infinicore
from framework.base import BaseOperatorTest, TensorSpec, TestCase
from framework.runner import GenericTestRunner

# Test cases format: (input_shape, input_strides_or_None, dim, index_shape)
_TEST_CASES_DATA = [
    ((3, 4), None, 1, (2,)),
    ((5, 6), (30, 1), 0, (3,)),
    ((2, 3, 4), None, 2, (2,)),
    ((4, 4), None, -1, (1,)),
    ((6, 2), (12, 1), 1, (2,)),
    ((3, 5), None, 0, (1,)),
]

_TOLERANCE_MAP = {infinicore.float32: {"atol": 1e-5, "rtol": 1e-4}}
_TENSOR_DTYPES = [infinicore.float32]


def parse_test_cases():
    test_cases = []
    for shape, strides, dim, idx_shape in _TEST_CASES_DATA:
        input_spec = TensorSpec.from_tensor(shape, strides, infinicore.float32)
        # index_select requires a 1-D index tensor
        index_len = idx_shape[0]
        from framework.tensor import TensorInitializer

        index_spec = TensorSpec.from_tensor(
            (index_len,),
            None,
            infinicore.int64,
            init_mode=TensorInitializer.RANDINT,
            low=0,
            high=shape[dim],
        )

        # Use positional dim to match infinicore.index_select(input, dim, index)
        test_cases.append(
            TestCase(
                inputs=[input_spec, dim, index_spec],
                kwargs={},
                output_spec=None,
                comparison_target=None,
                tolerance=_TOLERANCE_MAP[infinicore.float32],
                description=f"index_select - OUT_OF_PLACE",
            )
        )

    return test_cases


class OpTest(BaseOperatorTest):
    """IndexSelect operator test with simplified implementation"""

    def __init__(self):
        super().__init__("IndexSelect")

    def get_test_cases(self):
        return parse_test_cases()

    def torch_operator(self, *args, **kwargs):
        return torch.index_select(*args, **kwargs)

    # def infinicore_operator(self, *args, **kwargs):
    #     """InfiniCore implementation (operator not yet available)."""
    #     return infinicore.index_select(*args, **kwargs)


def main():
    """Main entry point"""
    runner = GenericTestRunner(OpTest)
    runner.run_and_exit()


if __name__ == "__main__":
    main()
