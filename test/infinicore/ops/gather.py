import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import infinicore
from framework.base import BaseOperatorTest, TensorSpec, TestCase
from framework.runner import GenericTestRunner
from framework.tensor import TensorInitializer
from framework.utils import is_broadcast

# Test cases format: (input_shape, input_strides_or_None, dim, index_shape)
_TEST_CASES_DATA = [
    ((3, 4), None, 1, (3, 2)),
    ((5, 6), (30, 1), 0, (2, 6)),
    ((2, 3, 4), None, 2, (2, 3, 2)),
    ((4, 4), None, -1, (4, 2)),
    ((6, 2), (12, 1), 1, (6, 1)),
    ((3, 5), None, 0, (1, 5)),
]

_TOLERANCE_MAP = {infinicore.float32: {"atol": 1e-5, "rtol": 1e-4}}
_TENSOR_DTYPES = [infinicore.float32]


def parse_test_cases():
    test_cases = []
    for shape, strides, dim, idx_shape in _TEST_CASES_DATA:
        input_spec = TensorSpec.from_tensor(shape, strides, infinicore.float32)
        input_ndim = len(shape)
        if dim < 0:
            actual_dim = input_ndim + dim
        else:
            actual_dim = dim

        size_dim = shape[actual_dim]

        # index tensor spec：值必须在 [0, size_dim)
        index_spec = TensorSpec.from_tensor(
            idx_shape,
            None,
            infinicore.int64,
            init_mode=TensorInitializer.RANDINT,
            low=0,
            high=size_dim,  # exclusive bound
        )

        # gather returns same dtype as input
        kwargs = {"dim": dim}

        test_cases.append(
            TestCase(
                inputs=[input_spec, index_spec],
                kwargs=kwargs,
                output_spec=None,
                comparison_target=None,
                tolerance=_TOLERANCE_MAP[infinicore.float32],
                description=f"gather - OUT_OF_PLACE",
            )
        )

    return test_cases


class OpTest(BaseOperatorTest):
    """Gather operator test with simplified implementation"""

    def __init__(self):
        super().__init__("Gather")

    def get_test_cases(self):
        return parse_test_cases()

    def torch_operator(self, *args, **kwargs):
        if "dim" not in kwargs:
            raise TypeError("gather test did not provide 'dim' parameter")
        dim = kwargs.pop("dim")
        input = args[0]
        index = args[1]

        return torch.gather(input, dim, index)

    # def infinicore_operator(self, *args, **kwargs):
    #     """InfiniCore implementation (operator not yet available)."""
    #     return infinicore.gather(*args, **kwargs)


def main():
    """Main entry point"""
    runner = GenericTestRunner(OpTest)
    runner.run_and_exit()


if __name__ == "__main__":
    main()
