import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import infinicore
from framework.base import BaseOperatorTest, TensorSpec, TestCase
from framework.runner import GenericTestRunner
from framework.datatypes import to_torch_dtype

# Test cases format: (shape, stride, dtype)
# empty_strided creates a tensor with given stride; use small shapes to exercise strides.
_TEST_CASES_DATA = [
    ((3, 4), (16, 1), infinicore.float32),
    ((4, 3), (12, 4), infinicore.float16),
    ((2, 5), (20, 1), infinicore.float32),
    ((1, 6), (48, 8), infinicore.bfloat16),
    ((2, 2, 2), (8, 4, 2), infinicore.float32),
    ((5,), (1,), infinicore.float32),
]

_TOLERANCE_MAP = {
    infinicore.float32: {"atol": 0, "rtol": 0},
    infinicore.float16: {"atol": 1e-3, "rtol": 1e-3},
    infinicore.bfloat16: {"atol": 1e-2, "rtol": 1e-2},
}


def parse_test_cases():
    test_cases = []
    for shape, stride, dtype in _TEST_CASES_DATA:
        kwargs = {"size": shape, "stride": stride, "dtype": dtype}

        test_cases.append(
            TestCase(
                inputs=[],
                kwargs=kwargs,
                output_spec=None,
                comparison_target=None,
                tolerance=_TOLERANCE_MAP.get(dtype, {"atol": 0, "rtol": 0}),
                description=f"empty_strided - OUT_OF_PLACE",
            )
        )

    return test_cases


class OpTest(BaseOperatorTest):
    """EmptyStrided operator test with simplified implementation"""

    def __init__(self):
        super().__init__("EmptyStrided")

    def get_test_cases(self):
        return parse_test_cases()

    def torch_operator(self, *args, **kwargs):
        if "size" not in kwargs:
            raise TypeError("empty_strided test did not provide 'size' parameter")
        size = kwargs.pop("size")

        if "stride" not in kwargs:
            raise TypeError("empty_strided test did not provide 'stride' parameter")
        stride = kwargs.pop("stride")

        if "dtype" not in kwargs:
            raise TypeError("empty_strided test did not provide 'dtype' parameter")
        dtype_torch = to_torch_dtype(kwargs.pop("dtype"))
        if dtype_torch is None:
            raise TypeError("empty_strided test provided unsupported 'dtype' parameter")

        return torch.empty_strided(tuple(size), stride=stride, dtype=dtype_torch)

    # def infinicore_operator(self, *args, **kwargs):
    #     """InfiniCore implementation (operator not yet available)."""
    #     return infinicore.empty_strided(*args, **kwargs)


def main():
    """Main entry point"""
    runner = GenericTestRunner(OpTest)
    runner.run_and_exit()


if __name__ == "__main__":
    main()
