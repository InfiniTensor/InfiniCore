import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import infinicore
from framework.base import BaseOperatorTest, TensorSpec, TestCase
from framework.datatypes import to_torch_dtype
from framework.runner import GenericTestRunner

# Test cases format: (base_shape, base_strides_or_None, dtype_or_None)
# Note: empty_like returns uninitialized memory; we validate shape/dtype via output_spec
_TEST_CASES_DATA = [
    ((3, 4), None, None),
    ((6, 2), (12, 1), infinicore.float16),
    ((5, 5), None, infinicore.float32),
    ((1, 7), None, infinicore.bfloat16),
    ((8, 3), (24, 1), None),
    ((2, 2, 2), None, infinicore.float32),
]

_TOLERANCE_MAP = {
    infinicore.float32: {"atol": 1e-6, "rtol": 1e-5},
    infinicore.float16: {"atol": 1e-3, "rtol": 1e-3},
    infinicore.bfloat16: {"atol": 1e-2, "rtol": 1e-2},
}

_TENSOR_DTYPES = [infinicore.float32, infinicore.float16, infinicore.bfloat16]


def parse_test_cases():
    test_cases = []
    for base_shape, base_strides, dtype in _TEST_CASES_DATA:
        for input_dtype in _TENSOR_DTYPES:
            base_spec = TensorSpec.from_tensor(base_shape, base_strides, input_dtype)

            kwargs = {"dtype": dtype} if dtype is not None else {}

            test_cases.append(
                TestCase(
                    inputs=[base_spec],
                    kwargs=kwargs,
                    output_spec=None,
                    comparison_target=None,
                    tolerance=_TOLERANCE_MAP.get(
                        input_dtype, {"atol": 1e-5, "rtol": 1e-4}
                    ),
                    description="empty_like - OUT_OF_PLACE",
                )
            )

    return test_cases


class OpTest(BaseOperatorTest):
    """EmptyLike operator test with simplified implementation"""

    def __init__(self):
        super().__init__("EmptyLike")

    def get_test_cases(self):
        return parse_test_cases()

    def torch_operator(self, *args, **kwargs):
        if "dtype" not in kwargs:
            dtype_torch = None
        else:
            dtype_torch = to_torch_dtype(kwargs.pop("dtype"))
        return torch.empty_like(*args, dtype=dtype_torch)

    # def infinicore_operator(self, *args, **kwargs):
    #     """InfiniCore implementation (operator not yet available)."""
    #     return infinicore.empty_like(*args, **kwargs)


def main():
    """Main entry point"""
    runner = GenericTestRunner(OpTest)
    runner.run_and_exit()


if __name__ == "__main__":
    main()
