import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import infinicore
from framework.base import BaseOperatorTest, TensorSpec, TestCase
from framework.runner import GenericTestRunner
from framework.utils import is_broadcast

# Test cases format: (in_shape, in_strides_or_None, dim_or_None, keepdim_or_None, dtype_or_None)
# sum computes the sum along dim(s) or overall

_TEST_CASES_DATA = [
    ((8, 8), None, None, None, None),
    ((8, 8), (16, 1), 1, False, None),
    ((2, 3, 4), None, 0, True, None),
    ((1, 8), None, (0,), False, None),
    ((16, 64), (128, 1), None, None, None),
    ((4, 5, 6), (60, 12, 2), 2, True, None),
]

_TOLERANCE_MAP = {
    infinicore.float16: {"atol": 1e-3, "rtol": 1e-2},
    infinicore.float32: {"atol": 1e-5, "rtol": 1e-4},
}

_TENSOR_DTYPES = [infinicore.float16, infinicore.float32]


def parse_test_cases():
    test_cases = []
    for data in _TEST_CASES_DATA:
        shape, strides, dim, keepdim, dtype_param = data
        # out_supports_inplace = not is_broadcast(out_strides)

        for dtype in _TENSOR_DTYPES:
            tol = _TOLERANCE_MAP.get(dtype, {"atol": 1e-5, "rtol": 1e-3})
            in_spec = TensorSpec.from_tensor(shape, strides, dtype)

            kwargs = {}
            if dim is not None:
                kwargs["dim"] = dim
            if keepdim is not None:
                kwargs["keepdim"] = keepdim
            if dtype_param is not None:
                kwargs["dtype"] = dtype_param

            test_cases.append(
                TestCase(
                    inputs=[in_spec],
                    kwargs=kwargs,
                    output_spec=None,
                    comparison_target=None,
                    tolerance=tol,
                    description="Sum - OUT_OF_PLACE",
                )
            )

    return test_cases


class OpTest(BaseOperatorTest):
    """Sum operator test with simplified implementation"""

    def __init__(self):
        super().__init__("Sum")

    def get_test_cases(self):
        return parse_test_cases()

    def torch_operator(self, *args, **kwargs):
        return torch.sum(*args, **kwargs)

    # def infinicore_operator(self, *args, **kwargs):
    #     """InfiniCore implementation (operator not yet available)."""
    #     return infinicore.sum(*args, **kwargs)


def main():
    """Main entry point"""
    runner = GenericTestRunner(OpTest)
    runner.run_and_exit()


if __name__ == "__main__":
    main()
