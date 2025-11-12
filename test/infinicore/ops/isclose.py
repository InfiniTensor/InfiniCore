import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import infinicore
from framework.base import BaseOperatorTest, TensorSpec, TestCase
from framework.runner import GenericTestRunner
from framework.utils import is_broadcast

# Test cases format: (shape, a_strides_or_None, b_strides_or_None, extra_kwargs_or_None)
# isclose compares two tensors with atol/rtol; we include kwargs combinations

_TEST_CASES_DATA = [
    ((8, 8), None, None, {"rtol": 1e-05, "atol": 1e-08}),
    ((8, 8), (16, 1), (16, 1), {"rtol": 1e-03, "atol": 1e-05}),
    ((8, 8), None, (0, 1), {"rtol": 1e-02, "atol": 1e-03}),
    ((2, 3, 4), None, None, {"rtol": 1e-02, "atol": 1e-03}),
    ((1, 8), None, None, {"equal_nan": True}),
    ((16, 64), (128, 1), (128, 1), None),
]

_TOLERANCE_MAP = {
    infinicore.float16: {"atol": 1e-3, "rtol": 1e-2},
    infinicore.float32: {"atol": 1e-5, "rtol": 1e-4},
    infinicore.bfloat16: {"atol": 1e-2, "rtol": 5e-2},
}

_TENSOR_DTYPES = [infinicore.float16, infinicore.float32, infinicore.bfloat16]


def parse_test_cases():
    test_cases = []
    for data in _TEST_CASES_DATA:
        shape, a_strides, b_strides, extra = data[0], data[1], data[2], data[3]

        a_supports_inplace = not is_broadcast(a_strides)
        b_supports_inplace = not is_broadcast(b_strides)

        for dtype in _TENSOR_DTYPES:
            tol = _TOLERANCE_MAP.get(dtype, {"atol": 1e-5, "rtol": 1e-3})
            a_spec = TensorSpec.from_tensor(shape, a_strides, dtype)
            b_spec = TensorSpec.from_tensor(shape, b_strides, dtype)

            kwargs = {}
            if extra is not None:
                kwargs.update(extra)

            # Out-of-place  not support 'out='
            test_cases.append(
                TestCase(
                    inputs=[a_spec, b_spec],
                    kwargs=kwargs,
                    output_spec=None,
                    comparison_target=None,
                    tolerance=tol,
                    description="IsClose - OUT_OF_PLACE",
                )
            )

    return test_cases


class OpTest(BaseOperatorTest):
    """IsClose operator test with simplified implementation"""

    def __init__(self):
        super().__init__("IsClose")

    def get_test_cases(self):
        return parse_test_cases()

    def torch_operator(self, *args, **kwargs):
        return torch.isclose(*args, **kwargs)

    # def infinicore_operator(self, *args, **kwargs):
    #     """InfiniCore implementation (operator not yet available)."""
    #     return infinicore.isclose(*args, **kwargs)


def main():
    """Main entry point"""
    runner = GenericTestRunner(OpTest)
    runner.run_and_exit()


if __name__ == "__main__":
    main()
