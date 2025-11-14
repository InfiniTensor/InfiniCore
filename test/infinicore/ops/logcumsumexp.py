import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import infinicore
from framework.base import BaseOperatorTest, TensorSpec, TestCase
from framework.runner import GenericTestRunner
from framework.utils import is_broadcast

# Test cases format: (shape, dim, input_strides_or_None)
# logcumsumexp computes log of cumulative sum of exponentials along dim.

_TEST_CASES_DATA = [
    ((13, 4), 1, None),
    ((13, 4), 0, (10, 1)),
    ((8, 16), 1, None),
    ((2, 3, 5), 2, None),
    ((16, 64), 1, None),
    ((4, 5, 6), 0, None),
]

_TOLERANCE_MAP = {
    infinicore.float16: {"atol": 1e-3, "rtol": 1e-2},
    infinicore.float32: {"atol": 1e-5, "rtol": 1e-4},
    infinicore.bfloat16: {"atol": 1e-2, "rtol": 5e-2},
}

_TENSOR_DTYPES = [infinicore.float16, infinicore.bfloat16, infinicore.float32]


def parse_test_cases():
    test_cases = []
    for data in _TEST_CASES_DATA:
        shape, dim = data[0], data[1]
        in_strides = data[2] if len(data) > 2 else None

        supports_inplace = not is_broadcast(in_strides)

        for dtype in _TENSOR_DTYPES:
            tol = _TOLERANCE_MAP.get(dtype, {"atol": 1e-5, "rtol": 1e-4})
            input_spec = TensorSpec.from_tensor(shape, in_strides, dtype)
            out_spec = TensorSpec.from_tensor(shape, None, dtype)

            # Out-of-place
            kwargs = {"dim": dim}
            test_cases.append(
                TestCase(
                    inputs=[input_spec],
                    kwargs=kwargs,
                    output_spec=None,
                    comparison_target=None,
                    tolerance=tol,
                    description=f"logcumsumexp - OUT_OF_PLACE",
                )
            )

            # PyTorch does not expose explicit out for logcumsumexp — skip out tests

    return test_cases


class OpTest(BaseOperatorTest):
    """LogCumsumExp operator test with simplified implementation"""

    def __init__(self):
        super().__init__("LogCumsumExp")

    def get_test_cases(self):
        return parse_test_cases()

    def torch_operator(self, *args, **kwargs):
        return torch.logcumsumexp(*args, **kwargs)

    # def infinicore_operator(self, *args, **kwargs):
    #     """InfiniCore implementation (operator not yet available)."""
    #     return infinicore.logcumsumexp(*args, **kwargs)


def main():
    """Main entry point"""
    runner = GenericTestRunner(OpTest)
    runner.run_and_exit()


if __name__ == "__main__":
    main()
