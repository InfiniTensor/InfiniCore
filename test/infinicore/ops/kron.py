import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import infinicore
from framework.base import BaseOperatorTest, TensorSpec, TestCase
from framework.runner import GenericTestRunner

# Test cases format: (a_shape, b_shape, a_strides_or_None, b_strides_or_None)
# infinicore.kron(a, b)

_TEST_CASES_DATA = [
    ((2, 3), (4, 1), None, None),
    ((1,), (3,), None, None),
    ((4, 4), (2, 2), (64, 16), (8, 1)),
    ((6,), (6,), None, None),
    ((3, 2), (2, 3), None, (12, 1)),
    ((8, 1), (1, 8), None, None),
]

_TOLERANCE_MAP = {
    infinicore.float16: {"atol": 1e-3, "rtol": 1e-2},
    infinicore.float32: {"atol": 1e-5, "rtol": 1e-4},
    infinicore.bfloat16: {"atol": 1e-2, "rtol": 5e-2},
}

_TENSOR_DTYPES = [infinicore.float16, infinicore.bfloat16, infinicore.float32]


def parse_test_cases():
    test_cases = []
    for a_shape, b_shape, a_strides, b_strides in _TEST_CASES_DATA:
        for dtype in _TENSOR_DTYPES:
            tol = _TOLERANCE_MAP.get(dtype, {"atol": 1e-5, "rtol": 1e-4})
            a = TensorSpec.from_tensor(a_shape, a_strides, dtype)
            b = TensorSpec.from_tensor(b_shape, b_strides, dtype)

            test_cases.append(
                TestCase(
                    inputs=[a, b],
                    kwargs={},
                    output_spec=None,
                    comparison_target=None,
                    tolerance=tol,
                    description="kron - OUT_OF_PLACE",
                )
            )

    return test_cases


class OpTest(BaseOperatorTest):
    """kron operator test with simplified implementation"""

    def __init__(self):
        super().__init__("kron")

    def get_test_cases(self):
        return parse_test_cases()

    def torch_operator(self, *args, **kwargs):
        return torch.kron(*args, **kwargs)

    # def infinicore_operator(self, *args, **kwargs):
    #     """InfiniCore implementation (operator not yet available)."""
    #     return infinicore.kron(*args, **kwargs)


def main():
    """Main entry point"""
    runner = GenericTestRunner(OpTest)
    runner.run_and_exit()


if __name__ == "__main__":
    main()
