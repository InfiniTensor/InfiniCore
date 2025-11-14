import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import infinicore
from framework.base import BaseOperatorTest, TensorSpec, TestCase
from framework.runner import GenericTestRunner

# Test cases format: (vec1_shape, vec2_shape, vec1_strides_or_None, vec2_strides_or_None)
# vdot(a, b) — conjugate dot product for 1-D vectors

_TEST_CASES_DATA = [
    ((3,), (3,), None, None),
    ((8,), (8,), (0,), None),
    ((1,), (1,), None, None),
    ((16,), (16,), None, (256,)),
    ((5,), (5,), None, None),
    ((32,), (32,), (64,), (64,)),
]

_TOLERANCE_MAP = {
    infinicore.float16: {"atol": 1e-3, "rtol": 1e-2},
    infinicore.float32: {"atol": 1e-5, "rtol": 1e-4},
    infinicore.bfloat16: {"atol": 1e-2, "rtol": 5e-2},
}

_TENSOR_DTYPES = [infinicore.float16, infinicore.bfloat16, infinicore.float32]


def parse_test_cases():
    test_cases = []
    for s1, s2, st1, st2 in _TEST_CASES_DATA:
        for dtype in _TENSOR_DTYPES:
            tol = _TOLERANCE_MAP.get(dtype, {"atol": 1e-5, "rtol": 1e-4})
            a = TensorSpec.from_tensor(s1, st1, dtype)
            b = TensorSpec.from_tensor(s2, st2, dtype)

            test_cases.append(
                TestCase(
                    inputs=[a, b],
                    kwargs={},
                    output_spec=None,
                    comparison_target=None,
                    tolerance=tol,
                    description="vdot - OUT_OF_PLACE",
                )
            )

    return test_cases


class OpTest(BaseOperatorTest):
    """vdot operator test with simplified implementation"""

    def __init__(self):
        super().__init__("vdot")

    def get_test_cases(self):
        return parse_test_cases()

    def torch_operator(self, *args, **kwargs):
        return torch.vdot(*args, **kwargs)

    # def infinicore_operator(self, *args, **kwargs):
    #     """InfiniCore implementation (operator not yet available)."""
    #     return infinicore.vdot(*args, **kwargs)


def main():
    """Main entry point"""
    runner = GenericTestRunner(OpTest)
    runner.run_and_exit()


if __name__ == "__main__":
    main()
