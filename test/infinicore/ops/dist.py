import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import infinicore
from framework.base import BaseOperatorTest, TensorSpec, TestCase
from framework.runner import GenericTestRunner

# Test cases format: (shape, a_strides_or_None, b_strides_or_None, p_or_None)
# dist computes p-norm distance between two tensors

_TEST_CASES_DATA = [
    ((8, 8), None, None, None),
    ((8, 8), (16, 1), (16, 1), 1.0),
    ((2, 3, 4), None, None, 2.0),
    ((1, 8), None, (0, 1), None),
    ((16, 64), (128, 1), (128, 1), 3.0),
    ((4, 5, 6), (60, 12, 2), (60, 12, 2), 0.5),
]

_TOLERANCE_MAP = {
    infinicore.float16: {"atol": 1e-3, "rtol": 1e-2},
    infinicore.float32: {"atol": 1e-5, "rtol": 1e-4},
}

_TENSOR_DTYPES = [infinicore.float16, infinicore.float32]


def parse_test_cases():
    test_cases = []
    for data in _TEST_CASES_DATA:
        shape, a_strides, b_strides, p = data
        for dtype in _TENSOR_DTYPES:
            tol = _TOLERANCE_MAP.get(dtype, {"atol": 1e-5, "rtol": 1e-3})
            a_spec = TensorSpec.from_tensor(shape, a_strides, dtype)
            b_spec = TensorSpec.from_tensor(shape, b_strides, dtype)

            kwargs = {}
            if p is not None:
                kwargs["p"] = p

            test_cases.append(
                TestCase(
                    inputs=[a_spec, b_spec],
                    kwargs=kwargs,
                    output_spec=None,
                    comparison_target=None,
                    tolerance=tol,
                    description="Dist - OUT_OF_PLACE",
                )
            )

    return test_cases


class OpTest(BaseOperatorTest):
    """Dist operator test with simplified implementation"""

    def __init__(self):
        super().__init__("Dist")

    def get_test_cases(self):
        return parse_test_cases()

    def torch_operator(self, *args, **kwargs):
        return torch.dist(*args, **kwargs)

    # def infinicore_operator(self, *args, **kwargs):
    #     """InfiniCore implementation (operator not yet available)."""
    #     return infinicore.dist(*args, **kwargs)


def main():
    """Main entry point"""
    runner = GenericTestRunner(OpTest)
    runner.run_and_exit()


if __name__ == "__main__":
    main()
