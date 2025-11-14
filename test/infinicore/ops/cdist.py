import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import infinicore
from framework.base import BaseOperatorTest, TensorSpec, TestCase
from framework.runner import GenericTestRunner

# Test cases format: (x1_shape, x2_shape, x1_strides_or_None, x2_strides_or_None, p_or_None)

_TEST_CASES_DATA = [
    ((5, 3), (6, 3), None, None, None),
    ((1, 4), (2, 4), None, None, 1.0),
    ((8, 16), (8, 16), (128, 16), (128, 16), 2.0),
    ((3, 2), (4, 2), None, (0, 2), None),
    ((10, 5), (7, 5), None, None, float("inf")),
    ((2, 1), (3, 1), None, None, None),
]

_TOLERANCE_MAP = {
    infinicore.float32: {"atol": 1e-5, "rtol": 1e-4},
}

_TENSOR_DTYPES = [infinicore.float32]


def parse_test_cases():
    test_cases = []
    for x1_shape, x2_shape, x1_strides, x2_strides, p in _TEST_CASES_DATA:
        for dtype in _TENSOR_DTYPES:
            tol = _TOLERANCE_MAP.get(dtype, {"atol": 1e-5, "rtol": 1e-4})
            x1_spec = TensorSpec.from_tensor(x1_shape, x1_strides, dtype)
            x2_spec = TensorSpec.from_tensor(x2_shape, x2_strides, dtype)

            kwargs = {}
            if p is not None:
                kwargs["p"] = p

            test_cases.append(
                TestCase(
                    inputs=[x1_spec, x2_spec],
                    kwargs=kwargs,
                    output_spec=None,
                    comparison_target=None,
                    tolerance=tol,
                    description="cdist - OUT_OF_PLACE",
                )
            )

    return test_cases


class OpTest(BaseOperatorTest):
    """cdist operator test with simplified implementation"""

    def __init__(self):
        super().__init__("cdist")

    def get_test_cases(self):
        return parse_test_cases()

    def torch_operator(self, *args, **kwargs):
        return torch.cdist(*args, **kwargs)

    # def infinicore_operator(self, *args, **kwargs):
    #     """InfiniCore implementation (operator not yet available)."""
    #     return infinicore.cdist(*args, **kwargs)


def main():
    """Main entry point"""
    runner = GenericTestRunner(OpTest)
    runner.run_and_exit()


if __name__ == "__main__":
    main()
