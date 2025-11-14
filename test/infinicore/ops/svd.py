import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import infinicore
from framework.base import BaseOperatorTest, TensorSpec, TestCase
from framework.runner import GenericTestRunner

# Test cases format: (matrix_shape, strides_or_None, compute_uv_or_None)
# infinicore.svd(a, some=True, compute_uv=True) — different return shapes depending on flags

_TEST_CASES_DATA = [
    ((3, 3), None, True),
    ((4, 2), None, True),
    ((6, 6), (360, 60), True),
    ((2, 4), None, False),
    ((8, 8), None, True),
    ((1, 1), None, True),
]

_TOLERANCE_MAP = {
    infinicore.float32: {"atol": 1e-5, "rtol": 1e-4},
}

_TENSOR_DTYPES = [infinicore.float32]


def parse_test_cases():
    test_cases = []
    for shape, strides, compute_uv in _TEST_CASES_DATA:
        for dtype in _TENSOR_DTYPES:
            tol = _TOLERANCE_MAP.get(dtype, {"atol": 1e-5, "rtol": 1e-4})
            spec = TensorSpec.from_tensor(shape, strides, dtype)
            kwargs = {}
            if compute_uv is not None:
                kwargs["compute_uv"] = compute_uv

            test_cases.append(
                TestCase(
                    inputs=[spec],
                    kwargs=kwargs,
                    output_spec=None,
                    comparison_target=None,
                    tolerance=tol,
                    description="svd - OUT_OF_PLACE",
                )
            )

    return test_cases


class OpTest(BaseOperatorTest):
    """svd operator test with simplified implementation"""

    def __init__(self):
        super().__init__("svd")

    def get_test_cases(self):
        return parse_test_cases()

    def torch_operator(self, *args, **kwargs):
        return torch.svd(*args, **kwargs)

    # def infinicore_operator(self, *args, **kwargs):
    #     """InfiniCore implementation (operator not yet available)."""
    #     return infinicore.svd(*args, **kwargs)


def main():
    """Main entry point"""
    runner = GenericTestRunner(OpTest)
    runner.run_and_exit()


if __name__ == "__main__":
    main()
