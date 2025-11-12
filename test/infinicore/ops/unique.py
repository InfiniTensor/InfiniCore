import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import infinicore
from framework.base import BaseOperatorTest, TensorSpec, TestCase
from framework.runner import GenericTestRunner
from framework.utils import is_broadcast

# Test cases format: (in_shape, in_strides_or_None, sorted_or_None, return_inverse_or_None, return_counts_or_None)
# unique returns unique values and optionally inverse indices and counts

_TEST_CASES_DATA = [
    ((8,), None, True, False, False),
    ((8, 8), None, False, True, False),
    ((2, 3, 4), (24, 8, 2), True, False, True),
    ((1, 8), None, None, True, True),
    ((16, 64), (128, 1), False, False, True),
    ((4, 5), (20, 4), True, True, False),
]

_TOLERANCE_MAP = {
    infinicore.float16: {"atol": 1e-3, "rtol": 1e-2},
    infinicore.float32: {"atol": 1e-5, "rtol": 1e-4},
    infinicore.int32: {"atol": 0, "rtol": 0},
}

_TENSOR_DTYPES = [infinicore.int32, infinicore.float32]


def parse_test_cases():
    test_cases = []
    for data in _TEST_CASES_DATA:
        shape, strides, sorted_flag, return_inverse, return_counts = data

        for dtype in _TENSOR_DTYPES:
            tol = _TOLERANCE_MAP.get(dtype, {"atol": 1e-5, "rtol": 1e-3})
            in_spec = TensorSpec.from_tensor(shape, strides, dtype)

            kwargs = {}
            if sorted_flag is not None:
                kwargs["sorted"] = sorted_flag
            if return_inverse:
                kwargs["return_inverse"] = True
            if return_counts:
                kwargs["return_counts"] = True

            test_cases.append(
                TestCase(
                    inputs=[in_spec],
                    kwargs=kwargs,
                    output_spec=None,
                    comparison_target=None,
                    tolerance=tol,
                    description="Unique - OUT_OF_PLACE",
                )
            )

    return test_cases


class OpTest(BaseOperatorTest):
    """Unique operator test with simplified implementation"""

    def __init__(self):
        super().__init__("Unique")

    def get_test_cases(self):
        return parse_test_cases()

    def torch_operator(self, *args, **kwargs):
        return torch.unique(*args, **kwargs)

    # def infinicore_operator(self, *args, **kwargs):
    #     """InfiniCore implementation (operator not yet available)."""
    #     return infinicore.unique(*args, **kwargs)


def main():
    """Main entry point"""
    runner = GenericTestRunner(OpTest)
    runner.run_and_exit()


if __name__ == "__main__":
    main()
