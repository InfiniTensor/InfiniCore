import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import infinicore
from framework.base import BaseOperatorTest, TensorSpec, TestCase
from framework.runner import GenericTestRunner

# Test cases format: (in_shape, in_strides_or_None)
# signbit returns boolean-like tensor; tolerances are exact.

_TEST_CASES_DATA = [
    ((2, 3), None),
    ((1, 4, 8), (32, 8, 1)),
    ((3, 2, 5, 7), None),
    ((2, 1, 16), None),
    ((1, 8, 9, 11), (792, 99, 11, 1)),
    ((2, 6, 10), None),
]

_TOLERANCE_MAP = {infinicore.bool: {"atol": 0.0, "rtol": 0.0}}
_TENSOR_DTYPES = [infinicore.float16, infinicore.float32]


def parse_test_cases():
    cases = []
    for shape, strides in _TEST_CASES_DATA:
        for dtype in _TENSOR_DTYPES:
            tol = _TOLERANCE_MAP[infinicore.bool]
            in_spec = TensorSpec.from_tensor(shape, strides, dtype)

            # Out-of-place (returns boolean mask)
            cases.append(
                TestCase(
                    inputs=[in_spec],
                    kwargs={},
                    output_spec=None,
                    comparison_target=None,
                    tolerance=tol,
                    description="signbit_out_of_place",
                )
            )

            # Explicit out (boolean dtype out)
            out_spec = TensorSpec.from_tensor(shape, None, infinicore.bool)
            cases.append(
                TestCase(
                    inputs=[in_spec],
                    kwargs={},
                    output_spec=out_spec,
                    comparison_target="out",
                    tolerance=tol,
                    description="signbit_explicit_out",
                )
            )

            # In-place on input (not typical for boolean outputs) - skip because in-place would change dtype
            # Note: PyTorch does not support in-place signbit that preserves dtype; therefore no in-place case.

    return cases


class OpTest(BaseOperatorTest):
    """SignBit operator test with simplified implementation"""

    def __init__(self):
        super().__init__("SignBit")

    def get_test_cases(self):
        return parse_test_cases()

    def torch_operator(self, *args, **kwargs):
        return torch.signbit(*args, **kwargs)

    # def infinicore_operator(self, *args, **kwargs):
    #     """InfiniCore implementation (operator not yet available)."""
    #     return infinicore.signbit(*args, **kwargs)


def main():
    """Main entry point"""
    runner = GenericTestRunner(OpTest)
    runner.run_and_exit()


if __name__ == "__main__":
    main()
