import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import infinicore
from framework.base import BaseOperatorTest, TensorSpec, TestCase
from framework.runner import GenericTestRunner

# Test cases format: (input_shape, input_strides_or_None, N)
_TEST_CASES_DATA = [
    ((5,), None, None),
    ((6,), None, 3),
    ((4,), None, 5),
    ((7,), None, None),
    ((8,), None, 4),
    ((3,), None, 2),
]

_TOLERANCE_MAP = {infinicore.float32: {"atol": 1e-5, "rtol": 1e-4}}
_TENSOR_DTYPES = [infinicore.float32]


def parse_test_cases():
    test_cases = []
    for data in _TEST_CASES_DATA:
        shape, strides, N = data

        for dtype in _TENSOR_DTYPES:
            tol = _TOLERANCE_MAP.get(dtype)
            input_spec = TensorSpec.from_tensor(shape, strides, dtype)

            kwargs = {}
            if N is not None:
                kwargs["N"] = N

            test_cases.append(
                TestCase(
                    inputs=[input_spec],
                    kwargs=kwargs,
                    output_spec=None,
                    comparison_target=None,
                    tolerance=tol,
                    description=f"vander - OUT_OF_PLACE",
                )
            )

    return test_cases


class OpTest(BaseOperatorTest):
    """Vander operator test with simplified implementation"""

    def __init__(self):
        super().__init__("Vander")

    def get_test_cases(self):
        return parse_test_cases()

    def torch_operator(self, *args, **kwargs):
        return torch.vander(*args, **kwargs)

    # def infinicore_operator(self, *args, **kwargs):
    #     """InfiniCore implementation (operator not yet available)."""
    #     return infinicore.vander(*args, **kwargs)


def main():
    """Main entry point"""
    runner = GenericTestRunner(OpTest)
    runner.run_and_exit()


if __name__ == "__main__":
    main()
