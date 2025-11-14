import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import infinicore
from framework.base import BaseOperatorTest, TensorSpec, TestCase
from framework.runner import GenericTestRunner
from framework.utils import is_broadcast

# Test cases format: (shape, p, a_strides_or_None)
# infinicore.pdist(input, p=2.0) computes pairwise distances between rows of input

_TEST_CASES_DATA = [
    ((8, 16), 2.0, None),
    ((10, 5), 2.0, (80, 8)),
    ((4, 3), 1.0, None),
    ((6, 8), 3.0, None),
    ((2, 16), 2.0, None),
    ((12, 4), 2.0, None),
]

_TOLERANCE_MAP = {
    infinicore.float32: {"atol": 1e-5, "rtol": 1e-4},
}

_TENSOR_DTYPES = [infinicore.float32]


def parse_test_cases():
    test_cases = []
    for data in _TEST_CASES_DATA:
        shape, p = data[0], data[1]
        in_strides = data[2] if len(data) > 2 else None

        for dtype in _TENSOR_DTYPES:
            tol = _TOLERANCE_MAP.get(dtype, {"atol": 1e-5, "rtol": 1e-4})
            input_spec = TensorSpec.from_tensor(shape, in_strides, dtype)

            kwargs = {"p": p}
            test_cases.append(
                TestCase(
                    inputs=[input_spec],
                    kwargs=kwargs,
                    output_spec=None,
                    comparison_target=None,
                    tolerance=tol,
                    description=f"pdist - OUT_OF_PLACE",
                )
            )

    return test_cases


class OpTest(BaseOperatorTest):
    """Pdist operator test with simplified implementation"""

    def __init__(self):
        super().__init__("Pdist")

    def get_test_cases(self):
        return parse_test_cases()

    def torch_operator(self, *args, **kwargs):
        return torch.pdist(*args, **kwargs)

    # def infinicore_operator(self, *args, **kwargs):
    #     """InfiniCore implementation (operator not yet available)."""
    #     return infinicore.pdist(*args, **kwargs)


def main():
    """Main entry point"""
    runner = GenericTestRunner(OpTest)
    runner.run_and_exit()


if __name__ == "__main__":
    main()
