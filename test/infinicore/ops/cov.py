import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import infinicore
from framework.base import BaseOperatorTest, TensorSpec, TestCase
from framework.runner import GenericTestRunner

# Test cases format: (input_shape, input_strides_or_None, correction, fweights, aweights)
_TEST_CASES_DATA = [
    ((5,), None, 0, None, None),
    ((3, 5), None, 1, None, None),
    ((4, 4), (16, 1), 0, None, None),
    ((2, 8), None, 1, None, None),
    ((6, 6), None, 0, None, None),
    ((1, 7), None, 0, None, None),
]


_TOLERANCE_MAP = {infinicore.float32: {"atol": 1e-5, "rtol": 1e-4}}
_TENSOR_DTYPES = [infinicore.float32]


def parse_test_cases():
    test_cases = []
    for shape, strides, correction, fweights, aweights in _TEST_CASES_DATA:
        input_spec = TensorSpec.from_tensor(shape, strides, infinicore.float32)
        kwargs = {"correction": correction}
        if fweights is not None:
            kwargs["fweights"] = fweights
        if aweights is not None:
            kwargs["aweights"] = aweights

        for dtype in _TENSOR_DTYPES:
            tol = _TOLERANCE_MAP.get(dtype)

            test_cases.append(
                TestCase(
                    inputs=[input_spec],
                    kwargs=kwargs,
                    output_spec=None,
                    comparison_target=None,
                    tolerance=tol,
                    description=f"cov - OUT_OF_PLACE",
                )
            )

    return test_cases


class OpTest(BaseOperatorTest):
    """Cov operator test with simplified implementation"""

    def __init__(self):
        super().__init__("Cov")

    def get_test_cases(self):
        return parse_test_cases()

    def torch_operator(self, *args, **kwargs):
        return torch.cov(*args, **kwargs)

    # def infinicore_operator(self, *args, **kwargs):
    #     """InfiniCore implementation (operator not yet available)."""
    #     return infinicore.cov(*args, **kwargs)


def main():
    """Main entry point"""
    runner = GenericTestRunner(OpTest)
    runner.run_and_exit()


if __name__ == "__main__":
    main()
