import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import infinicore
from framework.base import BaseOperatorTest, TensorSpec, TestCase
from framework.runner import GenericTestRunner

# Test cases format: (input_shape, input_strides_or_None, bins_or_sequence, range_or_None)
_TEST_CASES_DATA = [
    ((100,), None, 10, (0.0, 1.0)),
    ((20,), None, 5, (0.0, 2.0)),
    ((10,), None, 4, (0.0, 1.0)),
    ((200,), None, 20, (-1.0, 1.0)),
    ((1,), None, 3, (0.0, 1.0)),
]

_TOLERANCE_MAP = {infinicore.float32: {"atol": 1e-5, "rtol": 1e-4}}
_TENSOR_DTYPES = [infinicore.float32]


def parse_test_cases():
    test_cases = []
    for data in _TEST_CASES_DATA:
        shape, strides, bins, rng = data

        for dtype in _TENSOR_DTYPES:
            tol = _TOLERANCE_MAP.get(dtype)
            input_spec = TensorSpec.from_tensor(shape, strides, dtype)

            kwargs = {"bins": bins}
            if rng is not None:
                kwargs["range"] = rng

            test_cases.append(
                TestCase(
                    inputs=[input_spec],
                    kwargs=kwargs,
                    output_spec=None,
                    comparison_target=None,
                    tolerance=tol,
                    description=f"histogram - OUT_OF_PLACE",
                )
            )

    return test_cases


class OpTest(BaseOperatorTest):
    """Histogram operator test with simplified implementation"""

    def __init__(self):
        super().__init__("Histogram")

    def get_test_cases(self):
        return parse_test_cases()

    def torch_operator(self, *args, **kwargs):
        return torch.histogram(*args, **kwargs)

    # def infinicore_operator(self, *args, **kwargs):
    #     """InfiniCore implementation (operator not yet available)."""
    #     return infinicore.histogram(*args, **kwargs)


def main():
    """Main entry point"""
    runner = GenericTestRunner(OpTest)
    runner.run_and_exit()


if __name__ == "__main__":
    main()
