import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import infinicore
from framework.base import BaseOperatorTest, TensorSpec, TestCase
from framework.runner import GenericTestRunner

# Test cases format: (input_shape, input_strides_or_None, offset_or_None, dim1_or_None, dim2_or_None)
# infinicore.diagonal(input, offset=0, dim1=0, dim2=1)

_TEST_CASES_DATA = [
    ((3, 4), None, None, None, None),
    ((5, 5), (300, 60), 1, None, None),
    ((2, 3, 3), None, 0, -2, -1),
    ((4, 4), None, -1, None, None),
    ((1, 6), None, None, None, None),
    ((8, 8), (512, 1), 2, None, None),
]

_TOLERANCE_MAP = {
    infinicore.float16: {"atol": 1e-3, "rtol": 1e-2},
    infinicore.float32: {"atol": 1e-5, "rtol": 1e-4},
    infinicore.bfloat16: {"atol": 1e-2, "rtol": 5e-2},
}

_TENSOR_DTYPES = [infinicore.float16, infinicore.bfloat16, infinicore.float32]


def parse_test_cases():
    test_cases = []
    for shape, strides, offset, d1, d2 in _TEST_CASES_DATA:
        for dtype in _TENSOR_DTYPES:
            tol = _TOLERANCE_MAP.get(dtype, {"atol": 1e-5, "rtol": 1e-4})
            spec = TensorSpec.from_tensor(shape, strides, dtype)
            kwargs = {}
            if offset is not None:
                kwargs["offset"] = offset
            if d1 is not None:
                kwargs["dim1"] = d1
            if d2 is not None:
                kwargs["dim2"] = d2

            test_cases.append(
                TestCase(
                    inputs=[spec],
                    kwargs=kwargs,
                    output_spec=None,
                    comparison_target=None,
                    tolerance=tol,
                    description="diagonal - OUT_OF_PLACE",
                )
            )

    return test_cases


class OpTest(BaseOperatorTest):
    """diagonal operator test with simplified implementation"""

    def __init__(self):
        super().__init__("diagonal")

    def get_test_cases(self):
        return parse_test_cases()

    def torch_operator(self, *args, **kwargs):
        return torch.diagonal(*args, **kwargs)

    # def infinicore_operator(self, *args, **kwargs):
    #     """InfiniCore implementation (operator not yet available)."""
    #     return infinicore.diagonal(*args, **kwargs)


def main():
    """Main entry point"""
    runner = GenericTestRunner(OpTest)
    runner.run_and_exit()


if __name__ == "__main__":
    main()
