import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import infinicore
from framework.base import BaseOperatorTest, TensorSpec, TestCase
from framework.runner import GenericTestRunner

# Test cases format: (input_shape, input_strides_or_None, split_size_or_sections, dim_or_None)
# infinicore.split(tensor, split_size_or_sections, dim=0)

_TEST_CASES_DATA = [
    ((8, 6), None, 2, 0),
    ((4, 9), None, [3, 6], 1),
    ((6, 12, 3), None, 4, 1),
    ((10,), None, 5, 0),
    ((3, 8), (24, 8), [2, 1], 0),
    ((12, 4), None, 3, 0),
]

_TOLERANCE_MAP = {
    infinicore.float16: {"atol": 1e-3, "rtol": 1e-2},
    infinicore.float32: {"atol": 1e-5, "rtol": 1e-4},
    infinicore.bfloat16: {"atol": 1e-2, "rtol": 5e-2},
}

_TENSOR_DTYPES = [infinicore.float16, infinicore.bfloat16, infinicore.float32]


def parse_test_cases():
    test_cases = []
    for shape, strides, sections, dim in _TEST_CASES_DATA:
        for dtype in _TENSOR_DTYPES:
            tol = _TOLERANCE_MAP.get(dtype, {"atol": 1e-5, "rtol": 1e-4})
            inp = TensorSpec.from_tensor(shape, strides, dtype)
            kwargs = {}
            kwargs["split_size_or_sections"] = sections
            if dim is not None:
                kwargs["dim"] = dim

            test_cases.append(
                TestCase(
                    inputs=[inp],
                    kwargs=kwargs,
                    output_spec=None,
                    comparison_target=None,
                    tolerance=tol,
                    description="split - OUT_OF_PLACE",
                )
            )

    return test_cases


class OpTest(BaseOperatorTest):
    """split operator test with simplified implementation"""

    def __init__(self):
        super().__init__("split")

    def get_test_cases(self):
        return parse_test_cases()

    def torch_operator(self, *args, **kwargs):
        # infinicore.split signature differs; test runner will map kwargs accordingly
        return torch.split(*args, **kwargs)

    # def infinicore_operator(self, *args, **kwargs):
    #     """InfiniCore implementation (operator not yet available)."""
    #     return infinicore.split(*args, **kwargs)


def main():
    """Main entry point"""
    runner = GenericTestRunner(OpTest)
    runner.run_and_exit()


if __name__ == "__main__":
    main()
