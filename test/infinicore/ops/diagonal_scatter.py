import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import infinicore
from framework.base import BaseOperatorTest, TensorSpec, TestCase
from framework.runner import GenericTestRunner
from framework.utils import is_broadcast

# ===============================================================================
# Operator-specific configuration
# ===============================================================================

# Test cases format: (shape, input_strides, src_strides_or_None, offset)
# diagonal_scatter writes values from src into input along diagonals specified by offset
_TEST_CASES_DATA = [
    ((6, 6), None, None, 0),
    ((8, 8), (16, 1), None, 1),
    ((7, 5), None, (10, 1), -1),
    ((4, 9), None, None, 2),
    ((10, 10), (20, 1), (20, 1), 0),
    ((3, 5), None, None, -2),
]
# Test cases format: (shape, input_strides_or_None, src_strides_or_None, offset, optional_dim1, optional_dim2)
_TEST_CASES_DATA = [
    ((6, 6), None, None, 0, 0, 1),
    ((8, 8), (16, 1), None, 1, 0, 1),
    ((7, 5), None, (4,), -1, 0, 1),
    ((4, 9), None, None, 2, 0, 1),
    ((10, 10), (20, 1), (2,), 0, 0, 1),
    ((3, 5), None, None, -2, 0, 1),
]

# Tolerance configuration
_TOLERANCE_MAP = {
    infinicore.float16: {"atol": 0, "rtol": 1e-2},
    infinicore.float32: {"atol": 0, "rtol": 1e-3},
    infinicore.bfloat16: {"atol": 0, "rtol": 5e-2},
}

# Data types to test for payload tensors
_TENSOR_DTYPES = [infinicore.float16, infinicore.bfloat16, infinicore.float32]


def parse_test_cases():
    """
    Parse diagonal_scatter test cases.
    Format: (shape, input_strides, index_strides, src_strides, offset)
    """
    test_cases = []

    for data in _TEST_CASES_DATA:
        shape, in_strides, src_strides, offset, dim1, dim2 = data

        # Determine in-place support by checking if input/src are broadcast
        in_supports_inplace = not is_broadcast(in_strides)
        src_supports_inplace = not is_broadcast(src_strides)

        for dtype in _TENSOR_DTYPES:
            tolerance = _TOLERANCE_MAP.get(dtype, {"atol": 0, "rtol": 1e-3})

            input_spec = TensorSpec.from_tensor(shape, in_strides, dtype)

            dummy = torch.zeros(*shape)
            diag = torch.diagonal(dummy, offset=offset, dim1=dim1, dim2=dim2)
            diag_len = diag.numel()
            src_shape = (diag_len,)
            src_spec = TensorSpec.from_tensor(src_shape, src_strides, dtype)

            # Out-of-place (return value)
            test_cases.append(
                TestCase(
                    inputs=[input_spec, src_spec],
                    kwargs={"offset": offset, "dim1": dim1, "dim2": dim2},
                    output_spec=None,
                    comparison_target=None,
                    tolerance=tolerance,
                    description=f"diagonal_scatter - OUT_OF_PLACE",
                )
            )

            # In-place on input (modify input)
            if in_supports_inplace:
                test_cases.append(
                    TestCase(
                        inputs=[input_spec, src_spec],
                        kwargs={"offset": offset, "dim1": dim1, "dim2": dim2},
                        output_spec=None,
                        comparison_target=0,
                        tolerance=tolerance,
                        description=f"diagonal_scatter - INPLACE(input)",
                    )
                )

    return test_cases


class OpTest(BaseOperatorTest):
    """DiagonalScatter operator test with simplified implementation"""

    def __init__(self):
        super().__init__("DiagonalScatter")

    def get_test_cases(self):
        return parse_test_cases()

    def torch_operator(self, *args, **kwargs):
        return torch.diagonal_scatter(*args, **kwargs)

    # def infinicore_operator(self, *args, **kwargs):
    #     """InfiniCore implementation (operator not yet available)."""
    #     return infinicore.diagonal_scatter(*args, **kwargs)


def main():
    """Main entry point"""
    runner = GenericTestRunner(OpTest)
    runner.run_and_exit()


if __name__ == "__main__":
    main()
