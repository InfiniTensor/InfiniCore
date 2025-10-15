import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import infinicore
from framework import create_test_cases
from framework.base import BaseOperatorTest, TestCase
from framework.runner import GenericTestRunner

# ==============================================================================
# Operator-specific configuration
# ==============================================================================

# Test cases with operation mode as first parameter
# Format: (operation_mode, shape_info, ...)
_TEST_CASES_DATA = [
    (TestCase.BOTH, (13, 4)),
    (TestCase.BOTH, (13, 4), (10, 1), (10, 1), (10, 1)),
    (TestCase.BOTH, (13, 4), (0, 1), None, None),
    (TestCase.BOTH, (13, 4, 4), None, None, None),
    (TestCase.BOTH, (13, 4, 4), (20, 4, 1), (20, 4, 1), (20, 4, 1)),
    (TestCase.BOTH, (13, 4, 4), (4, 0, 1), (0, 4, 1), None),
    (TestCase.BOTH, (16, 5632), None, None, None),
    (TestCase.BOTH, (16, 5632), (13312, 1), (13312, 1), (13312, 1)),
    (TestCase.BOTH, (13, 16, 2), (128, 4, 1), (0, 2, 1), (64, 4, 1)),
    (TestCase.BOTH, (13, 16, 2), (128, 4, 1), (2, 0, 1), (64, 4, 1)),
    (TestCase.BOTH, (4, 4, 5632), None, None, None),
    (TestCase.BOTH, (4, 4, 5632), (45056, 5632, 1), (45056, 5632, 1), (45056, 5632, 1)),
]

# Parameter mapping configuration for add operator
# Format: (a_shape, b_shape, result_shape, a_stride, b_stride, c_stride)
# Call signature: add(input, other)
_ADD_PARAMETER_MAPPING = (
    "add",  # operator_name
    "add(input, other)",  # call_signature
    [  # input_configs
        {"shape": 0, "stride": 1},  # input: shape from index 0, stride from index 1
        {"shape": 0, "stride": 2},  # other: shape from index 0, stride from index 2
    ],
    {"shape": 0, "stride": 3},  # output: shape from index 0, stride from index 3
)

# Parse test cases using add parameter mapping
_TEST_CASES = create_test_cases(_TEST_CASES_DATA, _ADD_PARAMETER_MAPPING)

# Data types
_TENSOR_DTYPES = [infinicore.float16, infinicore.bfloat16, infinicore.float32]

# Tolerance
_TOLERANCE_MAP = {
    infinicore.float16: {"atol": 0, "rtol": 1e-2},
    infinicore.float32: {"atol": 0, "rtol": 1e-3},
    infinicore.bfloat16: {"atol": 0, "rtol": 5e-2},
}

# ==============================================================================
# Operator test class with unified operator functions
# ==============================================================================


class AddTest(BaseOperatorTest):
    """Add test with unified operator functions"""

    def __init__(self):
        super().__init__("add")

    def get_test_cases(self):
        return _TEST_CASES

    def get_tensor_dtypes(self):
        return _TENSOR_DTYPES

    def get_tolerance_map(self):
        return _TOLERANCE_MAP

    def torch_operator(self, a, b, out=None, **kwargs):
        """
        Unified PyTorch add operation - handles both in-place and out-of-place

        Args:
            a: First input tensor
            b: Second input tensor
            out: Optional output tensor for in-place operation
            **kwargs: Additional arguments

        Returns:
            Result tensor for out-of-place, or output tensor for in-place
        """
        return torch.add(a, b, out=out)

    def infinicore_operator(self, a, b, out=None, **kwargs):
        """
        Unified Infinicore add operation - handles both in-place and out-of-place

        Args:
            a: First input tensor
            b: Second input tensor
            out: Optional output tensor for in-place operation
            **kwargs: Additional arguments

        Returns:
            Result tensor for out-of-place, or output tensor for in-place
        """
        return infinicore.add(a, b, out=out)


# ==============================================================================
# Main execution
# ==============================================================================


def main():
    """Main entry point"""
    runner = GenericTestRunner(AddTest)
    runner.run_and_exit()


if __name__ == "__main__":
    main()
