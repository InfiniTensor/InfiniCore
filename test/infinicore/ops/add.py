import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import infinicore
from framework import create_test_cases
from framework.base import BaseOperatorTest
from framework.runner import GenericTestRunner

# ==============================================================================
# Operator-specific configuration
# ==============================================================================

# Test cases in flexible format
_TEST_CASES_DATA = [
    ((13, 4)),
    ((13, 4), (10, 1), (10, 1), (10, 1)),
    ((13, 4), (0, 1), None, None),
    ((13, 4, 4), None, None, None),
    ((13, 4, 4), (20, 4, 1), (20, 4, 1), (20, 4, 1)),
    ((13, 4, 4), (4, 0, 1), (0, 4, 1), None),
    ((16, 5632), None, None, None),
    ((16, 5632), (13312, 1), (13312, 1), (13312, 1)),
    ((13, 16, 2), (128, 4, 1), (0, 2, 1), (64, 4, 1)),
    ((13, 16, 2), (128, 4, 1), (2, 0, 1), (64, 4, 1)),
    ((4, 4, 5632), None, None, None),
    ((4, 4, 5632), (45056, 5632, 1), (45056, 5632, 1), (45056, 5632, 1)),
]

# Parameter mapping configuration for add operator
_ADD_PARAMETER_MAPPING = (
    "add",
    "add(input, other)",
    [
        {"shape": 0, "stride": 1},
        {"shape": 0, "stride": 2},
    ],
    {"shape": 0, "stride": 3},
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
# Operator test class with specific test functions
# ==============================================================================


class AddTest(BaseOperatorTest):
    """Add test with operator-specific test functions"""

    def __init__(self):
        super().__init__("add")

    def get_test_cases(self):
        return _TEST_CASES

    def get_tensor_dtypes(self):
        return _TENSOR_DTYPES

    def get_tolerance_map(self):
        return _TOLERANCE_MAP

    def torch_operator_inplace(self, a, b, out=None, **kwargs):
        """PyTorch in-place add operation"""
        torch.add(a, b, out=out)

    def infinicore_operator_inplace(self, a, b, out=None, **kwargs):
        """Infinicore in-place add operation"""
        infinicore.add(a, b, out=out)

    def torch_operator_out_of_place(self, a, b, **kwargs):
        """PyTorch out-of-place add operation"""
        return torch.add(a, b)

    def infinicore_operator_out_of_place(self, a, b, **kwargs):
        """Infinicore out-of-place add operation"""
        return infinicore.add(a, b)


# ==============================================================================
# Main execution
# ==============================================================================


def main():
    """Main entry point"""
    runner = GenericTestRunner(AddTest)
    runner.run_and_exit()


if __name__ == "__main__":
    main()
