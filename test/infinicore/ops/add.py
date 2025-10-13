import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import infinicore
from framework import create_test_cases
from framework.templates import BinaryOperatorTest
from framework.runner import GenericTestRunner

# ==============================================================================
# Operator-specific configuration
# ==============================================================================

# Test cases in flexible format:
# - Single shape tuple: (13, 4) → automatically expands to ((13, 4), None, None, None)
# - Nested single shape: ((13, 4),) → automatically expands to ((13, 4), None, None, None)
# - Full format: ((13, 4), None, None, None) or ((13, 4), (10, 1), (10, 1), (10, 1))
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
# Format: (shape, a_stride, b_stride, c_stride)
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
# Operator test class
# ==============================================================================


class AddTest(BinaryOperatorTest):
    """Add test"""

    def __init__(self):
        super().__init__("add", _TEST_CASES, _TENSOR_DTYPES, _TOLERANCE_MAP)


# ==============================================================================
# Main execution
# ==============================================================================


def main():
    """Main entry point"""
    runner = GenericTestRunner(AddTest)
    runner.run_and_exit()


if __name__ == "__main__":
    main()
