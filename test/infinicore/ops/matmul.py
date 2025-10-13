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

# Test cases: (a_shape, b_shape, result_shape, a_stride, b_stride, c_stride)
# For cases without strides, only provide first 3 elements
_TEST_CASES_DATA = [
    ((2, 3), (3, 4), (2, 4)),
    ((128, 256), (256, 64), (128, 64)),
    ((2, 4, 2048), (2, 2048, 2048), (2, 4, 2048)),
    ((1, 2048), (2048, 2048), (1, 2048), (4096, 1), (4096, 1), (4096, 1)),
    ((6, 2048), (2048, 2560), (6, 2560), (2048, 1), (1, 2048), (2560, 1)),
    ((4, 48, 64), (4, 64, 6), (4, 48, 6)),
]

# Parameter mapping configuration for matmul operator
# Format: (a_shape, b_shape, result_shape, a_stride, b_stride, c_stride)
# Call signature: matmul(a, b)
_MATMUL_PARAMETER_MAPPING = (
    "matmul",  # operator_name
    "matmul(a, b)",  # call_signature
    [  # input_configs
        {"shape": 0, "stride": 3},  # a: shape from index 0, stride from index 3
        {"shape": 1, "stride": 4},  # b: shape from index 1, stride from index 4
    ],
    {"shape": 2, "stride": 5},  # output: shape from index 2, stride from index 5
)

# Parse test cases using matmul parameter mapping
_TEST_CASES = create_test_cases(_TEST_CASES_DATA, _MATMUL_PARAMETER_MAPPING)

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


class MatmulTest(BinaryOperatorTest):
    """Matmul test"""

    def __init__(self):
        super().__init__("matmul", _TEST_CASES, _TENSOR_DTYPES, _TOLERANCE_MAP)


# ==============================================================================
# Main execution
# ==============================================================================


def main():
    """Main entry point"""
    runner = GenericTestRunner(MatmulTest)
    runner.run_and_exit()


if __name__ == "__main__":
    main()
