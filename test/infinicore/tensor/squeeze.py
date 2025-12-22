import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import infinicore
from framework.base import BaseOperatorTest, TensorSpec, TestCase
from framework.runner import GenericTestRunner
from framework.utils import is_broadcast

# ==============================================================================
# Operator-specific configuration
# ==============================================================================

# Test cases format: (shape, strides, dim)
_TEST_CASES_DATA = [
    # Basic cases
    ((1, 1, 1), None, 1),
    ((1, 1, 1), None, 0),
    ((1, 2, 4), None, 0),
    ((2, 1, 4), (4, 0, 1), 1),
    ((1, 4, 1, 32), (32, 32, 32, 1), 2),
]

# Tolerance configuration
_TOLERANCE_MAP = {
    infinicore.float16: {"atol": 0, "rtol": 0},
    infinicore.float32: {"atol": 0, "rtol": 0},
    infinicore.bfloat16: {"atol": 0, "rtol": 0},
}

# Data types to test
_TENSOR_DTYPES = [infinicore.float16, infinicore.bfloat16, infinicore.float32]


def parse_test_cases():
    """
    Parse test case data and return list of TestCase objects for all operation types.
    Each test case contains all necessary information for execution and validation.
    """
    test_cases = []

    for data in _TEST_CASES_DATA:
        shape = data[0]
        strides = data[1]
        dim = data[2]

        # Generate test cases for all data types
        for dtype in _TENSOR_DTYPES:
            tolerance = _TOLERANCE_MAP.get(dtype, {"atol": 0, "rtol": 0})

            # Create typed tensor specs
            a_spec = TensorSpec.from_tensor(shape, strides, dtype)
            test_cases.append(
                TestCase(
                    inputs=[a_spec, dim],
                    kwargs={},
                    output_spec=None,
                    comparison_target=None,  # Compare output
                    tolerance=tolerance,
                    description=f"squeeze",
                )
            )

    return test_cases


class OpTest(BaseOperatorTest):
    """squeeze operator test with simplified implementation"""

    def __init__(self):
        super().__init__("squeeze")

    def get_test_cases(self):
        return parse_test_cases()

    def torch_operator(self, *args, **kwargs):
        """PyTorch squeeze implementation"""
        return torch.squeeze(*args, **kwargs)

    def infinicore_operator(self, *args, **kwargs):
        """InfiniCore squeeze implementation"""
        return infinicore.squeeze(*args, **kwargs)


def main():
    """Main entry point"""
    runner = GenericTestRunner(OpTest)
    runner.run_and_exit()


if __name__ == "__main__":
    main()
