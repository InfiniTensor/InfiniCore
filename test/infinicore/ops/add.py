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

# Test cases format: (shape, a_strides, b_strides, c_strides)
_TEST_CASES_DATA = [
    # Basic cases
    ((13, 4), None, None, None),
    ((13, 4), (10, 1), (10, 1), None),
    # Strided cases
    ((13, 4), None, None, (10, 1)),
    ((13, 4), (10, 1), (10, 1), (10, 1)),
    # 3D cases
    ((13, 4, 4), None, None, None),
    ((13, 4, 4), (20, 4, 1), (20, 4, 1), None),
    # Broadcast cases
    ((13, 4, 4), (4, 0, 1), (0, 4, 1), None),
    # Large tensors
    ((16, 5632), None, None, None),
    ((16, 5632), (13312, 1), (13312, 1), None),
]

# Tolerance configuration
_TOLERANCE_MAP = {
    infinicore.float16: {"atol": 0, "rtol": 1e-2},
    infinicore.float32: {"atol": 0, "rtol": 1e-3},
    infinicore.bfloat16: {"atol": 0, "rtol": 5e-2},
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
        a_strides = data[1] if len(data) > 1 else None
        b_strides = data[2] if len(data) > 2 else None
        c_strides = data[3] if len(data) > 3 else None

        # Check if tensors support in-place operations
        a_supports_inplace = not is_broadcast(a_strides)
        b_supports_inplace = not is_broadcast(b_strides)
        c_supports_inplace = not is_broadcast(c_strides)

        # Generate test cases for all data types
        for dtype in _TENSOR_DTYPES:
            tolerance = _TOLERANCE_MAP.get(dtype, {"atol": 0, "rtol": 1e-3})

            # Create typed tensor specs
            a_spec = TensorSpec.from_tensor(shape, a_strides, dtype, name="a")
            b_spec = TensorSpec.from_tensor(shape, b_strides, dtype, name="b")
            c_spec = TensorSpec.from_tensor(shape, c_strides, dtype, name="c")

            # Test Case 1: Out-of-place (return value)
            test_cases.append(
                TestCase(
                    inputs=[a_spec, b_spec],
                    kwargs={},
                    output_spec=None,
                    comparison_target=None,
                    tolerance=tolerance,
                    description=f"Add - OUT_OF_PLACE",
                )
            )

            # Test Case 2: In-place with explicit output tensor (add(a, b, out=c))
            if c_supports_inplace:
                test_cases.append(
                    TestCase(
                        inputs=[a_spec, b_spec],
                        kwargs=None,
                        output_spec=c_spec,  # Specify the output tensor spec
                        comparison_target="out",
                        tolerance=tolerance,
                        description=f"Add - INPLACE(out)",
                    )
                )

            # Test Case 3: In-place on first input (add(a, b, out=a))
            if a_supports_inplace:
                test_cases.append(
                    TestCase(
                        inputs=[a_spec, b_spec],
                        kwargs={"out": 0},  # Use index 0 for first input
                        output_spec=None,
                        comparison_target=0,  # Compare first input
                        tolerance=tolerance,
                        description=f"Add - INPLACE(a)",
                    )
                )

            # Test Case 4: In-place on second input (add(a, b, out=b))
            if b_supports_inplace:
                test_cases.append(
                    TestCase(
                        inputs=[a_spec, b_spec],
                        kwargs={"out": 1},  # Use index 1 for second input
                        output_spec=None,
                        comparison_target=1,  # Compare second input
                        tolerance=tolerance,
                        description=f"Add - INPLACE(b)",
                    )
                )

    return test_cases


class OpTest(BaseOperatorTest):
    """Add operator test with simplified implementation"""

    def __init__(self):
        super().__init__("Add")

    def get_test_cases(self):
        return parse_test_cases()

    def torch_operator(self, *args, **kwargs):
        """PyTorch add implementation"""
        return torch.add(*args, **kwargs)

    def infinicore_operator(self, *args, **kwargs):
        """InfiniCore add implementation"""
        return infinicore.add(*args, **kwargs)


def main():
    """Main entry point"""
    runner = GenericTestRunner(OpTest)
    runner.run_and_exit()


if __name__ == "__main__":
    main()
