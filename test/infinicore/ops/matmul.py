import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import infinicore
from framework import (
    BaseOperatorTest,
    TensorSpec,
    TestCase,
    GenericTestRunner,
    is_broadcast,
)

# ==============================================================================
# Operator-specific configuration
# ==============================================================================

# Test cases format: (nbatch, m, n, k, a_strides, b_strides, c_strides)
# If nbatch is None: a_shape=(m, k), b_shape=(k, n), c_shape=(m, n)
# If nbatch is provided: a_shape=(nbatch, m, k), b_shape=(nbatch, k, n), c_shape=(nbatch, m, n)
_TEST_CASES_DATA = [
    # Basic 2D matmul
    (None, 2, 4, 3, None, None, None),
    (None, 128, 64, 256, None, None, None),
    # Batched matmul
    (2, 4, 2048, 2048, None, None, None),
    (4, 48, 6, 64, None, None, None),
    # Strided tensors
    (None, 1, 2048, 2048, (4096, 1), (4096, 1), (4096, 1)),
    (None, 6, 2560, 2048, (2048, 1), (1, 2048), (2560, 1)),
    # Mixed cases
    (8, 16, 32, 16, None, None, None),
]

# Tolerance configuration
_TOLERANCE_MAP = {
    infinicore.float16: {"atol": 0, "rtol": 1e-2},
    infinicore.float32: {"atol": 1e-4, "rtol": 1e-3},
    infinicore.bfloat16: {"atol": 0, "rtol": 5e-2},
}

# Data types to test
_TENSOR_DTYPES = [infinicore.float16, infinicore.bfloat16, infinicore.float32]


def parse_test_cases():
    """
    Parse test case data and return list of TestCase objects for matmul operation.
    Each test case contains all necessary information for execution and validation.
    """
    test_cases = []

    for data in _TEST_CASES_DATA:
        nbatch = data[0]
        m, n, k = data[1], data[2], data[3]
        a_strides = data[4] if len(data) > 4 else None
        b_strides = data[5] if len(data) > 5 else None
        c_strides = data[6] if len(data) > 6 else None

        # Determine shapes based on batch dimension
        if nbatch is None:
            a_shape = (m, k)
            b_shape = (k, n)
            c_shape = (m, n)
        else:
            a_shape = (nbatch, m, k)
            b_shape = (nbatch, k, n)
            c_shape = (nbatch, m, n)

        # Check if tensors support in-place operations
        c_supports_inplace = not is_broadcast(c_strides)

        # Generate test cases for all data types
        for dtype in _TENSOR_DTYPES:
            tolerance = _TOLERANCE_MAP.get(dtype, {"atol": 0, "rtol": 1e-3})

            # Create typed tensor specs
            a_spec = TensorSpec.from_tensor(a_shape, a_strides, dtype)
            b_spec = TensorSpec.from_tensor(b_shape, b_strides, dtype)
            c_spec = TensorSpec.from_tensor(c_shape, c_strides, dtype)

            # Test Case 1: Out-of-place (return value)
            test_cases.append(
                TestCase(
                    inputs=[a_spec, b_spec],
                    kwargs={},
                    output_spec=None,
                    comparison_target=None,
                    tolerance=tolerance,
                    description=f"Matmul - OUT_OF_PLACE",
                )
            )

            # Test Case 2: In-place with explicit output tensor (matmul(a, b, out=c))
            if c_supports_inplace:
                test_cases.append(
                    TestCase(
                        inputs=[a_spec, b_spec],
                        kwargs=None,
                        output_spec=c_spec,  # Specify the output tensor spec
                        comparison_target="out",
                        tolerance=tolerance,
                        description=f"Matmul - INPLACE(out)",
                    )
                )

    return test_cases


class OpTest(BaseOperatorTest):
    """Matmul operator test with simplified implementation"""

    def __init__(self):
        super().__init__("Matmul")

    def get_test_cases(self):
        return parse_test_cases()

    def torch_operator(self, *args, **kwargs):
        """PyTorch matmul implementation"""
        return torch.matmul(*args, **kwargs)

    def infinicore_operator(self, *args, **kwargs):
        """InfiniCore matmul implementation"""
        return infinicore.matmul(*args, **kwargs)


def main():
    """Main entry point"""
    runner = GenericTestRunner(OpTest)
    runner.run_and_exit()


if __name__ == "__main__":
    main()
