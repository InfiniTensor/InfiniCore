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

# Test cases format: (shape, a_strides, b_strides, c_strides)
# SwiGLU operates element-wise on two tensors of the same shape: output = a * b * sigmoid(b)
_TEST_CASES_DATA = [
    # Basic 2D SwiGLU
    ((2, 4), None, None, None),
    ((128, 64), None, None, None),
    # 3D SwiGLU
    ((2, 4, 8), None, None, None),
    ((4, 48, 6), None, None, None),
    # Strided tensors
    ((1, 2048), (4096, 1), (4096, 1), (4096, 1)),
    ((6, 2560), (2048, 1), (1, 2048), (2560, 1)),
    # Mixed cases
    ((8, 16, 32), None, None, None),
    # Large tensors
    ((16, 5632), None, None, None),
    ((4, 4, 5632), None, None, None),
]

# Tolerance configuration
_TOLERANCE_MAP = {
    infinicore.float16: {"atol": 1e-3, "rtol": 1e-3},
    infinicore.float32: {"atol": 1e-5, "rtol": 1e-5},
    infinicore.bfloat16: {"atol": 5e-3, "rtol": 1e-2},
}

# Data types to test
_TENSOR_DTYPES = [infinicore.float16, infinicore.bfloat16, infinicore.float32]


def parse_test_cases():
    """
    Parse SwiGLU test case data according to format:
    (shape, a_strides, b_strides, c_strides)
    SwiGLU is a two-input operation: output = a * b * sigmoid(b)
    """
    test_cases = []

    for data in _TEST_CASES_DATA:
        shape = data[0]
        a_strides = data[1] if len(data) > 1 else None
        b_strides = data[2] if len(data) > 2 else None
        c_strides = data[3] if len(data) > 3 else None

        # Check if tensors support in-place operations
        a_supports_inplace = not is_broadcast(a_strides) and a_strides == b_strides
        b_supports_inplace = not is_broadcast(b_strides) and a_strides == b_strides
        c_supports_inplace = not is_broadcast(c_strides)

        # Generate test cases for all data types
        for dtype in _TENSOR_DTYPES:
            tolerance = _TOLERANCE_MAP.get(dtype, {"atol": 1e-5, "rtol": 1e-4})

            # Create typed tensor specs
            a_spec = TensorSpec.from_tensor(shape, a_strides, dtype)
            b_spec = TensorSpec.from_tensor(shape, b_strides, dtype)
            c_spec = TensorSpec.from_tensor(shape, c_strides, dtype)

            # Test Case 1: Out-of-place (return value)
            test_cases.append(
                TestCase(
                    inputs=[a_spec, b_spec],
                    kwargs={},
                    output_spec=None,
                    comparison_target=None,
                    tolerance=tolerance,
                    description=f"SwiGLU - OUT_OF_PLACE",
                )
            )

            # Test Case 2: In-place with explicit output tensor (swiglu(a, b, out=c))
            if c_supports_inplace:
                test_cases.append(
                    TestCase(
                        inputs=[a_spec, b_spec],
                        kwargs=None,
                        output_spec=c_spec,  # Specify the output tensor spec
                        comparison_target="out",
                        tolerance=tolerance,
                        description=f"SwiGLU - INPLACE(out)",
                    )
                )

            # Test Case 3: In-place on first input (swiglu(a, b, out=a))
            if a_supports_inplace:
                test_cases.append(
                    TestCase(
                        inputs=[a_spec, b_spec],
                        kwargs={"out": 0},  # Use index 0 for first input
                        output_spec=None,
                        comparison_target=0,  # Compare first input
                        tolerance=tolerance,
                        description=f"SwiGLU - INPLACE(a)",
                    )
                )

            # Test Case 4: In-place on second input (swiglu(a, b, out=b))
            if b_supports_inplace:
                test_cases.append(
                    TestCase(
                        inputs=[a_spec, b_spec],
                        kwargs={"out": 1},  # Use index 1 for second input
                        output_spec=None,
                        comparison_target=1,  # Compare second input
                        tolerance=tolerance,
                        description=f"SwiGLU - INPLACE(b)",
                    )
                )

    return test_cases


class OpTest(BaseOperatorTest):
    """SwiGLU operator test with simplified implementation"""

    def __init__(self):
        super().__init__("SwiGLU")

    def get_test_cases(self):
        return parse_test_cases()

    def torch_operator(self, a, b, out=None, **kwargs):
        """PyTorch SwiGLU implementation: a * b * sigmoid(b)"""
        sigmoid_b = torch.sigmoid(b)
        result = a * b * sigmoid_b

        if out is not None:
            out.copy_(result)
            return out
        return result

    def infinicore_operator(self, a, b, out=None, **kwargs):
        """InfiniCore SwiGLU implementation"""
        import infinicore.nn.functional as F

        return F.swiglu(a, b, out=out)


def main():
    """Main entry point"""
    runner = GenericTestRunner(OpTest)
    runner.run_and_exit()


if __name__ == "__main__":
    main()
