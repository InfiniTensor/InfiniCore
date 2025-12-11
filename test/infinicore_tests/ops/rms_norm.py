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

# Test cases format: (y_shape, x_shape, w_shape, y_strides, x_strides)
_TEST_CASES_DATA = [
    # Basic cases
    ((1, 4), (1, 4), (4,), None, None),
    ((2, 4), (2, 4), (4,), None, None),
    ((2, 2, 4), (2, 2, 4), (4,), None, None),
    # Strided cases
    ((2, 2, 4), (2, 2, 4), (4,), (12, 8, 1), (12, 8, 1)),
    # Large tensors
    ((16, 2048), (16, 2048), (2048,), None, None),
    ((16, 2048), (16, 2048), (2048,), (4096, 1), (4096, 1)),
]

# Tolerance configuration
_TOLERANCE_MAP = {
    infinicore.float16: {"atol": 2e-3, "rtol": 2e-3},
    infinicore.bfloat16: {"atol": 1e-2, "rtol": 1e-2},
    infinicore.float32: {"atol": 1e-5, "rtol": 1e-4},
}

# Data types for individual tensors
_INPUT_DTYPES = [infinicore.float16, infinicore.bfloat16]
_WEIGHT_DTYPES = [infinicore.float16, infinicore.bfloat16, infinicore.float32]

# EPSILON constant for RMSNorm
_EPSILON = 1e-5


def parse_test_cases():
    """
    Parse RMSNorm test case data and return list of TestCase objects.
    Format: (y_shape, x_shape, w_shape, y_strides, x_strides)
    """
    test_cases = []

    for data in _TEST_CASES_DATA:
        y_shape = data[0]  # Output shape
        x_shape = data[1]  # Input shape
        w_shape = data[2]  # Weight shape (1D)
        y_strides = data[3] if len(data) > 3 else None
        x_strides = data[4] if len(data) > 4 else None

        # Check if tensors support in-place operations
        x_supports_inplace = not is_broadcast(x_strides)
        y_supports_inplace = not is_broadcast(y_strides)

        # Generate test cases for all dtype combinations
        for input_dtype in _INPUT_DTYPES:
            for weight_dtype in _WEIGHT_DTYPES:
                # Use input dtype tolerance for output
                tolerance = _TOLERANCE_MAP.get(
                    input_dtype, {"atol": 1e-5, "rtol": 1e-4}
                )

                # Create typed tensor specs
                x_spec = TensorSpec.from_tensor(x_shape, x_strides, input_dtype)
                w_spec = TensorSpec.from_tensor(
                    w_shape, None, weight_dtype
                )  # Weight is always contiguous
                y_spec = TensorSpec.from_tensor(y_shape, y_strides, input_dtype)

                # Test Case 1: Out-of-place (return value)
                test_cases.append(
                    TestCase(
                        inputs=[x_spec, w_spec],
                        kwargs={"epsilon": _EPSILON},
                        output_spec=None,
                        comparison_target=None,
                        tolerance=tolerance,
                        description=f"RMSNorm - OUT_OF_PLACE",
                    )
                )

                # Test Case 2: In-place with explicit output tensor (rms_norm(x, w, out=y))
                if y_supports_inplace:
                    test_cases.append(
                        TestCase(
                            inputs=[x_spec, w_spec],
                            kwargs={"epsilon": _EPSILON},
                            output_spec=y_spec,  # Specify the output tensor spec
                            comparison_target="out",
                            tolerance=tolerance,
                            description=f"RMSNorm - INPLACE(out)",
                        )
                    )

                # Test Case 3: In-place on input tensor (rms_norm(x, w, out=x))
                if x_supports_inplace:
                    test_cases.append(
                        TestCase(
                            inputs=[x_spec, w_spec],
                            kwargs={
                                "out": 0,
                                "epsilon": _EPSILON,
                            },  # Use index 0 for first input
                            output_spec=None,
                            comparison_target=0,  # Compare first input
                            tolerance=tolerance,
                            description=f"RMSNorm - INPLACE(x)",
                        )
                    )

    return test_cases


class OpTest(BaseOperatorTest):
    """RMSNorm operator test with simplified implementation"""

    def __init__(self):
        super().__init__("RMSNorm")

    def get_test_cases(self):
        return parse_test_cases()

    def torch_operator(self, x, weight, epsilon=_EPSILON, out=None, **kwargs):
        """PyTorch RMSNorm implementation"""
        input_dtype = x.dtype

        # Convert to float32 for numerical stability
        hidden_states = x.to(torch.float32)
        weight_fp32 = weight.to(torch.float32)

        # Calculate RMSNorm: x * weight / sqrt(mean(x^2) + epsilon)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        result = hidden_states * torch.rsqrt(variance + epsilon) * weight_fp32

        # Convert back to original dtype
        result = result.to(input_dtype)

        if out is not None:
            out.copy_(result)
            return out
        return result

    def infinicore_operator(self, x, weight, epsilon=_EPSILON, out=None, **kwargs):
        """InfiniCore RMSNorm implementation"""
        import infinicore.nn.functional as F

        return F.rms_norm(x, weight.shape, weight, epsilon, out=out)


def main():
    """Main entry point"""
    runner = GenericTestRunner(OpTest)
    runner.run_and_exit()


if __name__ == "__main__":
    main()
