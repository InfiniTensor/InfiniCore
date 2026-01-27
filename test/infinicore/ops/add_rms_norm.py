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

# Test cases format: (y_shape, a_shape, b_shape, w_shape, y_strides, a_strides, b_strides)
_TEST_CASES_DATA = [
    # Basic cases
    ((1, 4), (1, 4), (1, 4), (4,), None, None, None),
    ((2, 4), (2, 4), (2, 4), (4,), None, None, None),
    ((2, 2, 4), (2, 2, 4), (2, 2, 4), (4,), None, None, None),
    # Strided cases
    ((2, 2, 4), (2, 2, 4), (2, 2, 4), (4,), (12, 8, 1), (12, 8, 1), (12, 8, 1)),
    # Large tensors
    ((16, 2048), (16, 2048), (16, 2048), (2048,), None, None, None),
    ((16, 2048), (16, 2048), (16, 2048), (2048,), (4096, 1), (4096, 1), (4096, 1)),
    ((15, 3584), (15, 3584), (15, 3584), (3584,), None, None, None),
    ((4, 4, 2048), (4, 4, 2048), (4, 4, 2048), (2048,), None, None, None),
    (
        (4, 4, 2048),
        (4, 4, 2048),
        (4, 4, 2048),
        (2048,),
        (2048, 8192, 1),
        (2048, 8192, 1),
        (2048, 8192, 1),
    ),
    (
        (4, 4, 2048),
        (4, 4, 2048),
        (4, 4, 2048),
        (2048,),
        (16384, 4096, 1),
        (16384, 4096, 1),
        (16384, 4096, 1),
    ),
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

# EPSILON constant for AddRMSNorm
_EPSILON = 1e-5


def parse_test_cases():
    """
    Parse AddRMSNorm test case data and return list of TestCase objects.
    Format: (y_shape, a_shape, b_shape, w_shape, y_strides, a_strides, b_strides)
    """
    test_cases = []

    for data in _TEST_CASES_DATA:
        y_shape = data[0]  # Output shape
        a_shape = data[1]  # First input shape
        b_shape = data[2]  # Second input shape
        w_shape = data[3]  # Weight shape (1D)
        y_strides = data[4] if len(data) > 4 else None
        a_strides = data[5] if len(data) > 5 else None
        b_strides = data[6] if len(data) > 6 else None

        # Check if tensors support in-place operations
        a_supports_inplace = not is_broadcast(a_strides)
        b_supports_inplace = not is_broadcast(b_strides)
        y_supports_inplace = not is_broadcast(y_strides)

        # Generate test cases for all dtype combinations
        for input_dtype in _INPUT_DTYPES:
            for weight_dtype in _WEIGHT_DTYPES:
                # Use input dtype tolerance for output
                tolerance = _TOLERANCE_MAP.get(
                    input_dtype, {"atol": 1e-5, "rtol": 1e-4}
                )

                # Create typed tensor specs
                a_spec = TensorSpec.from_tensor(a_shape, a_strides, input_dtype)
                b_spec = TensorSpec.from_tensor(b_shape, b_strides, input_dtype)
                w_spec = TensorSpec.from_tensor(
                    w_shape, None, weight_dtype
                )  # Weight is always contiguous
                y_spec = TensorSpec.from_tensor(y_shape, y_strides, input_dtype)

                # Test Case 1: Out-of-place (return value) - returns (normalized_result, add_result)
                residual_out_spec = TensorSpec.from_tensor(
                    a_shape, a_strides, input_dtype
                )
                test_cases.append(
                    TestCase(
                        inputs=[a_spec, b_spec, w_spec],
                        kwargs={"epsilon": _EPSILON},
                        output_specs=None,  # Two outputs
                        comparison_target=None,
                        tolerance=tolerance,
                        output_count=2,  # Two outputs: normalized_result and add_result
                        description=f"AddRMSNorm - OUT_OF_PLACE",
                    )
                )

                # Test Case 2: In-place with explicit output tensors (add_rms_norm_(y, residual_out, a, b, w))
                # if y_supports_inplace:
                #     residual_out_spec = TensorSpec.from_tensor(
                #         a_shape, a_strides, input_dtype
                #     )
                #     test_cases.append(
                #         TestCase(
                #             inputs=[a_spec, b_spec, w_spec],
                #             kwargs={
                #                 "epsilon": _EPSILON,
                #                 "out": y_spec,
                #                 "residual": residual_out_spec,
                #             },
                #             output_specs=[y_spec, residual_out_spec],  # Two outputs
                #             comparison_target="out",
                #             tolerance=tolerance,
                #             output_count=2,
                #             description=f"AddRMSNorm - INPLACE(out)",
                #         )
                #     )

    return test_cases


class OpTest(BaseOperatorTest):
    """AddRMSNorm operator test with simplified implementation"""

    def __init__(self):
        super().__init__("AddRMSNorm")

    def get_test_cases(self):
        return parse_test_cases()

    def torch_operator(
        self, a, b, weight, epsilon=_EPSILON, out=None, residual=None, **kwargs
    ):
        """PyTorch AddRMSNorm implementation - returns (normalized_result, add_result)"""
        input_dtype = a.dtype

        # Compute add(a, b)
        sum_tensor = a.to(torch.float32) + b.to(torch.float32)
        weight_fp32 = weight.to(torch.float32)

        # Calculate RMSNorm: (a + b) * weight / sqrt(mean((a+b)^2) + epsilon)
        variance = sum_tensor.pow(2).mean(-1, keepdim=True)
        normalized_result = sum_tensor * torch.rsqrt(variance + epsilon) * weight_fp32

        # Convert back to original dtype
        normalized_result = normalized_result.to(input_dtype)
        add_result = sum_tensor.to(input_dtype)

        if out is not None:
            out.copy_(normalized_result)
        if residual is not None:
            residual.copy_(add_result)

        return (normalized_result, add_result)

    def infinicore_operator(
        self, a, b, weight, epsilon=_EPSILON, out=None, residual=None, **kwargs
    ):
        """InfiniCore AddRMSNorm implementation - returns (normalized_result, add_result)"""
        return infinicore.add_rms_norm(
            a, b, weight, epsilon, out=out, residual=residual
        )


def main():
    """Main entry point"""
    runner = GenericTestRunner(OpTest)
    runner.run_and_exit()


if __name__ == "__main__":
    main()
