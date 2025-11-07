import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import infinicore
from framework.base import BaseOperatorTest, TensorSpec, TestCase
from framework.runner import GenericTestRunner
from framework.utils import is_broadcast

# ==============================================================================
# Operator-specific configuration for aminmax
# ==============================================================================

# Test cases format: (shape, dim, keepdim, input_strides, min_strides, max_strides)
_TEST_CASES_DATA = [
    # Basic cases - out-of-place
    ((13, 4), None, False, None, None, None),
    ((13, 4), 0, False, None, None, None),
    ((13, 4), 1, False, None, None, None),
    ((13, 4), -1, False, None, None, None),
    # With keepdim - out-of-place
    ((13, 4), None, True, None, None, None),
    ((13, 4), 0, True, None, None, None),
    ((13, 4), 1, True, None, None, None),
    # 3D cases - out-of-place
    ((4, 5, 6), None, False, None, None, None),
    ((4, 5, 6), 1, False, None, None, None),
    ((4, 5, 6), 1, True, None, None, None),
    ((4, 5, 6), -1, True, None, None, None),
    # Edge cases - out-of-place
    ((10,), None, False, None, None, None),
    ((10,), 0, False, None, None, None),
    ((1, 5), None, False, None, None, None),
    # In-place cases with strided tensors
    (
        (13, 4),
        None,
        False,
        (10, 1),
        None,
        None,
    ),  # Global min/max - no strides for scalar outputs
    ((13, 4), 0, False, None, (3,), (3,)),
    ((13, 4), 1, False, (20, 1), (10,), (10,)),
    # 3D in-place cases
    ((4, 5, 6), 1, True, None, (4, 1, 6), (4, 1, 6)),
    ((4, 5, 6), -1, False, (30, 6, 1), (4, 5), (4, 5)),
]

# Tolerance configuration
_TOLERANCE_MAP = {
    infinicore.float16: {"atol": 1e-3, "rtol": 1e-2},
    infinicore.float32: {"atol": 1e-5, "rtol": 1e-4},
    infinicore.bfloat16: {"atol": 1e-2, "rtol": 5e-2},
}

# Data types to test
_TENSOR_DTYPES = [infinicore.float16, infinicore.bfloat16, infinicore.float32]


def calculate_output_shape(input_shape, dim, keepdim):
    """
    Calculate the output shape for aminmax operation based on input shape, dim, and keepdim
    """
    if dim is None:
        # Global min/max - output should be scalar tensors
        if keepdim:
            # When keepdim=True with dim=None, output has same rank but all dimensions are 1
            return tuple(1 for _ in input_shape)
        else:
            # Scalar tensors
            return ()
    else:
        # Reduction along specific dimension
        output_shape = list(input_shape)
        if keepdim:
            output_shape[dim] = 1
        else:
            output_shape.pop(dim)
        return tuple(output_shape)


def parse_test_cases():
    """
    Parse aminmax test cases including both out-of-place and in-place variants
    aminmax supports: torch.aminmax(input, *, dim=None, keepdim=False, out=(min_tensor, max_tensor))
    """
    test_cases = []

    for data in _TEST_CASES_DATA:
        shape = data[0]
        dim = data[1] if len(data) > 1 else None
        keepdim = data[2] if len(data) > 2 else False
        input_strides = data[3] if len(data) > 3 else None
        min_strides = data[4] if len(data) > 4 else None
        max_strides = data[5] if len(data) > 5 else None

        # Generate test cases for all data types
        for dtype in _TENSOR_DTYPES:
            tolerance = _TOLERANCE_MAP.get(dtype, {"atol": 1e-5, "rtol": 1e-4})

            # Create input tensor spec
            input_spec = TensorSpec.from_tensor(shape, input_strides, dtype)

            # Build description
            description_parts = ["aminmax"]
            if dim is not None:
                description_parts.append(f"dim={dim}")
            if keepdim:
                description_parts.append("keepdim=True")
            if input_strides is not None:
                description_parts.append(f"input_strides={input_strides}")

            base_description = " - ".join(description_parts)

            # Prepare common kwargs
            kwargs = {}
            if dim is not None:
                kwargs["dim"] = dim
            kwargs["keepdim"] = keepdim

            # ==================================================================
            # Test Case 1: Out-of-place (return values)
            # ==================================================================
            test_cases.append(
                TestCase(
                    inputs=[input_spec],
                    kwargs=kwargs,
                    output_spec=None,  # No output spec for return value comparison
                    comparison_target=None,  # Compare return values
                    tolerance=tolerance,
                    description=f"{base_description} - OUT_OF_PLACE",
                    output_count=2,  # aminmax returns 2 tensors: (min, max)
                )
            )

            # ==================================================================
            # Test Case 2: In-place with explicit output tensors
            # ==================================================================
            # Only create in-place test cases if we have valid output configurations
            # For global min/max (dim=None), we need special handling
            if dim is None:
                # Global min/max - output shapes are either () or (1,1,...) depending on keepdim
                output_shape = calculate_output_shape(shape, dim, keepdim)

                # For scalar outputs, we don't use strides (they would be empty tuples)
                if output_shape == ():
                    # Scalar tensors - create without strides
                    min_spec = TensorSpec.from_tensor(output_shape, None, dtype)
                    max_spec = TensorSpec.from_tensor(output_shape, None, dtype)
                else:
                    # keepdim=True case - use provided strides or None
                    min_spec = TensorSpec.from_tensor(output_shape, min_strides, dtype)
                    max_spec = TensorSpec.from_tensor(output_shape, max_strides, dtype)

                # Check if output tensors support in-place operations
                min_supports_inplace = not is_broadcast(
                    getattr(min_spec, "strides", None)
                )
                max_supports_inplace = not is_broadcast(
                    getattr(max_spec, "strides", None)
                )

                if min_supports_inplace and max_supports_inplace:
                    inplace_kwargs = kwargs.copy()

                    test_cases.append(
                        TestCase(
                            inputs=[input_spec],
                            kwargs=inplace_kwargs,
                            output_specs=[
                                min_spec,
                                max_spec,
                            ],  # Multiple output specs for in-place
                            comparison_target="out",  # Compare the output tuple from kwargs
                            tolerance=tolerance,
                            description=f"{base_description} - INPLACE(out)",
                            output_count=2,  # Specify 2 outputs
                        )
                    )

            else:
                # Reduction along specific dimension
                if min_strides is not None and max_strides is not None:
                    output_shape = calculate_output_shape(shape, dim, keepdim)

                    # Create output tensor specs
                    min_spec = TensorSpec.from_tensor(output_shape, min_strides, dtype)
                    max_spec = TensorSpec.from_tensor(output_shape, max_strides, dtype)

                    # Check if output tensors support in-place operations
                    min_supports_inplace = not is_broadcast(min_strides)
                    max_supports_inplace = not is_broadcast(max_strides)

                    if min_supports_inplace and max_supports_inplace:
                        inplace_kwargs = kwargs.copy()

                        test_cases.append(
                            TestCase(
                                inputs=[input_spec],
                                kwargs=inplace_kwargs,
                                output_specs=[
                                    min_spec,
                                    max_spec,
                                ],  # Multiple output specs for in-place
                                comparison_target="out",  # Compare the output tuple from kwargs
                                tolerance=tolerance,
                                description=f"{base_description} - INPLACE(out)",
                                output_count=2,  # Specify 2 outputs
                            )
                        )

    return test_cases


class OpTest(BaseOperatorTest):
    """aminmax operator test with multiple outputs support"""

    def __init__(self):
        super().__init__("aminmax")

    def get_test_cases(self):
        return parse_test_cases()

    def torch_operator(self, x, dim=None, keepdim=False, out=None, **kwargs):
        return torch.aminmax(x, dim=dim, keepdim=keepdim, out=out)


def main():
    """Main entry point"""
    runner = GenericTestRunner(OpTest)
    runner.run_and_exit()


if __name__ == "__main__":
    main()
