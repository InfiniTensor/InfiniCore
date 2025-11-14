import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import infinicore
from framework.base import BaseOperatorTest, TensorSpec, TestCase
from framework.runner import GenericTestRunner
from framework.utils import is_broadcast

# ==============================================================================
# Operator-specific configuration for sort
# ==============================================================================

# Test cases format: (shape, dim, descending, input_strides, values_strides, indices_strides)
_TEST_CASES_DATA = [
    # Basic cases
    ((13, 4), None, False, None, None, None),
    ((13, 4), 0, False, None, None, None),
    ((13, 4), 1, False, None, None, None),
    ((13, 4), -1, False, None, None, None),
    # Descending
    ((13, 4), 1, True, None, None, None),
    # Stable flag (PyTorch 1.8+ supports stable sort; include it to match 2.9 signature)
    ((4, 5, 6), 1, False, None, None, None),
    ((4, 5, 6), -1, True, None, None, None),
    # 3D in-place cases
    ((4, 5, 6), 1, False, None, (4, 1, 6), (4, 1, 6)),
    ((4, 5, 6), -1, False, (30, 6, 1), (64, 1, 5), (64, 1, 5)),
    # Strided inputs and outputs
    ((13, 4), None, False, (4, 1), (12, 1), (24, 1)),
    ((13, 4), 0, False, (1, 4), (64, 1), (1, 4)),
    ((13, 4), 1, False, (1, 4), (64, 1), (1, 4)),
]

# Tolerance configuration
_TOLERANCE_MAP = {
    infinicore.float16: {"atol": 1e-3, "rtol": 1e-2},
    infinicore.float32: {"atol": 1e-5, "rtol": 1e-4},
    infinicore.bfloat16: {"atol": 1e-2, "rtol": 5e-2},
}

# Data types to test
_TENSOR_DTYPES = [infinicore.float16, infinicore.bfloat16, infinicore.float32]


def calculate_output_shape(input_shape, dim):
    """
    Calculate the output shape for sort (values and indices share the same shape)
    """
    if dim is None:
        # Default behavior: sort on last dimension
        dim = len(input_shape) - 1 if len(input_shape) > 0 else 0
    # normalize negative dim
    if dim < 0:
        dim = dim + len(input_shape)
    output_shape = list(input_shape)
    return tuple(output_shape)


def parse_test_cases():
    """
    Parse sort test cases including both out-of-place and in-place (out=...) variants.
    torch.sort(input, dim=-1, descending=False, stable=False, out=None)
    returns (values, indices)
    """
    test_cases = []

    for data in _TEST_CASES_DATA:
        shape = data[0]
        dim = data[1] if len(data) > 1 else None
        descending = data[2] if len(data) > 2 else False
        input_strides = data[3] if len(data) > 3 else None
        values_strides = data[4] if len(data) > 4 else None
        indices_strides = data[5] if len(data) > 5 else None

        for dtype in _TENSOR_DTYPES:
            tolerance = _TOLERANCE_MAP.get(dtype, {"atol": 1e-5, "rtol": 1e-4})

            # Create tensor specs
            input_spec = TensorSpec.from_tensor(shape, input_strides, dtype)

            # Build description
            description_parts = ["sort"]
            if dim is not None:
                description_parts.append(f"dim={dim}")
            if descending:
                description_parts.append("descending=True")
            if input_strides is not None:
                description_parts.append(f"input_strides={input_strides}")

            base_description = " - ".join(description_parts)

            # Common kwargs
            kwargs = {}
            if dim is not None:
                kwargs["dim"] = dim
            kwargs["descending"] = descending
            # stable is available in newer PyTorch; keep default False

            # ==================================================================
            # Test Case 1: Out-of-place (return values)
            # ==================================================================
            test_cases.append(
                TestCase(
                    inputs=[input_spec],
                    kwargs=kwargs,
                    output_spec=None,  # return values will be compared
                    comparison_target=None,
                    tolerance=tolerance,
                    description=f"{base_description} - OUT_OF_PLACE",
                    output_count=2,  # (values, indices)
                )
            )

            # ==================================================================
            # Test Case 2: In-place with explicit output tensors (out=(values, indices))
            # ==================================================================
            output_shape = calculate_output_shape(shape, dim)

            # Create output specs if strides provided; otherwise None
            values_spec = TensorSpec.from_tensor(output_shape, values_strides, dtype)
            # indices are integer type (long) in PyTorch
            indices_spec = TensorSpec.from_tensor(
                output_shape, indices_strides, infinicore.int64
            )

            values_supports_inplace = not is_broadcast(
                getattr(values_spec, "strides", None)
            )
            indices_supports_inplace = not is_broadcast(
                getattr(indices_spec, "strides", None)
            )

            if values_supports_inplace and indices_supports_inplace:
                inplace_kwargs = kwargs.copy()

                test_cases.append(
                    TestCase(
                        inputs=[input_spec],
                        kwargs=inplace_kwargs,
                        output_specs=[values_spec, indices_spec],
                        comparison_target="out",
                        tolerance=tolerance,
                        description=f"{base_description} - INPLACE(out)",
                        output_count=2,
                    )
                )

    return test_cases


class OpTest(BaseOperatorTest):
    """Sort operator test with multiple outputs (values, indices)"""

    def __init__(self):
        super().__init__("sort")

    def get_test_cases(self):
        return parse_test_cases()

    def torch_operator(
        self, x, dim=-1, descending=False, stable=False, out=None, **kwargs
    ):
        # forward to torch.sort; stable kwarg included for compatibility
        return torch.sort(x, dim=dim, descending=descending, stable=stable, out=out)

    # def infinicore_operator(self, x, dim=-1, descending=False, stable=False, out=None, **kwargs):
    #     # assume infinicore provides a similar API
    #     return infinicore.sort(x, dim=dim, descending=descending, stable=stable, out=out)


def main():
    runner = GenericTestRunner(OpTest)
    runner.run_and_exit()


if __name__ == "__main__":
    main()
