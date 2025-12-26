import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch

from framework import (
    BaseOperatorTest,
    GenericTestRunner,
    TensorSpec,
    TestCase
)

import infinicore

# ==============================================================================
# Operator-specific configuration
# ==============================================================================

# Test cases format: (x_shape, weight_shape, bias_shape, bias)
# weight – the  weights of the module of shape (out_features,in_features)
_TEST_CASES_DATA = [
    # Basic cases
    ((1, 10), (2, 10), (2,), True),
    ((4, 10), (2, 10), (2,), False),
    ((1, 1024), (3072, 1024), (3072,), True),
    ((5, 1024), (3072, 1024), (3072,), False),
    ((2, 5, 1024), (3072, 1024), (3072,), True),
    ((10, 5, 1024), (3072, 1024), (3072,), False),
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

    for x_shape, weight_shape, bias_shape, has_bias in _TEST_CASES_DATA:
        strides = None

        # Generate test cases for all data types
        for dtype in _TENSOR_DTYPES:
            tolerance = _TOLERANCE_MAP.get(dtype, {"atol": 0, "rtol": 1e-3})

            # Create typed tensor specs
            x_spec = TensorSpec.from_tensor(x_shape, strides, dtype, name="x")
            weight_spec = TensorSpec.from_tensor(
                weight_shape, strides, dtype, name="weight"
            )
            bias_spec = TensorSpec.from_tensor(bias_shape, strides, dtype, name="bias")

            # Test Case 1: Out-of-place (return value)
            test_cases.append(
                TestCase(
                    inputs=[x_spec, weight_spec, bias_spec],
                    kwargs={"has_bias": has_bias},
                    output_spec=None,
                    comparison_target=None,
                    tolerance=tolerance,
                    description=f"nn.Linear - OUT_OF_PLACE",
                )
            )

    return test_cases


class InfiniNet(infinicore.nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = False,
        device=None,
        dtype=None,
    ):
        super().__init__()
        factory_kwargs = {"device": device, "dtype": dtype}
        self.l = infinicore.nn.Linear(
            in_features, out_features, bias=bias, **factory_kwargs
        )

    def forward(self, x):
        return self.l(x)


class TorchNet(torch.nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = False,
        device=None,
        dtype=None,
    ):
        super().__init__()

        factory_kwargs = {"device": device, "dtype": dtype}
        self.l = torch.nn.Linear(in_features, out_features, bias=bias, **factory_kwargs)

    def forward(self, x):
        return self.l(x)


class OpTest(BaseOperatorTest):
    """nn.Linear test with simplified implementation"""

    def __init__(self):
        super().__init__("nn.Linear")

    def get_test_cases(self):
        return parse_test_cases()

    def torch_operator(self, x, weight, bias, has_bias):
        """PyTorch nn.Linear implementation"""
        out_features, in_features = weight.shape
        params_dict = {"l.weight": weight}
        if has_bias:
            params_dict["l.bias"] = bias

        model = TorchNet(
            in_features,
            out_features,
            bias=has_bias,
            device=weight.device,
            dtype=weight.dtype,
        )
        model.load_state_dict(params_dict)

        with torch.no_grad():
            y = model(x)
        return y

    def infinicore_operator(self, x, weight, bias, has_bias):
        """InfiniCore nn.Linear implementation"""

        out_features, in_features = weight.shape
        params_dict = {"l.weight": weight}
        if has_bias:
            params_dict["l.bias"] = bias

        model = InfiniNet(
            in_features,
            out_features,
            bias=has_bias,
            device=weight.device,
            dtype=weight.dtype,
        )
        model.load_state_dict(params_dict)

        y = model(x)
        return y


def main():
    """Main entry point"""
    runner = GenericTestRunner(OpTest)
    runner.run_and_exit()


if __name__ == "__main__":
    main()
