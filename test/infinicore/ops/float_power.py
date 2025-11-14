import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import infinicore
from framework.base import BaseOperatorTest, TensorSpec, TestCase
from framework.runner import GenericTestRunner
from framework.utils import is_broadcast

# Test cases format: (in_shape, in_strides_or_None, exponent_scalar_or_None, exponent_tensor_shape_or_None)
# infinicore.float_power(input, exponent)

_TEST_CASES_DATA = [
    ((2, 3, 4), None, 2.0, None),
    ((1, 4, 8), (32, 8, 1), None, (1, 4, 8)),
    ((3, 2, 5, 7), None, 3.0, None),
    ((2, 1, 16), None, None, (2, 1, 16)),
    ((1, 8, 9, 11), (792, 99, 11, 1), 1.5, None),
    ((2, 6, 10), None, None, (2, 6, 10)),
]

_TOLERANCE_MAP = {
    infinicore.float16: {"atol": 1e-3, "rtol": 1e-2},
    infinicore.float32: {"atol": 1e-5, "rtol": 1e-4},
}
_TENSOR_DTYPES = [infinicore.float16, infinicore.float32]


def parse_test_cases():
    cases = []
    dtype_map = {
        infinicore.float16: infinicore.float64,
        infinicore.float32: infinicore.float64,
        infinicore.complex64: infinicore.complex128,
    }

    for shape, strides, exp_scalar, exp_tensor_shape in _TEST_CASES_DATA:
        for dtype in _TENSOR_DTYPES:
            tol = _TOLERANCE_MAP[dtype]
            input_spec = TensorSpec.from_tensor(shape, strides, dtype)
            out_dtype = dtype_map.get(dtype, dtype)

            # exponent as scalar
            if exp_scalar is not None:
                kwargs = {}
                cases.append(
                    TestCase(
                        inputs=[input_spec, exp_scalar],
                        kwargs=kwargs,
                        output_spec=None,
                        comparison_target=None,
                        tolerance=tol,
                        description="float_power_scalar_exp - OUT_OF_PLACE",
                    )
                )
                out_spec = TensorSpec.from_tensor(shape, None, out_dtype)
                cases.append(
                    TestCase(
                        inputs=[input_spec, exp_scalar],
                        kwargs={},
                        output_spec=out_spec,
                        comparison_target="out",
                        tolerance=tol,
                        description="float_power_scalar_exp - INPLACE(out)",
                    )
                )

            # exponent as tensor
            if exp_tensor_shape is not None:
                exp_spec = TensorSpec.from_tensor(exp_tensor_shape, None, dtype)
                cases.append(
                    TestCase(
                        inputs=[input_spec, exp_spec],
                        kwargs={},
                        output_spec=None,
                        comparison_target=None,
                        tolerance=tol,
                        description="float_power_tensor_exp - OUT_OF_PLACE",
                    )
                )
                out_spec = TensorSpec.from_tensor(shape, None, out_dtype)
                cases.append(
                    TestCase(
                        inputs=[input_spec, exp_spec],
                        kwargs={},
                        output_spec=out_spec,
                        comparison_target="out",
                        tolerance=tol,
                        description="float_power_tensor_exp_explicit_out",
                    )
                )

    return cases


class OpTest(BaseOperatorTest):
    """FloatPower operator test with simplified implementation"""

    def __init__(self):
        super().__init__("FloatPower")

    def get_test_cases(self):
        return parse_test_cases()

    def torch_operator(self, *args, **kwargs):
        return torch.float_power(*args, **kwargs)


# def infinicore_operator(self, *args, **kwargs):
# """InfiniCore implementation (operator not yet available)."""
#     return infinicore.float_power(*args, **kwargs)


def main():
    """Main entry point"""
    runner = GenericTestRunner(OpTest)
    runner.run_and_exit()


if __name__ == "__main__":
    main()
