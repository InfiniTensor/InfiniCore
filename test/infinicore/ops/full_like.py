import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import infinicore
from framework.base import BaseOperatorTest, TensorSpec, TestCase
from framework.runner import GenericTestRunner
from framework.datatypes import to_torch_dtype

# Test cases format: (base_shape, base_strides_or_None, fill_value, dtype_or_None)
_TEST_CASES_DATA = [
    ((3, 4), None, 0.0, None),
    ((6, 2), (12, 1), 1.5, infinicore.float16),
    ((5, 5), None, -2.0, infinicore.float32),
    ((1, 7), None, 3.14, infinicore.bfloat16),
    ((8, 3), (24, 1), 42.0, None),
    ((2, 2, 2), None, 0.5, infinicore.float16),
]

_TOLERANCE_MAP = {
    infinicore.float32: {"atol": 0, "rtol": 0},
    infinicore.float16: {"atol": 1e-3, "rtol": 1e-3},
    infinicore.bfloat16: {"atol": 1e-2, "rtol": 1e-2},
}

_TENSOR_DTYPES = [infinicore.float32, infinicore.float16, infinicore.bfloat16]


def parse_test_cases():
    test_cases = []
    for base_shape, base_strides, val, dtype in _TEST_CASES_DATA:
        for input_dtype in _TENSOR_DTYPES:
            base_spec = TensorSpec.from_tensor(base_shape, base_strides, input_dtype)

            kwargs = {"fill_value": val}
            if dtype is not None:
                kwargs["dtype"] = dtype

            # torch.full_like does not accept an `out=` kwarg in most PyTorch
            # versions; call out-of-place and compare the return value instead.
            test_cases.append(
                TestCase(
                    inputs=[base_spec],
                    kwargs=kwargs,
                    output_spec=None,
                    comparison_target=None,
                    tolerance=_TOLERANCE_MAP.get(dtype, {"atol": 1e-5, "rtol": 1e-4}),
                    description=f"full_like - OUT_OF_PLACE",
                )
            )

    return test_cases


class OpTest(BaseOperatorTest):
    """FullLike operator test with simplified implementation"""

    def __init__(self):
        super().__init__("FullLike")

    def get_test_cases(self):
        return parse_test_cases()

    def torch_operator(self, *args, **kwargs):
        if "fill_value" not in kwargs:
            raise TypeError("full_like test did not provide 'fill_value' parameter")
        fill_value = kwargs.pop("fill_value")

        if "dtype" not in kwargs:
            dtype_torch = None
        else:
            dtype_torch = to_torch_dtype(kwargs.pop("dtype"))
        return torch.full_like(*args, fill_value=fill_value, dtype=dtype_torch)

    # def infinicore_operator(self, *args, **kwargs):
    #     """InfiniCore full_like implementation (operator not yet available)."""
    #     # Mirror the torch_operator signature/behavior but call infinicore implementation.
    #     if "fill_value" not in kwargs:
    #         raise TypeError("full_like test did not provide 'fill_value' parameter")
    #     fill_value = kwargs.pop("fill_value")
    #
    #     if "dtype" not in kwargs:
    #         dtype_infinicore = None
    #     else:
    #         # tests pass infinicore dtypes directly, so forward as-is
    #         dtype_infinicore = kwargs.pop("dtype")
    #     return infinicore.full_like(*args, fill_value=fill_value, dtype=dtype_infinicore)


def main():
    """Main entry point"""
    runner = GenericTestRunner(OpTest)
    runner.run_and_exit()


if __name__ == "__main__":
    main()
