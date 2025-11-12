import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import infinicore
from framework.base import BaseOperatorTest, TensorSpec, TestCase
from framework.runner import GenericTestRunner
from framework.utils import is_broadcast

# Test cases format: (in_shape, in_strides_or_None, weight_shape_or_None)
# Note: PReLU requires a weight parameter of shape (C,) or (1,), we create a per-channel weight when possible.

_TEST_CASES_DATA = [
    ((4, 4), None, None),
    ((8, 4, 4), (128, 32, 1), (4,)),
    ((2, 3, 6), None, (3,)),
]

_TOLERANCE_MAP = {
    infinicore.float16: {"atol": 1e-3, "rtol": 1e-2},
    infinicore.float32: {"atol": 1e-5, "rtol": 1e-4},
    infinicore.bfloat16: {"atol": 1e-2, "rtol": 5e-2},
}

_TENSOR_DTYPES = [infinicore.float16, infinicore.bfloat16, infinicore.float32]


def parse_test_cases():
    """prelu(input, weight)"""
    test_cases = []

    for data in _TEST_CASES_DATA:
        shape = data[0]
        in_strides = data[1] if len(data) > 1 else None
        weight_shape = data[2] if len(data) > 2 and data[2] is not None else None

        for dtype in _TENSOR_DTYPES:
            tolerance = _TOLERANCE_MAP.get(dtype, {"atol": 1e-5, "rtol": 1e-4})

            input_spec = TensorSpec.from_tensor(shape, in_strides, dtype)

            # Determine default weight shape: channel dimension if available
            if weight_shape is None:
                if len(shape) >= 2:
                    c = shape[1]
                    weight_spec = TensorSpec.from_tensor((c,), None, dtype)
                else:
                    weight_spec = TensorSpec.from_tensor((1,), None, dtype)
            else:
                weight_spec = TensorSpec.from_tensor(weight_shape, None, dtype)

            test_cases.append(
                TestCase(
                    inputs=[input_spec, weight_spec],
                    kwargs={},
                    output_spec=None,
                    comparison_target=None,
                    tolerance=tolerance,
                    description=f"PReLU - OUT_OF_PLACE",
                )
            )

    return test_cases


class OpTest(BaseOperatorTest):
    """PReLU operator test with simplified implementation"""

    def __init__(self):
        super().__init__("PReLU")

    def get_test_cases(self):
        return parse_test_cases()

    def torch_operator(self, *args, **kwargs):
        return torch.nn.functional.prelu(*args, **kwargs)

    # def infinicore_operator(self, *args, **kwargs):
    #     """InfiniCore implementation (operator not yet available)."""
    #     return infinicore.nn.functional.prelu(*args, **kwargs)


def main():
    """Main entry point"""
    runner = GenericTestRunner(OpTest)
    runner.run_and_exit()


if __name__ == "__main__":
    main()
