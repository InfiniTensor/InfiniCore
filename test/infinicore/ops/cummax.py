import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import infinicore
from framework.base import BaseOperatorTest, TensorSpec, TestCase
from framework.runner import GenericTestRunner
from framework.utils import is_broadcast

# Test cases format: (shape, dim, input_strides_or_None, out_strides_or_None)
# cummax returns (values, indices). We will validate the values tensor using the
# comparison_target; indices are returned by PyTorch but the framework compares
# on the primary output (values). Indices tests are not explicitly compared here.

_TEST_CASES_DATA = [
    ((13, 4), 1, None, None),
    ((13, 4), 0, (10, 1), None),
    ((8, 16), 1, None, None),
    ((2, 3, 4), 2, None, None),
    ((16, 64), 1, (128, 1), (128, 1)),
    ((4, 5, 6), 0, None, None),
]

_TOLERANCE_MAP = {
    infinicore.float16: {"atol": 0, "rtol": 1e-2},
    infinicore.float32: {"atol": 0, "rtol": 1e-3},
    infinicore.bfloat16: {"atol": 0, "rtol": 5e-2},
}

_TENSOR_DTYPES = [infinicore.float16, infinicore.bfloat16, infinicore.float32]


def parse_test_cases():
    """
    Generate test cases for cummax.
    """
    test_cases = []
    for data in _TEST_CASES_DATA:
        shape, dim = data[0], data[1]
        in_strides = data[2] if len(data) > 2 else None
        out_strides = data[3] if len(data) > 3 else None

        input_supports_inplace = not is_broadcast(in_strides)
        out_supports_inplace = not is_broadcast(out_strides)

        for dtype in _TENSOR_DTYPES:
            tol = _TOLERANCE_MAP.get(dtype, {"atol": 0, "rtol": 1e-3})
            input_spec = TensorSpec.from_tensor(shape, in_strides, dtype)
            out_spec = TensorSpec.from_tensor(shape, out_strides, dtype)

            # Out-of-place (returns values, indices) - compare values
            kwargs = {"dim": dim}
            test_cases.append(
                TestCase(
                    inputs=[input_spec],
                    kwargs=kwargs,
                    output_spec=None,
                    comparison_target=None,
                    tolerance=tol,
                    description=f"cummax - OUT_OF_PLACE",
                )
            )

            # Explicit out for values (if supported) - PyTorch doesn't accept out for cummax, so skip
            # Note: PyTorch does not support explicit 'out' for cummax, so we don't add out= cases.

            # In-place on input (overwrite) - not supported by PyTorch for cummax
            # Note: cummax does not support inplace modification via inplace=True, so no INPLACE cases added.

    return test_cases


class OpTest(BaseOperatorTest):
    """Cummax operator test with simplified implementation"""

    def __init__(self):
        super().__init__("Cummax")

    def get_test_cases(self):
        return parse_test_cases()

    def torch_operator(self, *args, **kwargs):
        return torch.cummax(*args, **kwargs)

    # def infinicore_operator(self, *args, **kwargs):
    #     """InfiniCore implementation (operator not yet available)."""
    #     return infinicore.cummax(*args, **kwargs)


def main():
    """Main entry point"""
    runner = GenericTestRunner(OpTest)
    runner.run_and_exit()


if __name__ == "__main__":
    main()
