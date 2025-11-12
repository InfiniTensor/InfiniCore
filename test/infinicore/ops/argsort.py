import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import infinicore
from framework.base import BaseOperatorTest, TensorSpec, TestCase
from framework.runner import GenericTestRunner
from framework.utils import is_broadcast

# Test cases format: (shape, input_strides, dim, descending)
_TEST_CASES_DATA = [
    ((6, 8), None, 1, False),
    ((8, 4), (16, 1), 0, True),
    ((5, 5), None, -1, False),
    ((3, 7), (14, 1), 1, True),
    ((10, 3), None, 1, False),
    ((2, 16), (32, 1), 0, True),
]

_TOLERANCE_MAP = {
    infinicore.float16: {"atol": 0, "rtol": 1e-3},
    infinicore.float32: {"atol": 0, "rtol": 1e-5},
    infinicore.bfloat16: {"atol": 0, "rtol": 1e-2},
}

# For argsort the output is an index tensor (int64). We keep input dtypes as floats.
_TENSOR_DTYPES = [infinicore.float16, infinicore.bfloat16, infinicore.float32]


def parse_test_cases():
    test_cases = []
    for data in _TEST_CASES_DATA:
        shape, in_strides, dim, desc = data
        out_supports_inplace = not is_broadcast(in_strides)

        for dtype in _TENSOR_DTYPES:
            tol = _TOLERANCE_MAP.get(dtype, {"atol": 0, "rtol": 1e-5})

            input_spec = TensorSpec.from_tensor(shape, in_strides, dtype)
            out_spec = TensorSpec.from_tensor(shape, None, infinicore.int64)

            # Out-of-place
            test_cases.append(
                TestCase(
                    inputs=[input_spec],
                    kwargs={"stable": False, "dim": dim, "descending": desc},
                    output_spec=None,
                    comparison_target=None,
                    tolerance=tol,
                    description="argsort - OUT_OF_PLACE",
                )
            )

            # Explicit out
            if out_supports_inplace:
                test_cases.append(
                    TestCase(
                        inputs=[input_spec],
                        kwargs={"stable": False, "dim": dim, "descending": desc},
                        output_spec=out_spec,
                        comparison_target="out",
                        tolerance=tol,
                        description="argsort - INPLACE(out)",
                    )
                )

    return test_cases


class OpTest(BaseOperatorTest):
    """Argsort operator test with simplified implementation"""

    def __init__(self):
        super().__init__("Argsort")

    def get_test_cases(self):
        return parse_test_cases()

    def torch_operator(self, *args, **kwargs):
        return torch.argsort(*args, **kwargs)

    # def infinicore_operator(self, *args, **kwargs):
    #     """InfiniCore implementation (operator not yet available)."""
    #     return infinicore.argsort(*args, **kwargs)


def main():
    """Main entry point"""
    runner = GenericTestRunner(OpTest)
    runner.run_and_exit()


if __name__ == "__main__":
    main()
