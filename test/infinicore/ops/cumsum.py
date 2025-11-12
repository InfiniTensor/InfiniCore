import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import infinicore
from framework.base import BaseOperatorTest, TensorSpec, TestCase
from framework.runner import GenericTestRunner
from framework.utils import is_broadcast

# Test cases format: (shape, dim, input_strides_or_None, out_strides_or_None)
# cumsum supports out= in PyTorch? PyTorch provides infinicore.cumsum(input, dim, *, out=None)
# so we include explicit out cases.

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
    test_cases = []
    for data in _TEST_CASES_DATA:
        shape, dim = data[0], data[1]
        in_strides = data[2] if len(data) > 2 else None
        out_strides = data[3] if len(data) > 3 else None

        out_supports_inplace = not is_broadcast(out_strides)
        input_supports_inplace = not is_broadcast(in_strides)

        for dtype in _TENSOR_DTYPES:
            tol = _TOLERANCE_MAP.get(dtype, {"atol": 0, "rtol": 1e-3})
            input_spec = TensorSpec.from_tensor(shape, in_strides, dtype)
            out_spec = TensorSpec.from_tensor(shape, out_strides, dtype)

            # Out-of-place: pass dim as positional argument to match infinicore.cumsum(input, dim, *, dtype=None, out=None)
            test_cases.append(
                TestCase(
                    inputs=[input_spec, dim],
                    kwargs={},
                    output_spec=None,
                    comparison_target=None,
                    tolerance=tol,
                    description=f"cumsum - OUT_OF_PLACE",
                )
            )

            # Explicit out
            if out_supports_inplace:
                test_cases.append(
                    TestCase(
                        inputs=[input_spec, dim],
                        kwargs=None,
                        output_spec=out_spec,
                        comparison_target="out",
                        tolerance=tol,
                        description=f"cumsum - INPLACE(out)",
                    )
                )

            # In-place on input (overwrite) - if input supports inplace and op accepts out param
            if input_supports_inplace:
                test_cases.append(
                    TestCase(
                        inputs=[input_spec, dim],
                        kwargs={"out": 0},
                        output_spec=None,
                        comparison_target=0,
                        tolerance=tol,
                        description=f"cumsum - INPLACE(input)",
                    )
                )

    return test_cases


class OpTest(BaseOperatorTest):
    """Cumsum operator test with simplified implementation"""

    def __init__(self):
        super().__init__("Cumsum")

    def get_test_cases(self):
        return parse_test_cases()

    def torch_operator(self, *args, **kwargs):
        return torch.cumsum(*args, **kwargs)

    # def infinicore_operator(self, *args, **kwargs):
    #     """InfiniCore implementation (operator not yet available)."""
    #     return infinicore.cumsum(*args, **kwargs)


def main():
    """Main entry point"""
    runner = GenericTestRunner(OpTest)
    runner.run_and_exit()


if __name__ == "__main__":
    main()
