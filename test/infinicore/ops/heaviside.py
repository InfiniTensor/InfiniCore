import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import infinicore
from framework.base import BaseOperatorTest, TensorSpec, TestCase
from framework.runner import GenericTestRunner
from framework.utils import is_broadcast

# =======================================================================
# Test cases format: (shape, a_strides_or_None, b_strides_or_None, out_strides_or_None)
# heaviside is binary: heaviside(input, values)
# =======================================================================

_TEST_CASES_DATA = [
    ((13, 4), None, None, None),
    ((13, 4), (10, 1), None, None),
    ((13, 4), None, (10, 1), None),
    ((8, 16), (40, 1), (40, 1), None),
    ((2, 3, 4), None, None, None),
    ((16, 5632), None, None, None),
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
        shape = data[0]
        a_strides = data[1] if len(data) > 1 else None
        b_strides = data[2] if len(data) > 2 else None
        out_strides = data[3] if len(data) > 3 else None

        a_supports_inplace = not is_broadcast(a_strides)
        b_supports_inplace = not is_broadcast(b_strides)
        out_supports_inplace = not is_broadcast(out_strides)

        for dtype in _TENSOR_DTYPES:
            tol = _TOLERANCE_MAP.get(dtype, {"atol": 0, "rtol": 1e-3})
            a_spec = TensorSpec.from_tensor(shape, a_strides, dtype)
            b_spec = TensorSpec.from_tensor(shape, b_strides, dtype)
            out_spec = TensorSpec.from_tensor(shape, out_strides, dtype)

            # Out-of-place
            test_cases.append(
                TestCase(
                    inputs=[a_spec, b_spec],
                    kwargs={},
                    output_spec=None,
                    comparison_target=None,
                    tolerance=tol,
                    description="heaviside - OUT_OF_PLACE",
                )
            )

            # Explicit out
            if out_supports_inplace:
                test_cases.append(
                    TestCase(
                        inputs=[a_spec, b_spec],
                        kwargs=None,
                        output_spec=out_spec,
                        comparison_target="out",
                        tolerance=tol,
                        description="heaviside - INPLACE(out)",
                    )
                )

            # In-place on first input
            if a_supports_inplace:
                test_cases.append(
                    TestCase(
                        inputs=[a_spec, b_spec],
                        kwargs={"out": 0},
                        output_spec=None,
                        comparison_target=0,
                        tolerance=tol,
                        description="heaviside - INPLACE(a)",
                    )
                )

            # In-place on second input
            if b_supports_inplace:
                test_cases.append(
                    TestCase(
                        inputs=[a_spec, b_spec],
                        kwargs={"out": 1},
                        output_spec=None,
                        comparison_target=1,
                        tolerance=tol,
                        description="heaviside - INPLACE(b)",
                    )
                )

    return test_cases


class OpTest(BaseOperatorTest):
    """Heaviside operator test with simplified implementation"""

    def __init__(self):
        super().__init__("Heaviside")

    def get_test_cases(self):
        return parse_test_cases()

    def torch_operator(self, *args, **kwargs):
        return torch.heaviside(*args, **kwargs)

    # def infinicore_operator(self, *args, **kwargs):
    #     """InfiniCore implementation (operator not yet available)."""
    #     return infinicore.heaviside(*args, **kwargs)


def main():
    """Main entry point"""
    runner = GenericTestRunner(OpTest)
    runner.run_and_exit()


if __name__ == "__main__":
    main()
