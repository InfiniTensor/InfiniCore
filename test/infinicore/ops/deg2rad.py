import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import infinicore
from framework.base import BaseOperatorTest, TensorSpec, TestCase
from framework.runner import GenericTestRunner
from framework.utils import is_broadcast

# =======================================================================
# Test cases format: (shape, input_strides_or_None)
# Test cases format: (in_shape, in_strides_or_None)
# =======================================================================

_TEST_CASES_DATA = [
    ((13, 4), None),
    ((13, 4), (10, 1)),
    ((13, 4), (0, 1)),
    ((8, 16), None),
    ((8, 16), (40, 1)),
    ((16, 5632), None),
    ((2, 3, 4), None),
]

_TOLERANCE_MAP = {
    infinicore.float16: {"atol": 1e-3, "rtol": 1e-2},
    infinicore.float32: {"atol": 1e-5, "rtol": 1e-4},
    infinicore.bfloat16: {"atol": 1e-2, "rtol": 5e-2},
}

_TENSOR_DTYPES = [infinicore.float16, infinicore.bfloat16, infinicore.float32]


def parse_test_cases():
    """
    Parse deg2rad test cases.
    """
    test_cases = []

    for data in _TEST_CASES_DATA:
        shape = data[0]
        in_strides = data[1] if len(data) > 1 else None

        input_supports_inplace = not is_broadcast(in_strides)

        for dtype in _TENSOR_DTYPES:
            tol = _TOLERANCE_MAP.get(dtype, {"atol": 1e-5, "rtol": 1e-4})
            input_spec = TensorSpec.from_tensor(shape, in_strides, dtype)
            out_spec = TensorSpec.from_tensor(shape, None, dtype)

            # Out-of-place
            test_cases.append(
                TestCase(
                    inputs=[input_spec],
                    kwargs={},
                    output_spec=None,
                    comparison_target=None,
                    tolerance=tol,
                    description="deg2rad - OUT_OF_PLACE",
                )
            )

            # Explicit out if output supports inplace (same shape and not broadcast)
            # We assume out supports inplace for same-shaped dense tensors.
            test_cases.append(
                TestCase(
                    inputs=[input_spec],
                    kwargs=None,
                    output_spec=out_spec,
                    comparison_target="out",
                    tolerance=tol,
                    description="deg2rad - INPLACE(out)",
                )
            )

            # In-place overwrite input
            if input_supports_inplace:
                test_cases.append(
                    TestCase(
                        inputs=[input_spec],
                        kwargs={"out": 0},
                        output_spec=None,
                        comparison_target=0,
                        tolerance=tol,
                        description="deg2rad - INPLACE(input)",
                    )
                )

    return test_cases


class OpTest(BaseOperatorTest):
    """Deg2rad operator test with simplified implementation"""

    def __init__(self):
        super().__init__("Deg2rad")

    def get_test_cases(self):
        return parse_test_cases()

    def torch_operator(self, *args, **kwargs):
        return torch.deg2rad(*args, **kwargs)

    # def infinicore_operator(self, *args, **kwargs):
    #     """InfiniCore implementation (operator not yet available)."""
    #     return infinicore.deg2rad(*args, **kwargs)


def main():
    """Main entry point"""
    runner = GenericTestRunner(OpTest)
    runner.run_and_exit()


if __name__ == "__main__":
    main()
