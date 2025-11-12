import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import infinicore
from framework.base import BaseOperatorTest, TensorSpec, TestCase
from framework.runner import GenericTestRunner
from framework.utils import is_broadcast

# # Test cases format: (shape, input_strides, src_strides, dim, start, end, step)
# _TEST_CASES_DATA = [
#     ((6, 8), None, None, 1, 0, 4, 1),
#     ((8, 4), (16, 1), (16, 1), 0, 1, 3, 1),
#     ((5, 7), None, (10, 1), 1, -3, None, 1),
#     ((3, 9), None, None, 1, 2, 8, 2),
#     ((10, 3), (30, 1), (30, 1), 0, None, None, 1),
#     ((2, 16), None, None, 1, 0, 2, 1),
# ]
# Format: (input_shape, input_strides, src_shape, src_strides, dim, start, end, step)
_TEST_CASES_DATA = [
    ((6, 8), None, (6, 4), None, 1, 0, 4, 1),
    ((8, 4), (16, 1), (2, 4), (16, 1), 0, 1, 3, 1),
    ((5, 7), None, (5, 3), (10, 1), 1, -3, None, 1),
    ((3, 9), None, (3, 3), None, 1, 2, 8, 2),
    ((10, 3), (30, 1), (10, 3), (30, 1), 0, None, None, 1),
    ((2, 16), None, (2, 2), None, 1, 0, 2, 1),
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
        input_shape, in_strides, src_shape, src_strides, dim, start, end, step = data

        in_supports_inplace = not is_broadcast(in_strides)

        for dtype in _TENSOR_DTYPES:
            tol = _TOLERANCE_MAP.get(dtype, {"atol": 0, "rtol": 1e-3})

            input_spec = TensorSpec.from_tensor(input_shape, in_strides, dtype)
            src_spec = TensorSpec.from_tensor(src_shape, src_strides, dtype)

            kwargs = {"dim": dim}
            if start is not None:
                kwargs["start"] = start
            if end is not None:
                kwargs["end"] = end
            if step is not None:
                kwargs["step"] = step

            # Out-of-place
            test_cases.append(
                TestCase(
                    inputs=[input_spec, src_spec],
                    kwargs=kwargs,
                    output_spec=None,
                    comparison_target=None,
                    tolerance=tol,
                    description=f"slice_scatter - OUT_OF_PLACE",
                )
            )

            if in_supports_inplace:
                test_cases.append(
                    TestCase(
                        inputs=[input_spec, src_spec],
                        kwargs=kwargs,
                        output_spec=None,
                        comparison_target=0,
                        tolerance=tol,
                        description=f"slice_scatter - INPLACE(input)",
                    )
                )

    return test_cases


class OpTest(BaseOperatorTest):
    """SliceScatter operator test with simplified implementation"""

    def __init__(self):
        super().__init__("SliceScatter")

    def get_test_cases(self):
        return parse_test_cases()

    def torch_operator(self, *args, **kwargs):
        return torch.slice_scatter(*args, **kwargs)

    # def infinicore_operator(self, *args, **kwargs):
    #     """InfiniCore implementation (operator not yet available)."""
    #     return infinicore.slice_scatter(*args, **kwargs)


def main():
    """Main entry point"""
    runner = GenericTestRunner(OpTest)
    runner.run_and_exit()


if __name__ == "__main__":
    main()
