import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import infinicore
from framework.base import BaseOperatorTest, TensorSpec, TestCase
from framework.runner import GenericTestRunner
from framework.utils import is_broadcast

# Test cases format: (shape, n, dim, input_strides_or_None)

_TEST_CASES_DATA = [
    ((13, 4), 1, 1, None),
    ((13, 6), 2, 1, (12, 1)),
    ((8, 16), 1, 0, None),
    ((2, 3, 5), 1, 2, None),
    ((16, 64), 3, 1, None),
    ((4, 5, 6), 2, 0, None),
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
        shape, n, dim = data[0], data[1], data[2]
        in_strides = data[3] if len(data) > 3 else None

        input_supports_inplace = not is_broadcast(in_strides)

        for dtype in _TENSOR_DTYPES:
            tol = _TOLERANCE_MAP.get(dtype, {"atol": 0, "rtol": 1e-3})
            input_spec = TensorSpec.from_tensor(shape, in_strides, dtype)
            out_shape = list(shape)
            # diff reduces size along dim by n (if valid) — shapes may differ
            try:
                out_shape[dim] = out_shape[dim] - n
            except Exception:
                pass
            out_spec = TensorSpec.from_tensor(tuple(out_shape), None, dtype)

            kwargs = {"n": n, "dim": dim}
            # Out-of-place
            test_cases.append(
                TestCase(
                    inputs=[input_spec],
                    kwargs=kwargs,
                    output_spec=None,
                    comparison_target=None,
                    tolerance=tol,
                    description=f"diff - OUT_OF_PLACE",
                )
            )

            # PyTorch does not support explicit out for diff — skip explicit out tests
            # Note: PyTorch diff does not accept out parameter; hence no INPLACE(out) cases.

    return test_cases


class OpTest(BaseOperatorTest):
    """Diff operator test with simplified implementation"""

    def __init__(self):
        super().__init__("Diff")

    def get_test_cases(self):
        return parse_test_cases()

    def torch_operator(self, *args, **kwargs):
        return torch.diff(*args, **kwargs)

    # def infinicore_operator(self, *args, **kwargs):
    #     """InfiniCore implementation (operator not yet available)."""
    #     return infinicore.diff(*args, **kwargs)


def main():
    """Main entry point"""
    runner = GenericTestRunner(OpTest)
    runner.run_and_exit()


if __name__ == "__main__":
    main()
