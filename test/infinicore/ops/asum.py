import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
from framework import (
    BaseOperatorTest,
    GenericTestRunner,
    TensorSpec,
    TestCase,
)

import infinicore

# =======================================================================
# Test cases format: (shape, x_strides_or_None)
# =======================================================================

_TEST_CASES_DATA = [
    ((13,), None),
    ((13,), (10,)),
    ((16,), None),
    ((16,), (4,)),
    ((255,), None),
    ((5632,), None),
]

_TOLERANCE_MAP = {
    infinicore.float16: {"atol": 1e-3, "rtol": 1e-2},
    infinicore.float32: {"atol": 1e-5, "rtol": 1e-4},
    infinicore.float64: {"atol": 1e-9, "rtol": 1e-6},
    infinicore.bfloat16: {"atol": 1e-2, "rtol": 5e-2},
}

_TENSOR_DTYPES = [
    # infinicore.float16,
    infinicore.float32,
    # infinicore.float64,
    # infinicore.bfloat16,
]


def torch_asum(x, *, out=None):
    def _asum(x, out):
        out.copy_(torch.sum(x.abs()))

    if out is None:
        out = torch.empty(1, dtype=x.dtype, device=x.device)

    _asum(x, out)
    return out


def parse_test_cases():
    test_cases = []
    for data in _TEST_CASES_DATA:
        shape = data[0]
        x_strides = data[1] if len(data) > 1 else None

        for dtype in _TENSOR_DTYPES:
            tol = _TOLERANCE_MAP.get(dtype, {"atol": 1e-5, "rtol": 1e-4})
            x_spec = TensorSpec.from_tensor(shape, x_strides, dtype)
            out_spec = TensorSpec.from_tensor((), None, dtype)

            test_cases.append(
                TestCase(
                    inputs=[x_spec],
                    kwargs={},
                    output_spec=None,
                    comparison_target=None,
                    tolerance=tol,
                    description="asum - OUT_OF_PLACE",
                )
            )

            test_cases.append(
                TestCase(
                    inputs=[x_spec],
                    kwargs={},
                    output_spec=out_spec,
                    comparison_target="out",
                    tolerance=tol,
                    description="asum - INPLACE(out)",
                )
            )

    return test_cases


class OpTest(BaseOperatorTest):
    """BLAS Level-1 asum operator test"""

    def __init__(self):
        super().__init__("Asum")

    def get_test_cases(self):
        return parse_test_cases()

    def torch_operator(self, *args, **kwargs):
        return torch_asum(*args, **kwargs)

    def infinicore_operator(self, *args, **kwargs):
        return infinicore.asum(*args, **kwargs)


def main():
    """Main entry point"""
    runner = GenericTestRunner(OpTest)
    runner.run_and_exit()


if __name__ == "__main__":
    main()
