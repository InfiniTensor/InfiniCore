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

_TEST_CASES_DATA = [
    ((13,), None),
    ((13,), (10,)),
    ((5632,), None),
    ((5632,), (5,)),
    ((16,), (4,)),
    ((5632,), (32,)),
]

_TOLERANCE_MAP = {
    infinicore.float16: {"atol": 1e-3, "rtol": 1e-3},
    infinicore.float32: {"atol": 1e-5, "rtol": 1e-5},
    infinicore.float64: {"atol": 1e-7, "rtol": 1e-7},
    infinicore.bfloat16: {"atol": 1e-2, "rtol": 1e-2},
}

_TENSOR_DTYPES = [
    infinicore.float16,
    infinicore.float32,
    # infinicore.float64,
    infinicore.bfloat16,
]


def torch_nrm2(x, *, out=None):
    result = torch.norm(x, p=2)
    if out is None:
        return result

    out.copy_(result)
    return out


def parse_test_cases():
    test_cases = []
    for shape, x_strides in _TEST_CASES_DATA:
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
                    description="nrm2 - OUT_OF_PLACE",
                )
            )

            test_cases.append(
                TestCase(
                    inputs=[x_spec],
                    kwargs={},
                    output_spec=out_spec,
                    comparison_target="out",
                    tolerance=tol,
                    description="nrm2 - INPLACE(out)",
                )
            )

    return test_cases


class OpTest(BaseOperatorTest):
    """BLAS Level-1 nrm2 operator test"""

    def __init__(self):
        super().__init__("Nrm2")

    def get_test_cases(self):
        return parse_test_cases()

    def torch_operator(self, *args, **kwargs):
        return torch_nrm2(*args, **kwargs)

    def infinicore_operator(self, *args, **kwargs):
        return infinicore.nrm2(*args, **kwargs)


def main():
    runner = GenericTestRunner(OpTest)
    runner.run_and_exit()


if __name__ == "__main__":
    main()
