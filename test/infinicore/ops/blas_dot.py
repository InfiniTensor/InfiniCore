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
from framework.tensor import TensorInitializer

import infinicore

_TEST_CASES_DATA = [
    ((3,), None, None),
    ((8,), (2,), (3,)),
    ((32,), None, (2,)),
    ((257,), (3,), None),
    ((65535,), None, None),
]

_TOLERANCE_MAP = {
    infinicore.float16: {"atol": 1e-3, "rtol": 1e-3},
    infinicore.float32: {"atol": 1e-5, "rtol": 1e-5},
    infinicore.float64: {"atol": 1e-9, "rtol": 1e-9},
    infinicore.bfloat16: {"atol": 1e-2, "rtol": 1e-2},
}

_TENSOR_DTYPES = [
    infinicore.float16,
    infinicore.float32,
    # infinicore.float64,
    infinicore.bfloat16,
]


def torch_blas_dot(x, y, *, out=None):
    if x.dtype in (torch.float16, torch.bfloat16):
        result = torch.dot(x.float(), y.float()).to(x.dtype)
    else:
        result = torch.dot(x, y)

    if out is None:
        return result

    out.copy_(result)
    return out


def parse_test_cases():
    test_cases = []
    for shape, x_strides, y_strides in _TEST_CASES_DATA:
        for dtype in _TENSOR_DTYPES:
            tol = _TOLERANCE_MAP.get(dtype, {"atol": 1e-5, "rtol": 1e-4})
            x_spec = TensorSpec.from_tensor(shape, x_strides, dtype)
            y_spec = TensorSpec.from_tensor(shape, y_strides, dtype)
            out_spec = TensorSpec.from_tensor(
                (), None, dtype, init_mode=TensorInitializer.ZEROS
            )

            test_cases.append(
                TestCase(
                    inputs=[x_spec, y_spec],
                    kwargs={},
                    output_spec=None,
                    comparison_target=None,
                    tolerance=tol,
                    description="blas_dot - OUT_OF_PLACE",
                )
            )

            test_cases.append(
                TestCase(
                    inputs=[x_spec, y_spec],
                    kwargs={},
                    output_spec=out_spec,
                    comparison_target="out",
                    tolerance=tol,
                    description="blas_dot - INPLACE(out)",
                )
            )

    return test_cases


class OpTest(BaseOperatorTest):
    """BLAS Level-1 dot operator test"""

    def __init__(self):
        super().__init__("BlasDot")

    def get_test_cases(self):
        return parse_test_cases()

    def torch_operator(self, *args, **kwargs):
        return torch_blas_dot(*args, **kwargs)

    def infinicore_operator(self, *args, **kwargs):
        return infinicore.blas_dot(*args, **kwargs)


def main():
    runner = GenericTestRunner(OpTest)
    runner.run_and_exit()


if __name__ == "__main__":
    main()
