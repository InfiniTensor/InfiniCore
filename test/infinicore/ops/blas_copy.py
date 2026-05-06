import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from framework import (
    BaseOperatorTest,
    GenericTestRunner,
    TensorSpec,
    TestCase,
)

import infinicore

_TEST_CASES_DATA = [
    ((3,), None, None),
    ((8,), (2,), (3,)),
    ((32,), None, (2,)),
    ((257,), (3,), None),
    ((65535,), None, None),
]

_TENSOR_DTYPES = [
    # infinicore.float16,
    infinicore.float32,
    # infinicore.float64,
    # infinicore.bfloat16,
]

_TOLERANCE_MAP = {
    infinicore.float16: {"atol": 1e-3, "rtol": 1e-3},
    infinicore.float32: {"atol": 1e-7, "rtol": 1e-7},
    infinicore.float64: {"atol": 1e-15, "rtol": 1e-15},
    infinicore.bfloat16: {"atol": 5e-3, "rtol": 1e-2},
}


def torch_blas_copy(x, y):
    y.copy_(x)
    return y


def parse_test_cases():
    test_cases = []
    for shape, x_strides, y_strides in _TEST_CASES_DATA:
        for dtype in _TENSOR_DTYPES:
            tol = _TOLERANCE_MAP.get(dtype, {"atol": 1e-5, "rtol": 1e-4})
            x_spec = TensorSpec.from_tensor(shape, x_strides, dtype)
            y_spec = TensorSpec.from_tensor(shape, y_strides, dtype)

            test_cases.append(
                TestCase(
                    inputs=[x_spec, y_spec],
                    kwargs={},
                    output_spec=None,
                    comparison_target=1,
                    tolerance=tol,
                    description="blas_copy - INPLACE",
                )
            )

    return test_cases


class OpTest(BaseOperatorTest):
    """BLAS Level-1 copy operator test"""

    def __init__(self):
        super().__init__("BlasCopy")

    def get_test_cases(self):
        return parse_test_cases()

    def torch_operator(self, *args, **kwargs):
        return torch_blas_copy(*args, **kwargs)

    def infinicore_operator(self, *args, **kwargs):
        return infinicore.blas_copy(*args, **kwargs)


def main():
    runner = GenericTestRunner(OpTest)
    runner.run_and_exit()


if __name__ == "__main__":
    main()
