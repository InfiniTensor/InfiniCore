import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

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

_TENSOR_DTYPES = [
    infinicore.float16,
    infinicore.float32,
    # infinicore.float64,
    infinicore.bfloat16,
]

_TOLERANCE_MAP = {
    infinicore.float16: {"atol": 1e-3, "rtol": 1e-3},
    infinicore.float32: {"atol": 1e-5, "rtol": 1e-5},
    infinicore.float64: {"atol": 1e-9, "rtol": 1e-9},
    infinicore.bfloat16: {"atol": 1e-2, "rtol": 1e-2},
}


def torch_axpy(alpha, x, y):
    y.add_(x, alpha=alpha.item())

    return y


def parse_test_cases():
    test_cases = []
    for shape, x_strides, y_strides in _TEST_CASES_DATA:
        for dtype in _TENSOR_DTYPES:
            tol = _TOLERANCE_MAP.get(dtype, {"atol": 1e-5, "rtol": 1e-4})
            alpha_spec = TensorSpec.from_tensor(
                (), None, dtype, init_mode=TensorInitializer.ONES
            )
            x_spec = TensorSpec.from_tensor(shape, x_strides, dtype)
            y_spec = TensorSpec.from_tensor(shape, y_strides, dtype)

            test_cases.append(
                TestCase(
                    inputs=[alpha_spec, x_spec, y_spec],
                    kwargs={},
                    output_spec=None,
                    comparison_target=2,
                    tolerance=tol,
                    description="axpy - INPLACE",
                )
            )

    return test_cases


class OpTest(BaseOperatorTest):
    """BLAS Level-1 axpy operator test"""

    def __init__(self):
        super().__init__("Axpy")

    def get_test_cases(self):
        return parse_test_cases()

    def torch_operator(self, *args, **kwargs):
        return torch_axpy(*args, **kwargs)

    def infinicore_operator(self, *args, **kwargs):
        return infinicore.axpy(*args, **kwargs)


def main():
    runner = GenericTestRunner(OpTest)
    runner.run_and_exit()


if __name__ == "__main__":
    main()
