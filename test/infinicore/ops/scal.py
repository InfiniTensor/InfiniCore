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
    ((13,), None),
    ((13,), (10,)),
    ((5632,), None),
    ((5632,), (5,)),
    ((16,), (4,)),
    ((5632,), (32,)),
]

_TENSOR_DTYPES = [
    infinicore.float16,
    infinicore.float32,
    # infinicore.float64,
    infinicore.bfloat16,
]

_TOLERANCE_MAP = {
    infinicore.float16: {"atol": 1e-3, "rtol": 1e-3},
    infinicore.float32: {"atol": 1e-7, "rtol": 1e-7},
    infinicore.float64: {"atol": 1e-15, "rtol": 1e-15},
    infinicore.bfloat16: {"atol": 5e-3, "rtol": 1e-2},
}


def torch_scal(x, alpha):
    x.mul_(alpha)
    return x


def parse_test_cases():
    test_cases = []
    for shape, x_strides in _TEST_CASES_DATA:
        for dtype in _TENSOR_DTYPES:
            tol = _TOLERANCE_MAP.get(dtype, {"atol": 1e-5, "rtol": 1e-4})
            x_spec = TensorSpec.from_tensor(shape, x_strides, dtype)
            alpha_spec = TensorSpec.from_tensor(
                (1,), None, dtype, init_mode=TensorInitializer.ONES
            )

            test_cases.append(
                TestCase(
                    inputs=[x_spec, alpha_spec],
                    kwargs={},
                    output_spec=None,
                    comparison_target=0,
                    tolerance=tol,
                    description="scal - INPLACE",
                )
            )

    return test_cases


class OpTest(BaseOperatorTest):
    """BLAS Level-1 scal operator test"""

    def __init__(self):
        super().__init__("Scal")

    def get_test_cases(self):
        return parse_test_cases()

    def torch_operator(self, *args, **kwargs):
        return torch_scal(*args, **kwargs)

    def infinicore_operator(self, *args, **kwargs):
        return infinicore.scal(*args, **kwargs)


def main():
    runner = GenericTestRunner(OpTest)
    runner.run_and_exit()


if __name__ == "__main__":
    main()
