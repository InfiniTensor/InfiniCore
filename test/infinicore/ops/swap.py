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
    ((13,), None, None),
    ((13,), (10,), (10,)),
    ((5632,), None, None),
    ((5632,), (5,), (5,)),
    ((16,), (4,), (4,)),
    ((5632,), (32,), (32,)),
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


def torch_swap(x, y):
    tmp = x.clone()
    x.copy_(y)
    y.copy_(tmp)
    return x, y


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
                    comparison_target=[0, 1],
                    tolerance=tol,
                    output_count=2,
                    description="swap - INPLACE",
                )
            )

    return test_cases


class OpTest(BaseOperatorTest):
    """BLAS Level-1 swap operator test"""

    def __init__(self):
        super().__init__("Swap")

    def get_test_cases(self):
        return parse_test_cases()

    def torch_operator(self, *args, **kwargs):
        return torch_swap(*args, **kwargs)

    def infinicore_operator(self, *args, **kwargs):
        return infinicore.swap(*args, **kwargs)


def main():
    runner = GenericTestRunner(OpTest)
    runner.run_and_exit()


if __name__ == "__main__":
    main()
