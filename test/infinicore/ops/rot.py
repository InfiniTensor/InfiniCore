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
    ((13,), None, None),
    ((13,), (10,), (10,)),
    ((5632,), None, None),
    ((5632,), (5,), (5,)),
    ((16,), (4,), (4,)),
    ((5632,), (32,), (32,)),
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


def torch_rot(x, y, c, s):
    x0 = x.clone()
    y0 = y.clone()
    x.copy_(c * x0 + s * y0)
    y.copy_(c * y0 - s * x0)
    return x, y


def parse_test_cases():
    test_cases = []
    for shape, x_strides, y_strides in _TEST_CASES_DATA:
        for dtype in _TENSOR_DTYPES:
            tol = _TOLERANCE_MAP.get(dtype, {"atol": 1e-5, "rtol": 1e-4})
            x_spec = TensorSpec.from_tensor(shape, x_strides, dtype)
            y_spec = TensorSpec.from_tensor(shape, y_strides, dtype)
            c_spec = TensorSpec.from_tensor(
                (),
                None,
                dtype,
                init_mode=TensorInitializer.MANUAL,
                set_tensor=torch.tensor(0.6),
            )
            s_spec = TensorSpec.from_tensor(
                (),
                None,
                dtype,
                init_mode=TensorInitializer.MANUAL,
                set_tensor=torch.tensor(0.8),
            )

            test_cases.append(
                TestCase(
                    inputs=[x_spec, y_spec, c_spec, s_spec],
                    kwargs={},
                    comparison_target=[0, 1],
                    tolerance=tol,
                    output_count=2,
                    description="rot - INPLACE",
                )
            )

    return test_cases


class OpTest(BaseOperatorTest):
    """BLAS Level-1 rot operator test"""

    def __init__(self):
        super().__init__("Rot")

    def get_test_cases(self):
        return parse_test_cases()

    def torch_operator(self, *args, **kwargs):
        return torch_rot(*args, **kwargs)

    def infinicore_operator(self, *args, **kwargs):
        return infinicore.rot(*args, **kwargs)


def main():
    runner = GenericTestRunner(OpTest)
    runner.run_and_exit()


if __name__ == "__main__":
    main()
