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
    ((13,), None, None, (-1.0, 1.2, -0.3, 0.4, 0.8)),
    ((13,), (10,), (10,), (0.0, 0.0, -0.25, 0.5, 0.0)),
    ((5632,), None, None, (1.0, 1.1, 0.0, 0.0, 0.9)),
    ((5632,), (5,), (5,), (-2.0, 0.0, 0.0, 0.0, 0.0)),
]

_TENSOR_DTYPES = [
    # infinicore.float16,
    infinicore.float32,
    # infinicore.float64,
    # infinicore.bfloat16,
]

_TOLERANCE_MAP = {
    infinicore.float16: {"atol": 1e-3, "rtol": 1e-3},
    infinicore.float32: {"atol": 1e-5, "rtol": 1e-5},
    infinicore.float64: {"atol": 1e-9, "rtol": 1e-9},
    infinicore.bfloat16: {"atol": 1e-2, "rtol": 1e-2},
}


def torch_rotm(x, y, param):
    sflag, sh11, sh21, sh12, sh22 = param
    if sflag == -2.0:
        return x, y

    w = x.clone()
    z = y.clone()

    if sflag < 0.0:
        x.copy_(w * sh11 + z * sh12)
        y.copy_(w * sh21 + z * sh22)
    elif sflag == 0.0:
        x.copy_(w + z * sh12)
        y.copy_(w * sh21 + z)
    else:
        x.copy_(w * sh11 + z)
        y.copy_(-w + sh22 * z)
    return x, y


def parse_test_cases():
    test_cases = []
    for shape, x_strides, y_strides, param in _TEST_CASES_DATA:
        for dtype in _TENSOR_DTYPES:
            tol = _TOLERANCE_MAP.get(dtype, {"atol": 1e-5, "rtol": 1e-4})
            x_spec = TensorSpec.from_tensor(shape, x_strides, dtype)
            y_spec = TensorSpec.from_tensor(shape, y_strides, dtype)
            param_spec = TensorSpec.from_tensor(
                (5,),
                None,
                dtype,
                init_mode=TensorInitializer.MANUAL,
                set_tensor=torch.tensor(param),
            )

            test_cases.append(
                TestCase(
                    inputs=[x_spec, y_spec, param_spec],
                    kwargs={},
                    comparison_target=[0, 1],
                    tolerance=tol,
                    output_count=2,
                    description="rotm - INPLACE",
                )
            )

    return test_cases


class OpTest(BaseOperatorTest):
    """BLAS Level-1 rotm operator test"""

    def __init__(self):
        super().__init__("Rotm")

    def get_test_cases(self):
        return parse_test_cases()

    def torch_operator(self, *args, **kwargs):
        return torch_rotm(*args, **kwargs)

    def infinicore_operator(self, *args, **kwargs):
        return infinicore.rotm(*args, **kwargs)


def main():
    runner = GenericTestRunner(OpTest)
    runner.run_and_exit()


if __name__ == "__main__":
    main()
