import math
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
    (0.0, 0.0),
    (3.0, 4.0),
    (-2.5, 5.0),
    (7.0, -1.5),
    (-3.2, -8.4),
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
    infinicore.float64: {"atol": 1e-7, "rtol": 1e-7},
    infinicore.bfloat16: {"atol": 1e-2, "rtol": 1e-2},
}


def torch_rotg(a, b, c, s):
    a0 = a.item()
    b0 = b.item()
    anorm = abs(a0)
    bnorm = abs(b0)
    if bnorm == 0.0:
        a.fill_(a0)
        b.zero_()
        c.fill_(1.0)
        s.zero_()
        return a, b, c, s
    if anorm == 0.0:
        a.fill_(b0)
        b.fill_(1.0)
        c.zero_()
        s.fill_(1.0)
        return a, b, c, s

    sigma = math.copysign(1.0, a0 if anorm > bnorm else b0)
    r = sigma * math.hypot(a0, b0)
    c0 = a0 / r
    s0 = b0 / r
    if anorm > bnorm:
        z = s0
    elif c0 != 0.0:
        z = 1.0 / c0
    else:
        z = 1.0

    a.fill_(r)
    b.fill_(z)
    c.fill_(c0)
    s.fill_(s0)
    return a, b, c, s


def parse_test_cases():
    test_cases = []
    for a_value, b_value in _TEST_CASES_DATA:
        for dtype in _TENSOR_DTYPES:
            tol = _TOLERANCE_MAP.get(dtype, {"atol": 1e-5, "rtol": 1e-4})
            a_spec = TensorSpec.from_tensor(
                (1,),
                None,
                dtype,
                init_mode=TensorInitializer.MANUAL,
                set_tensor=torch.tensor([a_value]),
            )
            b_spec = TensorSpec.from_tensor(
                (1,),
                None,
                dtype,
                init_mode=TensorInitializer.MANUAL,
                set_tensor=torch.tensor([b_value]),
            )
            c_spec = TensorSpec.from_tensor(
                (),
                None,
                dtype,
                init_mode=TensorInitializer.ZEROS,
            )
            s_spec = TensorSpec.from_tensor(
                (),
                None,
                dtype,
                init_mode=TensorInitializer.ZEROS,
            )

            test_cases.append(
                TestCase(
                    inputs=[a_spec, b_spec, c_spec, s_spec],
                    kwargs={},
                    comparison_target=[0, 1, 2, 3],
                    tolerance=tol,
                    output_count=4,
                    description="rotg - INPLACE",
                )
            )

    return test_cases


class OpTest(BaseOperatorTest):
    """BLAS Level-1 rotg operator test"""

    def __init__(self):
        super().__init__("Rotg")

    def get_test_cases(self):
        return parse_test_cases()

    def torch_operator(self, *args, **kwargs):
        return torch_rotg(*args, **kwargs)

    def infinicore_operator(self, *args, **kwargs):
        return infinicore.rotg(*args, **kwargs)


def main():
    runner = GenericTestRunner(OpTest)
    runner.run_and_exit()


if __name__ == "__main__":
    main()
