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
    (1.0, 2.0, 3.0, 4.0),
    (2.5, 0.5, -1.2, 0.8),
    (3.0, 4.0, 0.0, 2.0),
    (1.5, 1.5, 2.0, -3.0),
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
    infinicore.float64: {"atol": 1e-12, "rtol": 1e-12},
    infinicore.bfloat16: {"atol": 1e-2, "rtol": 1e-2},
}


def _rotmg_values(d1, d2, x1, y1):
    zero = 0.0
    one = 1.0
    two = 2.0
    gam = 4096.0
    gamsq = 1.67772e7
    rgamsq = 5.96046e-8

    param = [0.0] * 5
    sh11 = sh12 = sh21 = sh22 = 0.0

    if d1 < zero:
        sflag = -one
        d1 = d2 = x1 = zero
    else:
        sp2 = d2 * y1
        if sp2 == zero:
            param[0] = -two
            return d1, d2, x1, param

        sp1 = d1 * x1
        sq2 = sp2 * y1
        sq1 = sp1 * x1

        if abs(sq1) > abs(sq2):
            sh21 = -y1 / x1
            sh12 = sp2 / sp1
            su = one - sh12 * sh21
            if su > zero:
                sflag = zero
                d1 = d1 / su
                d2 = d2 / su
                x1 = x1 * su
            else:
                sflag = -one
                sh11 = sh12 = sh21 = sh22 = zero
                d1 = d2 = x1 = zero
        else:
            if sq2 < zero:
                sflag = -one
                d1 = d2 = x1 = zero
            else:
                sflag = one
                sh11 = sp1 / sp2
                sh22 = x1 / y1
                su = one + sh11 * sh22
                stemp = d2 / su
                d2 = d1 / su
                d1 = stemp
                x1 = y1 * su

        if d1 != zero:
            while d1 <= rgamsq or d1 >= gamsq:
                if sflag == zero:
                    sh11 = one
                    sh22 = one
                    sflag = -one
                else:
                    sh21 = -one
                    sh12 = one
                    sflag = -one
                if d1 <= rgamsq:
                    d1 = d1 * gam * gam
                    x1 = x1 / gam
                    sh11 = sh11 / gam
                    sh12 = sh12 / gam
                else:
                    d1 = d1 / (gam * gam)
                    x1 = x1 * gam
                    sh11 = sh11 * gam
                    sh12 = sh12 * gam

        if d2 != zero:
            while abs(d2) <= rgamsq or abs(d2) >= gamsq:
                if sflag == zero:
                    sh11 = one
                    sh22 = one
                    sflag = -one
                else:
                    sh21 = -one
                    sh12 = one
                    sflag = -one
                if abs(d2) <= rgamsq:
                    d2 = d2 * gam * gam
                    sh21 = sh21 / gam
                    sh22 = sh22 / gam
                else:
                    d2 = d2 / (gam * gam)
                    sh21 = sh21 * gam
                    sh22 = sh22 * gam

    if sflag < zero:
        param[1] = sh11
        param[2] = sh21
        param[3] = sh12
        param[4] = sh22
    elif sflag == zero:
        param[2] = sh21
        param[3] = sh12
    else:
        param[1] = sh11
        param[4] = sh22

    param[0] = sflag
    return d1, d2, x1, param


def torch_rotmg(d1, d2, x1, y1, param):
    out_d1, out_d2, out_x1, out_param = _rotmg_values(
        d1.item(), d2.item(), x1.item(), y1.item()
    )
    d1.fill_(out_d1)
    d2.fill_(out_d2)
    x1.fill_(out_x1)
    param.copy_(torch.tensor(out_param, dtype=param.dtype, device=param.device))
    return d1, d2, x1, param


def parse_test_cases():
    test_cases = []
    for d1_value, d2_value, x1_value, y1_value in _TEST_CASES_DATA:
        for dtype in _TENSOR_DTYPES:
            tol = _TOLERANCE_MAP.get(dtype, {"atol": 1e-5, "rtol": 1e-4})
            d1_spec = TensorSpec.from_tensor(
                (1,),
                None,
                dtype,
                init_mode=TensorInitializer.MANUAL,
                set_tensor=torch.tensor([d1_value]),
            )
            d2_spec = TensorSpec.from_tensor(
                (1,),
                None,
                dtype,
                init_mode=TensorInitializer.MANUAL,
                set_tensor=torch.tensor([d2_value]),
            )
            x1_spec = TensorSpec.from_tensor(
                (1,),
                None,
                dtype,
                init_mode=TensorInitializer.MANUAL,
                set_tensor=torch.tensor([x1_value]),
            )
            y1_spec = TensorSpec.from_tensor(
                (1,),
                None,
                dtype,
                init_mode=TensorInitializer.MANUAL,
                set_tensor=torch.tensor([y1_value]),
            )
            param_spec = TensorSpec.from_tensor(
                (5,),
                None,
                dtype,
                init_mode=TensorInitializer.ZEROS,
            )

            test_cases.append(
                TestCase(
                    inputs=[d1_spec, d2_spec, x1_spec, y1_spec, param_spec],
                    kwargs={},
                    comparison_target=[0, 1, 2, 4],
                    tolerance=tol,
                    output_count=4,
                    description="rotmg - INPLACE",
                )
            )

    return test_cases


class OpTest(BaseOperatorTest):
    """BLAS Level-1 rotmg operator test"""

    def __init__(self):
        super().__init__("Rotmg")

    def get_test_cases(self):
        return parse_test_cases()

    def torch_operator(self, *args, **kwargs):
        return torch_rotmg(*args, **kwargs)

    def infinicore_operator(self, *args, **kwargs):
        return infinicore.rotmg(*args, **kwargs)


def main():
    runner = GenericTestRunner(OpTest)
    runner.run_and_exit()


if __name__ == "__main__":
    main()
