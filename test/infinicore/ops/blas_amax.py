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
    ((3,), None),
    ((8,), (2,)),
    ((32,), None),
    ((257,), (3,)),
    ((65535,), None),
]

_TENSOR_DTYPES = [
    # infinicore.float16,
    infinicore.float32,
    # infinicore.float64,
    # infinicore.bfloat16,
]

_TOLERANCE = {"atol": 0, "rtol": 0}


def torch_blas_amax(x, *, out=None):
    result = torch.argmax(x.abs()).to(torch.int32) + 1
    if out is None:
        return result

    out.copy_(result)
    return out


def parse_test_cases():
    test_cases = []
    for shape, x_strides in _TEST_CASES_DATA:
        for dtype in _TENSOR_DTYPES:
            x_spec = TensorSpec.from_tensor(shape, x_strides, dtype)
            out_spec = TensorSpec.from_tensor(
                (), None, infinicore.int32, init_mode=TensorInitializer.ZEROS
            )

            test_cases.append(
                TestCase(
                    inputs=[x_spec],
                    kwargs={},
                    output_spec=None,
                    comparison_target=None,
                    tolerance=_TOLERANCE,
                    description="blas_amax - OUT_OF_PLACE",
                )
            )

            test_cases.append(
                TestCase(
                    inputs=[x_spec],
                    kwargs={},
                    output_spec=out_spec,
                    comparison_target="out",
                    tolerance=_TOLERANCE,
                    description="blas_amax - INPLACE(out)",
                )
            )

    return test_cases


class OpTest(BaseOperatorTest):
    """BLAS Level-1 amax operator test"""

    def __init__(self):
        super().__init__("BlasAmax")

    def get_test_cases(self):
        return parse_test_cases()

    def torch_operator(self, *args, **kwargs):
        return torch_blas_amax(*args, **kwargs)

    def infinicore_operator(self, *args, **kwargs):
        return infinicore.blas_amax(*args, **kwargs)


def main():
    runner = GenericTestRunner(OpTest)
    runner.run_and_exit()


if __name__ == "__main__":
    main()
