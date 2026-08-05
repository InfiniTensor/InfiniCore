import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
from framework import BaseOperatorTest, GenericTestRunner, TensorSpec, TestCase

import infinicore

_DTYPES = [infinicore.float16, infinicore.bfloat16, infinicore.float32]
_CASES = [
    ((2, 3, 4), None, (2, 4, 5), None, (2, 3, 5), None),
    ((2, 3, 4), (20, 5, 1), (2, 4, 5), (28, 7, 1), (2, 3, 5), None),
    ((2, 3, 4), None, (2, 4, 5), None, (2, 3, 5), (15, 1, 3)),
]


class OpTest(BaseOperatorTest):
    def __init__(self):
        super().__init__("BmmStrided")

    def get_test_cases(self):
        cases = []
        for a_shape, a_strides, b_shape, b_strides, out_shape, out_strides in _CASES:
            for dtype in _DTYPES:
                cases.append(
                    TestCase(
                        inputs=[
                            TensorSpec.from_tensor(a_shape, a_strides, dtype),
                            TensorSpec.from_tensor(b_shape, b_strides, dtype),
                        ],
                        output_spec=TensorSpec.from_tensor(
                            out_shape, out_strides, dtype
                        ),
                        comparison_target="out",
                        tolerance={"atol": 1e-3, "rtol": 5e-2},
                        description=f"bmm with output strides {out_strides}",
                    )
                )
        return cases

    def torch_operator(self, a, b, *, out):
        torch.bmm(a, b, out=out)
        return out

    def infinicore_operator(self, a, b, *, out):
        return infinicore.bmm_strided(a, b, out=out)


if __name__ == "__main__":
    GenericTestRunner(OpTest).run_and_exit()
