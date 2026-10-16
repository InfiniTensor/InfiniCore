import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import infinicore
import torch
from framework import BaseOperatorTest, GenericTestRunner, TensorSpec, TestCase


_TEST_CASES = (
    (1, 128),
    (4, 128),
    (1, 7168),
    (8, 7168),
)

_DTYPES = (infinicore.float16, infinicore.bfloat16)
_EPSILON = 1e-5


class OpTest(BaseOperatorTest):
    def __init__(self):
        super().__init__("AddRMSNormInplace")

    def get_test_cases(self):
        cases = []
        for shape in _TEST_CASES:
            for dtype in _DTYPES:
                tensor_spec = TensorSpec.from_tensor(shape, None, dtype)
                residual_spec = TensorSpec.from_tensor(shape, None, dtype)
                weight_spec = TensorSpec.from_tensor((shape[-1],), None, dtype)
                cases.append(
                    TestCase(
                        inputs=[tensor_spec, residual_spec, weight_spec],
                        kwargs={"epsilon": _EPSILON},
                        output_count=2,
                        tolerance={"atol": 1e-2, "rtol": 1e-2},
                        description=f"AddRMSNormInplace shape={shape} dtype={dtype}",
                    )
                )
        return cases

    def torch_operator(self, input, residual, weight, epsilon=_EPSILON, **kwargs):
        summed = (input.float() + residual.float()).to(input.dtype)
        normalized = summed.float() * torch.rsqrt(
            summed.float().pow(2).mean(dim=-1, keepdim=True) + epsilon
        )
        normalized *= weight.float()
        residual.copy_(summed)
        input.copy_(normalized.to(input.dtype))
        return input, residual

    def infinicore_operator(
        self, input, residual, weight, epsilon=_EPSILON, **kwargs
    ):
        infinicore.add_rms_norm_inplace(input, residual, weight, epsilon)
        return input, residual


def main():
    GenericTestRunner(OpTest).run_and_exit()


if __name__ == "__main__":
    main()
