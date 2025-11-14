import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import infinicore
from framework.base import BaseOperatorTest, TensorSpec, TestCase
from framework.runner import GenericTestRunner
from framework.utils import is_broadcast

# Test cases format: (input_shape, mat_shape, vec_shape, input_strides_or_None, mat_strides_or_None, vec_strides_or_None, beta_or_None, alpha_or_None)

_TEST_CASES_DATA = [
    ((4,), (4, 6), (6,), None, None, None, None, None),
    ((8,), (8, 8), (8,), None, None, None, 0.0, 1.0),
    ((3,), (3, 5), (5,), None, (15, 1), None, None, 0.5),
    ((16,), (16, 32), (32,), None, (512, 1), None, 1.0, None),
    ((1,), (1, 1), (1,), None, None, None, None, None),
    ((12,), (12, 12), (12,), None, None, None, None, None),
]

_TOLERANCE_MAP = {
    infinicore.float16: {"atol": 0, "rtol": 1e-2},
    infinicore.float32: {"atol": 0, "rtol": 1e-3},
    infinicore.bfloat16: {"atol": 0, "rtol": 5e-2},
}

_TENSOR_DTYPES = [infinicore.float16, infinicore.bfloat16, infinicore.float32]


def parse_test_cases():
    test_cases = []

    for d in _TEST_CASES_DATA:
        in_shape, mat_shape, vec_shape = d[0], d[1], d[2]
        in_strides, mat_strides, vec_strides = d[3], d[4], d[5]
        beta, alpha = d[6], d[7]

        out_supports_inplace = not is_broadcast(in_strides)

        for dtype in _TENSOR_DTYPES:
            tol = _TOLERANCE_MAP.get(dtype, {"atol": 0, "rtol": 1e-3})
            in_spec = TensorSpec.from_tensor(in_shape, in_strides, dtype)
            mat_spec = TensorSpec.from_tensor(mat_shape, mat_strides, dtype)
            vec_spec = TensorSpec.from_tensor(vec_shape, vec_strides, dtype)

            kwargs = {}
            if beta is not None:
                kwargs["beta"] = beta
            if alpha is not None:
                kwargs["alpha"] = alpha

            test_cases.append(
                TestCase(
                    inputs=[in_spec, mat_spec, vec_spec],
                    kwargs=kwargs,
                    output_spec=None,
                    comparison_target=None,
                    tolerance=tol,
                    description="addmv - OUT_OF_PLACE",
                )
            )

            if out_supports_inplace:
                test_cases.append(
                    TestCase(
                        inputs=[in_spec, mat_spec, vec_spec],
                        kwargs=kwargs,
                        output_spec=in_spec,
                        comparison_target="out",
                        tolerance=tol,
                        description="addmv - INPLACE(out)",
                    )
                )

    return test_cases


class OpTest(BaseOperatorTest):
    """addmv operator test with simplified implementation"""

    def __init__(self):
        super().__init__("addmv")

    def get_test_cases(self):
        return parse_test_cases()

    def torch_operator(self, *args, **kwargs):
        return torch.addmv(*args, **kwargs)

    # def infinicore_operator(self, *args, **kwargs):
    #     """InfiniCore implementation (operator not yet available)."""
    #     return infinicore.addmv(*args, **kwargs)


def main():
    """Main entry point"""
    runner = GenericTestRunner(OpTest)
    runner.run_and_exit()


if __name__ == "__main__":
    main()
