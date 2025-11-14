import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import infinicore
from framework.base import BaseOperatorTest, TensorSpec, TestCase
from framework.runner import GenericTestRunner
from framework.utils import is_broadcast

# Test cases format: (input_shape, batch1_shape, batch2_shape, input_strides_or_None, batch1_strides_or_None, batch2_strides_or_None, beta_or_None, alpha_or_None)
# addbmm(input, batch1, batch2, beta=1, alpha=1, out=None)

_TEST_CASES_DATA = [
    # small basic (shapes must satisfy: input (M,N), batch1 (B,M,K), batch2 (B,K,N))
    ((3, 5), (2, 3, 4), (2, 4, 5), None, None, None, None, None),
    # larger
    ((8, 8), (4, 8, 8), (4, 8, 8), None, None, None, 0.5, 2.0),
    # strided input
    ((5, 7), (2, 5, 6), (2, 6, 7), (30, 1), (0, 5, 1), None, None, None),
    # batched different strides
    ((2, 2), (4, 2, 3), (4, 3, 2), None, (24, 6, 1), (0, 3, 1), 1.0, None),
    # square
    ((16, 16), (2, 16, 16), (2, 16, 16), None, None, (512, 1, 1), None, 0.1),
    # edge small
    ((1, 1), (1, 1, 1), (1, 1, 1), None, None, None, None, None),
]

_TOLERANCE_MAP = {
    infinicore.float16: {"atol": 0, "rtol": 1e-2},
    infinicore.float32: {"atol": 0, "rtol": 1e-3},
    infinicore.bfloat16: {"atol": 0, "rtol": 5e-2},
}

_TENSOR_DTYPES = [infinicore.float16, infinicore.bfloat16, infinicore.float32]


def parse_test_cases():
    test_cases = []

    for data in _TEST_CASES_DATA:
        in_shape, b1_shape, b2_shape = data[0], data[1], data[2]
        in_strides = data[3] if len(data) > 3 else None
        b1_strides = data[4] if len(data) > 4 else None
        b2_strides = data[5] if len(data) > 5 else None
        beta = data[6] if len(data) > 6 else None
        alpha = data[7] if len(data) > 7 else None

        out_supports_inplace = not is_broadcast(in_strides)

        for dtype in _TENSOR_DTYPES:
            tol = _TOLERANCE_MAP.get(dtype, {"atol": 0, "rtol": 1e-3})

            in_spec = TensorSpec.from_tensor(in_shape, in_strides, dtype)
            b1_spec = TensorSpec.from_tensor(b1_shape, b1_strides, dtype)
            b2_spec = TensorSpec.from_tensor(b2_shape, b2_strides, dtype)

            kwargs = {}
            if beta is not None:
                kwargs["beta"] = beta
            if alpha is not None:
                kwargs["alpha"] = alpha

            # Out-of-place
            test_cases.append(
                TestCase(
                    inputs=[in_spec, b1_spec, b2_spec],
                    kwargs=kwargs,
                    output_spec=None,
                    comparison_target=None,
                    tolerance=tol,
                    description="addbmm - OUT_OF_PLACE",
                )
            )

            # In-place out= provided
            if out_supports_inplace:
                test_cases.append(
                    TestCase(
                        inputs=[in_spec, b1_spec, b2_spec],
                        kwargs=kwargs,
                        output_spec=in_spec,
                        comparison_target="out",
                        tolerance=tol,
                        description="addbmm - INPLACE(out)",
                    )
                )

    return test_cases


class OpTest(BaseOperatorTest):
    """addbmm operator test with simplified implementation"""

    def __init__(self):
        super().__init__("addbmm")

    def get_test_cases(self):
        return parse_test_cases()

    def torch_operator(self, *args, **kwargs):
        return torch.addbmm(*args, **kwargs)

    # def infinicore_operator(self, *args, **kwargs):
    #     """InfiniCore implementation (operator not yet available)."""
    #     return infinicore.addbmm(*args, **kwargs)


def main():
    """Main entry point"""
    runner = GenericTestRunner(OpTest)
    runner.run_and_exit()


if __name__ == "__main__":
    main()
