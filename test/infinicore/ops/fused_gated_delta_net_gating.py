import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import torch.nn.functional as F
from framework import (
    BaseOperatorTest,
    GenericTestRunner,
    TensorSpec,
    TestCase,
)

import infinicore

_TEST_CASES_DATA = [
    ((2, 1, 8), None, None),
    ((2, 3, 17), None, None),
    ((2, 3, 17), (80, 20, 1), (2,)),
]

_TENSOR_DTYPES = [infinicore.float16, infinicore.float32, infinicore.bfloat16]

_TOLERANCE_MAP = {
    infinicore.float16: {"atol": 2e-3, "rtol": 2e-3},
    infinicore.float32: {"atol": 1e-6, "rtol": 1e-6},
    infinicore.bfloat16: {"atol": 2e-2, "rtol": 2e-2},
}


def torch_fused_gdn_gating(A_log, a, b, dt_bias, beta=1.0, threshold=20.0, out=None):
    x = a.float() + dt_bias.float().view(1, 1, -1)
    softplus_x = torch.where(
        beta * x <= threshold,
        F.softplus(x, beta=beta, threshold=threshold),
        x,
    )
    g = -A_log.float().exp().view(1, 1, -1) * softplus_x
    beta_output = b.float().sigmoid()
    if b.shape[1] != 1:
        beta_output = beta_output.to(b.dtype).float()
    if out is not None:
        out_g, out_beta = out
        out_g.copy_(g)
        out_beta.copy_(beta_output)
        return out
    return g, beta_output


def parse_test_cases():
    tests = []
    for shape, tensor_strides, hidden_strides in _TEST_CASES_DATA:
        hidden = shape[-1]
        for dtype in _TENSOR_DTYPES:
            tol = _TOLERANCE_MAP[dtype]
            A_log = TensorSpec.from_tensor((hidden,), hidden_strides, dtype)
            a = TensorSpec.from_tensor(shape, tensor_strides, dtype)
            b = TensorSpec.from_tensor(shape, tensor_strides, dtype)
            dt_bias = TensorSpec.from_tensor((hidden,), hidden_strides, dtype)
            kwargs = {"beta": 1.0, "threshold": 20.0}

            tests.append(
                TestCase(
                    inputs=[A_log, a, b, dt_bias],
                    kwargs=kwargs,
                    output_spec=None,
                    comparison_target=None,
                    tolerance=tol,
                    description="FusedGatedDeltaNetGating - OUT_OF_PLACE",
                    output_count=2,
                )
            )

            out_g = TensorSpec.from_tensor(shape, None, infinicore.float32)
            out_beta = TensorSpec.from_tensor(shape, None, infinicore.float32)
            tests.append(
                TestCase(
                    inputs=[A_log, a, b, dt_bias],
                    kwargs=kwargs.copy(),
                    output_specs=[out_g, out_beta],
                    comparison_target="out",
                    tolerance=tol,
                    description="FusedGatedDeltaNetGating - INPLACE(out)",
                    output_count=2,
                )
            )

            if dtype == infinicore.bfloat16:
                tests.append(
                    TestCase(
                        inputs=[
                            TensorSpec.from_tensor(
                                (hidden,), hidden_strides, infinicore.float32
                            ),
                            a,
                            b,
                            TensorSpec.from_tensor(
                                (hidden,), hidden_strides, infinicore.float32
                            ),
                        ],
                        kwargs=kwargs.copy(),
                        output_spec=None,
                        comparison_target=None,
                        tolerance=tol,
                        description=(
                            "FusedGatedDeltaNetGating - BF16 activations "
                            "with FP32 parameters"
                        ),
                        output_count=2,
                    )
                )
    return tests


class OpTest(BaseOperatorTest):
    def __init__(self):
        super().__init__("FusedGatedDeltaNetGating")

    def get_test_cases(self):
        return parse_test_cases()

    def torch_operator(self, *args, **kwargs):
        return torch_fused_gdn_gating(*args, **kwargs)

    def infinicore_operator(self, *args, **kwargs):
        return infinicore.nn.functional.fused_gated_delta_net_gating(*args, **kwargs)


def main():
    runner = GenericTestRunner(OpTest)
    runner.run_and_exit()


if __name__ == "__main__":
    main()
