import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import infinicore
from framework.base import BaseOperatorTest, TensorSpec, TestCase
from framework.runner import GenericTestRunner
from framework.utils import is_broadcast

# Test cases format: (in_shape, in_strides_or_None, dim, keepdim_or_None, out_strides_or_None)
# logsumexp computes log(sum(exp(input), dim=dim)) with numerical stability

_TEST_CASES_DATA = [
    ((8, 8), None, 1, None, None),
    ((8, 8), (16, 1), 0, False, None),
    ((2, 3, 4), None, 2, True, (0, 1, 1)),
    ((1, 8), None, 0, False, None),
    ((16, 64), (128, 1), 1, True, None),
    ((4, 5, 6), (60, 12, 2), 2, True, (12, 4, 1)),
]

_TOLERANCE_MAP = {
    infinicore.float16: {"atol": 1e-3, "rtol": 1e-2},
    infinicore.float32: {"atol": 1e-5, "rtol": 1e-4},
}

_TENSOR_DTYPES = [infinicore.float16, infinicore.float32]


def _compute_out_shape(shape, dim, keepdim):
    if isinstance(dim, tuple):
        dims = sorted([(d if d >= 0 else len(shape) + d) for d in dim])
    else:
        dims = [dim]

    if dim is None:
        return ()
    if keepdim:
        out = list(shape)
        for d in dims:
            out[d] = 1
        return tuple(out)
    else:
        return tuple(s for i, s in enumerate(shape) if i not in dims)


def parse_test_cases():
    test_cases = []
    for data in _TEST_CASES_DATA:
        shape, strides, dim, keepdim, out_strides = data
        out_supports_inplace = not is_broadcast(out_strides)

        for dtype in _TENSOR_DTYPES:
            tol = _TOLERANCE_MAP.get(dtype, {"atol": 1e-5, "rtol": 1e-3})
            in_spec = TensorSpec.from_tensor(shape, strides, dtype)

            kwargs = {"dim": dim}
            if keepdim is not None:
                kwargs["keepdim"] = keepdim

            test_cases.append(
                TestCase(
                    inputs=[in_spec],
                    kwargs=kwargs,
                    output_spec=None,
                    comparison_target=None,
                    tolerance=tol,
                    description="LogSumExp - OUT_OF_PLACE",
                )
            )

            out_shape = _compute_out_shape(shape, dim, keepdim)
            out_spec = TensorSpec.from_tensor(out_shape, out_strides, dtype)
            if out_supports_inplace:
                test_cases.append(
                    TestCase(
                        inputs=[in_spec],
                        kwargs=kwargs,
                        output_spec=out_spec,
                        comparison_target="out",
                        tolerance=tol,
                        description="LogSumExp - INPLACE(out)",
                    )
                )

    return test_cases


class OpTest(BaseOperatorTest):
    """LogSumExp operator test with simplified implementation"""

    def __init__(self):
        super().__init__("LogSumExp")

    def get_test_cases(self):
        return parse_test_cases()

    def torch_operator(self, *args, **kwargs):
        return torch.logsumexp(*args, **kwargs)

    # def infinicore_operator(self, *args, **kwargs):
    #     """InfiniCore implementation (operator not yet available)."""
    #     return infinicore.logsumexp(*args, **kwargs)


def main():
    """Main entry point"""
    runner = GenericTestRunner(OpTest)
    runner.run_and_exit()


if __name__ == "__main__":
    main()
