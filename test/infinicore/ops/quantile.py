import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import infinicore
from framework import (
    BaseOperatorTest,
    TensorSpec,
    TestCase,
    GenericTestRunner,
    is_broadcast,
)

# Test cases format: (in_shape, in_strides_or_None, q_or_None, dim_or_None, keepdim_or_None, out_strides_or_None)
# quantile computes quantiles along dim or overall. q may be float or tensor

_TEST_CASES_DATA = [
    ((8, 8), None, 0.5, None, None, None),
    ((8, 8), (16, 1), 0.25, 1, False, None),
    ((2, 3, 4), None, 0.75, 2, True, (0, 1, 1)),
    ((16, 64), (128, 1), 0.5, None, None, None),
    ((4, 5, 6), (60, 12, 2), 0.5, 2, True, (12, 4, 1)),
]

_TOLERANCE_MAP = {
    infinicore.float32: {"atol": 1e-5, "rtol": 1e-4},
}

_TENSOR_DTYPES = [infinicore.float32]


def _compute_out_shape(shape, dim, keepdim, q_is_tensor=False):
    # if q is tensor with len>1, output shape may include q dim; keep simple: when q is tensor, return (len(q), ...) prefix
    if dim is None:
        base = ()
    else:
        if isinstance(dim, tuple):
            dims = sorted([(d if d >= 0 else len(shape) + d) for d in dim])
        else:
            dims = [dim]
        if keepdim:
            out = list(shape)
            for d in dims:
                out[d] = 1
            base = tuple(out)
        else:
            base = tuple(s for i, s in enumerate(shape) if i not in dims)

    if q_is_tensor:
        # Prepend q-length as first dim
        return (2,) + base
    return base


def parse_test_cases():
    test_cases = []
    for data in _TEST_CASES_DATA:
        shape, strides, q, dim, keepdim, out_strides = data
        q_is_tensor = isinstance(q, torch.Tensor)
        out_supports_inplace = not is_broadcast(out_strides)

        for dtype in _TENSOR_DTYPES:
            tol = _TOLERANCE_MAP.get(dtype, {"atol": 1e-5, "rtol": 1e-3})
            in_spec = TensorSpec.from_tensor(shape, strides, dtype)

            kwargs = {"q": q}
            if dim is not None:
                kwargs["dim"] = dim
            if keepdim is not None:
                kwargs["keepdim"] = keepdim

            test_cases.append(
                TestCase(
                    inputs=[in_spec],
                    kwargs=kwargs,
                    output_spec=None,
                    comparison_target=None,
                    tolerance=tol,
                    description="Quantile - OUT_OF_PLACE",
                )
            )

            out_shape = _compute_out_shape(shape, dim, keepdim, q_is_tensor=q_is_tensor)
            out_spec = TensorSpec.from_tensor(out_shape, out_strides, dtype)
            if out_supports_inplace:
                test_cases.append(
                    TestCase(
                        inputs=[in_spec],
                        kwargs=kwargs,
                        output_spec=out_spec,
                        comparison_target="out",
                        tolerance=tol,
                        description="Quantile - INPLACE(out)",
                    )
                )

    return test_cases


class OpTest(BaseOperatorTest):
    """Quantile operator test with simplified implementation"""

    def __init__(self):
        super().__init__("Quantile")

    def get_test_cases(self):
        return parse_test_cases()

    def torch_operator(self, *args, **kwargs):
        return torch.quantile(*args, **kwargs)

    # def infinicore_operator(self, *args, **kwargs):
    #     """InfiniCore implementation (operator not yet available)."""
    #     return infinicore.quantile(*args, **kwargs)


def main():
    """Main entry point"""
    runner = GenericTestRunner(OpTest)
    runner.run_and_exit()


if __name__ == "__main__":
    main()
