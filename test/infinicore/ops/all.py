import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import infinicore
from framework.base import BaseOperatorTest, TensorSpec, TestCase
from framework.runner import GenericTestRunner
from framework.utils import is_broadcast

# Test cases format: (in_shape, in_strides_or_None, dim_or_None, keepdim_or_None, out_strides_or_None)

_TEST_CASES_DATA = [
    ((8, 8), None, None, None, None),
    ((8, 8), (16, 1), 1, False, None),
    ((2, 3, 4), None, 0, True, (0, 1, 1)),
    ((1, 8), None, (0, 1), False, None),
    ((16, 64), (128, 1), None, None, None),
    ((4, 5, 6), (60, 12, 2), 2, True, (12, 4, 1)),
]

_TOLERANCE_MAP = {infinicore.bool: {"atol": 0, "rtol": 0}}

_TENSOR_DTYPES = [infinicore.bool, infinicore.uint8]


def _compute_out_shape(shape, dim, keepdim):
    if dim is None:
        return ()
    if isinstance(dim, tuple):
        dims = sorted([(d if d >= 0 else len(shape) + d) for d in dim])
        if keepdim:
            out = list(shape)
            for d in dims:
                out[d] = 1
            return tuple(out)
        else:
            return tuple(s for i, s in enumerate(shape) if i not in dims)
    else:
        d = dim if dim >= 0 else len(shape) + dim
        if keepdim:
            out = list(shape)
            out[d] = 1
            return tuple(out)
        else:
            return tuple(s for i, s in enumerate(shape) if i != d)


def parse_test_cases():
    test_cases = []
    for data in _TEST_CASES_DATA:
        shape, strides, dim, keepdim, out_strides = data
        input_supports_inplace = not is_broadcast(strides)
        out_supports_inplace = not is_broadcast(out_strides)

        for dtype in _TENSOR_DTYPES:
            tol = _TOLERANCE_MAP.get(dtype, {"atol": 0, "rtol": 0})
            in_spec = TensorSpec.from_tensor(shape, strides, dtype)

            # Out-of-place
            kwargs = {}
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
                    description="All - OUT_OF_PLACE",
                )
            )

            # explicit out when supported (create out tensor with computed shape)
            out_shape = _compute_out_shape(shape, dim, keepdim)
            out_spec = TensorSpec.from_tensor(out_shape, out_strides, infinicore.bool)
            if out_supports_inplace:
                test_cases.append(
                    TestCase(
                        inputs=[in_spec],
                        kwargs=kwargs,
                        output_spec=out_spec,
                        comparison_target="out",
                        tolerance=tol,
                        description="All - INPLACE(out)",
                    )
                )

    return test_cases


class OpTest(BaseOperatorTest):
    """All operator test with simplified implementation"""

    def __init__(self):
        super().__init__("All")

    def get_test_cases(self):
        return parse_test_cases()

    def torch_operator(self, *args, **kwargs):
        return torch.all(*args, **kwargs)

    # def infinicore_operator(self, *args, **kwargs):
    #     """InfiniCore implementation (operator not yet available)."""
    #     return infinicore.all(*args, **kwargs)


def main():
    """Main entry point"""
    runner = GenericTestRunner(OpTest)
    runner.run_and_exit()


if __name__ == "__main__":
    main()
