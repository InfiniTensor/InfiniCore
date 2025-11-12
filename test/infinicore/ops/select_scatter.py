import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import infinicore
from framework.base import BaseOperatorTest, TensorSpec, TestCase
from framework.runner import GenericTestRunner
from framework.utils import is_broadcast
from framework.tensor import TensorInitializer

# Test cases format: (shape, input_strides, index_strides, src_strides, dim)
_TEST_CASES_DATA = [
    ((6, 8), None, None, None, 1),
    ((8, 4), (16, 1), None, None, 0),
    ((5, 5), None, None, (10, 1), 1),
    ((3, 7), None, (14, 1), None, 1),
    ((10, 3), (30, 1), (30, 1), (30, 1), 0),
    ((2, 16), None, None, None, 1),
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
        shape, in_strides, idx_strides, src_strides, dim = data

        in_supports_inplace = not is_broadcast(in_strides)
        out_supports_inplace = not is_broadcast(src_strides)

        for dtype in _TENSOR_DTYPES:
            tol = _TOLERANCE_MAP.get(dtype, {"atol": 0, "rtol": 1e-3})

            input_spec = TensorSpec.from_tensor(shape, in_strides, dtype)
            src_shape = tuple(s for i, s in enumerate(shape) if i != dim)
            src_strides_for_src = (
                src_strides
                if (src_strides and len(src_strides) == len(src_shape))
                else None
            )
            src_spec = TensorSpec.from_tensor(src_shape, src_strides_for_src, dtype)

            high = max(1, shape[dim])
            index_spec = TensorSpec.from_tensor(
                shape,
                idx_strides,
                infinicore.int64,
                init_mode=TensorInitializer.RANDINT,
                low=0,
                high=high,
            )

            index_val = 0 if shape[dim] <= 1 else (shape[dim] // 2)

            test_cases.append(
                TestCase(
                    inputs=[input_spec, src_spec, dim, index_val],
                    kwargs=None,
                    output_spec=None,
                    comparison_target=None,
                    tolerance=tol,
                    description=f"select_scatter - OUT_OF_PLACE",
                )
            )

    return test_cases


class OpTest(BaseOperatorTest):
    """SelectScatter operator test with simplified implementation"""

    def __init__(self):
        super().__init__("SelectScatter")

    def get_test_cases(self):
        return parse_test_cases()

    def torch_operator(self, *args, **kwargs):
        return torch.select_scatter(*args, **kwargs)

    # def infinicore_operator(self, *args, **kwargs):
    #     """InfiniCore implementation (operator not yet available)."""
    #     return infinicore.select_scatter(*args, **kwargs)


def main():
    """Main entry point"""
    runner = GenericTestRunner(OpTest)
    runner.run_and_exit()


if __name__ == "__main__":
    main()
