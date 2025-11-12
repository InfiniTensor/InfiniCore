import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import infinicore
from framework.base import BaseOperatorTest, TensorSpec, TestCase
from framework.tensor import TensorInitializer
from framework.runner import GenericTestRunner
from framework.utils import is_broadcast

# Test cases format: (shape, input_strides, index_strides, src_strides, dim, reduce)
_TEST_CASES_DATA = [
    ((6, 6), None, None, None, 1, "sum"),
    ((8, 4), (16, 1), None, None, 0, "amax"),
    ((5, 5), None, None, (10, 1), 1, "mean"),
    ((3, 7), None, (14, 1), None, 1, "amin"),
    ((10, 3), (30, 1), (30, 1), (30, 1), 0, "sum"),
    ((2, 16), None, None, None, 1, "prod"),
]

_TOLERANCE_MAP = {
    infinicore.float16: {"atol": 0, "rtol": 1e-2},
    infinicore.float32: {"atol": 0, "rtol": 1e-3},
    infinicore.bfloat16: {"atol": 0, "rtol": 5e-2},
}

_TENSOR_DTYPES = [infinicore.float16, infinicore.bfloat16, infinicore.float32]


def parse_test_cases():
    """Format: (shape, input_strides, index_strides, src_strides, dim, reduce)"""
    test_cases = []

    for data in _TEST_CASES_DATA:
        shape, in_strides, idx_strides, src_strides, dim, reduce = data

        in_supports_inplace = not is_broadcast(in_strides)
        out_supports_inplace = not is_broadcast(src_strides)

        for dtype in _TENSOR_DTYPES:
            tol = _TOLERANCE_MAP.get(dtype, {"atol": 0, "rtol": 1e-3})

            input_spec = TensorSpec.from_tensor(shape, in_strides, dtype)
            # index_spec = TensorSpec.from_tensor(shape, idx_strides, infinicore.int64)
            # index tensor spec: 必须生成 [0, shape[dim]-1] 的 int64 值
            high = max(1, shape[dim])
            index_spec = TensorSpec.from_tensor(
                shape,
                idx_strides,
                infinicore.int64,
                init_mode=TensorInitializer.RANDINT,
                low=0,
                high=high,
            )
            src_spec = TensorSpec.from_tensor(shape, src_strides, dtype)

            # Out-of-place
            test_cases.append(
                TestCase(
                    inputs=[input_spec, dim, index_spec, src_spec],
                    kwargs={"reduce": reduce},
                    output_spec=None,
                    comparison_target=None,
                    tolerance=tol,
                    description=f"scatter_reduce - OUT_OF_PLACE",
                )
            )

            # In-place on input
            if in_supports_inplace:
                test_cases.append(
                    TestCase(
                        inputs=[input_spec, dim, index_spec, src_spec],
                        kwargs={"reduce": reduce, "out": 0},
                        output_spec=None,
                        comparison_target=0,
                        tolerance=tol,
                        description=f"scatter_reduce - INPLACE(input)",
                    )
                )

            # Explicit out tensor
            if out_supports_inplace:
                out_spec = TensorSpec.from_tensor(shape, None, dtype)
                test_cases.append(
                    TestCase(
                        inputs=[input_spec, dim, index_spec, src_spec],
                        kwargs={"reduce": reduce},
                        output_spec=out_spec,
                        comparison_target="out",
                        tolerance=tol,
                        description=f"scatter_reduce - INPLACE(out)",
                    )
                )

    return test_cases


class OpTest(BaseOperatorTest):
    """ScatterReduce operator test with simplified implementation"""

    def __init__(self):
        super().__init__("ScatterReduce")

    def get_test_cases(self):
        return parse_test_cases()

    def torch_operator(self, *args, **kwargs):
        return torch.scatter_reduce(*args, **kwargs)

    # def infinicore_operator(self, *args, **kwargs):
    #     """InfiniCore implementation (operator not yet available)."""
    #     return infinicore.scatter_reduce(*args, **kwargs)


def main():
    """Main entry point"""
    runner = GenericTestRunner(OpTest)
    runner.run_and_exit()


if __name__ == "__main__":
    main()
