import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import infinicore
from framework.base import BaseOperatorTest, TensorSpec, TestCase
from framework.datatypes import to_torch_dtype
from framework.runner import GenericTestRunner

# Test cases format: (shape, dtype)
# Note: infinicore.empty returns uninitialized memory. Tests will compare shape and dtype via output_spec
_TEST_CASES_DATA = [
    ((3, 4), infinicore.float32),
    ((6, 2), infinicore.float16),
    ((5, 5), infinicore.float32),
    ((1, 7), infinicore.bfloat16),
    ((8, 3), infinicore.float32),
    ((2, 2, 2), infinicore.float16),
]

_TOLERANCE_MAP = {infinicore.float32: {"atol": 0, "rtol": 0}}


def parse_test_cases():
    test_cases = []
    for shape, dtype in _TEST_CASES_DATA:
        out_spec = TensorSpec.from_tensor(shape, None, dtype)

        test_cases.append(
            TestCase(
                inputs=[],
                kwargs={"size": shape, "dtype": dtype},
                output_spec=out_spec,
                comparison_target="out",
                tolerance=_TOLERANCE_MAP.get(dtype, {"atol": 0, "rtol": 0}),
                description=f"empty - OUT_OF_PLACE",
            )
        )

    return test_cases


class OpTest(BaseOperatorTest):
    """Empty operator test with simplified implementation"""

    def __init__(self):
        super().__init__("Empty")

    def get_test_cases(self):
        return parse_test_cases()

    def torch_operator(self, *args, **kwargs):
        if "size" not in kwargs:
            raise TypeError("full test did not provide 'size' parameter")
        size = kwargs.pop("size")

        if "dtype" not in kwargs:
            raise TypeError("full test did not provide 'dtype' parameter")
        dtype_torch = to_torch_dtype(kwargs.pop("dtype"))
        if dtype_torch is None:
            raise TypeError("full test provided unsupported 'dtype' parameter")

        # 支持测试框架通过 kwargs 注入 out 参数
        out = kwargs.pop("out", None)

        if out is not None:
            return torch.empty(tuple(size), dtype=dtype_torch, out=out)
        else:
            return torch.empty(tuple(size), dtype=dtype_torch)
        # return infinicore.empty(*args, **kwargs)

    # def infinicore_operator(self, *args, **kwargs):
    #     """InfiniCore implementation (operator not yet available)."""
    #     return infinicore.empty(*args, **kwargs)


def main():
    """Main entry point"""
    runner = GenericTestRunner(OpTest)
    runner.run_and_exit()


if __name__ == "__main__":
    main()
