import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import infinicore
from framework.base import BaseOperatorTest, TensorSpec, TestCase
from framework.runner import GenericTestRunner
from framework.datatypes import to_torch_dtype

# Test cases format: (shape, fill_value, dtype)
_TEST_CASES_DATA = [
    ((3, 4), 0.0, infinicore.float32),
    ((6, 2), 1.5, infinicore.float16),
    ((5, 5), -2.0, infinicore.float32),
    ((1, 7), 3.14, infinicore.bfloat16),
    ((8, 3), 42.0, infinicore.float32),
    ((2, 2, 2), 0.5, infinicore.float16),
]

_TOLERANCE_MAP = {infinicore.float32: {"atol": 1e-5, "rtol": 1e-4}}
_TENSOR_DTYPES = [infinicore.float32, infinicore.float16, infinicore.bfloat16]


def parse_test_cases():
    test_cases = []
    for shape, val, dtype in _TEST_CASES_DATA:
        out_spec = TensorSpec.from_tensor(shape, None, dtype)
        # kwargs = {"fill_value": val, "size": shape, "dtype": to_torch_dtype(dtype)}
        kwargs = {"fill_value": val, "size": shape, "dtype": dtype}

        test_cases.append(
            TestCase(
                inputs=[],
                kwargs=kwargs,
                output_spec=out_spec,
                comparison_target="out",
                tolerance=_TOLERANCE_MAP.get(dtype, {"atol": 1e-5, "rtol": 1e-4}),
                description=f"full - OUT_OF_PLACE",
            )
        )

    return test_cases


class OpTest(BaseOperatorTest):
    """Full operator test with simplified implementation"""

    def __init__(self):
        super().__init__("Full")

    def get_test_cases(self):
        return parse_test_cases()

    def torch_operator(self, *args, **kwargs):
        if "fill_value" not in kwargs:
            raise TypeError("full test did not provide 'fill_value' parameter")
        fill_value = kwargs.pop("fill_value")

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
            return torch.full(tuple(size), fill_value, dtype=dtype_torch, out=out)
        else:
            return torch.full(tuple(size), fill_value, dtype=dtype_torch)

    # def infinicore_operator(self, *args, **kwargs):
    #     """InfiniCore implementation (operator not yet available)."""
    #     return infinicore.full(*args, **kwargs)


def main():
    """Main entry point"""
    runner = GenericTestRunner(OpTest)
    runner.run_and_exit()


if __name__ == "__main__":
    main()
