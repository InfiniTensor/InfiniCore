# elu_inplace.py - 专门测试 inplace=True
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import torch.nn.functional as F
import infinicore
from framework.base import BaseOperatorTest, TensorSpec, TestCase
from framework.runner import GenericTestRunner

# 只测试 inplace=True 的情况
_INPLACE_TEST_CASES_DATA = [
    (TestCase.IN_PLACE_0, (13, 4), None, None, None, {"inplace": True}),
    (TestCase.IN_PLACE_0, (13, 4), (10, 1), None, (10, 1), {"inplace": True}),
    (TestCase.IN_PLACE_0, (13, 4, 4), None, None, None, {"inplace": True}),
    (TestCase.IN_PLACE_0, (16, 5632), None, None, None, {"inplace": True}),
    (TestCase.IN_PLACE_0, (16, 5632), (13312, 1), None, (13312, 1), {"inplace": True}),
]


def parse_test_cases(data):
    operation_mode = data[0]
    shape = data[1]
    a_strides = data[2] if len(data) > 2 else None
    c_strides = data[4] if len(data) > 4 else None
    kwargs = data[5] if len(data) > 5 else {}

    inputs = []
    if a_strides is not None:
        inputs.append(TensorSpec.from_strided_tensor(shape, a_strides))
    else:
        inputs.append(TensorSpec.from_tensor(shape))

    # 对于 inplace 操作，输出就是输入本身（被修改后）
    if c_strides is not None:
        output = TensorSpec.from_strided_tensor(shape, c_strides)
    else:
        output = TensorSpec.from_tensor(shape)

    return TestCase(operation_mode, inputs, output, **kwargs)


_TEST_CASES = [parse_test_cases(d) for d in _INPLACE_TEST_CASES_DATA]

_TENSOR_DTYPES = [infinicore.float16, infinicore.bfloat16, infinicore.float32]

_TOLERANCE_MAP = {
    infinicore.float16: {"atol": 0, "rtol": 1e-2},
    infinicore.float32: {"atol": 0, "rtol": 1e-3},
    infinicore.bfloat16: {"atol": 0, "rtol": 5e-2},
}


class OpTest(BaseOperatorTest):
    def __init__(self):
        super().__init__("Elu_Inplace")

    def get_test_cases(self):
        return _TEST_CASES

    def get_tensor_dtypes(self):
        return _TENSOR_DTYPES

    def get_tolerance_map(self):
        return _TOLERANCE_MAP

    def torch_operator(self, a, out=None, **kwargs):
        # 强制使用 inplace=True
        return F.elu(a, inplace=True)

    # def infinicore_operator(self, a, out=None, **kwargs):
    #     # 强制使用 inplace=True
    #     return infinicore.elu(a, inplace=True)


def main():
    runner = GenericTestRunner(OpTest)
    runner.run_and_exit()


if __name__ == "__main__":
    main()
