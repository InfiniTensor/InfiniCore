import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import infinicore
from framework import BaseOperatorTest, TensorSpec, TestCase, GenericTestRunner


def row_major_strides(shape):
    """生成行优先stride"""
    stride = 1
    strides = [1]
    for dim in reversed(shape[1:]):
        stride *= dim
        strides.insert(0, stride)
    return tuple(strides)


def column_major_strides(shape):
    """生成列优先stride"""
    stride = 1
    strides = [stride]
    for dim in shape[:-1]:
        stride *= dim
        strides.append(stride)
    return tuple(strides)


# Test cases: (shape, input_strides, output_strides)
_TEST_CASES_DATA = [
    # 2D转置
    ((100, 100), (1, 100), (100, 1)),
    ((2000, 2000), (1, 2000), (2000, 1)),
    
    # 5D行列转置
    ((3, 4, 7, 53, 9), 
     row_major_strides((3, 4, 7, 53, 9)),
     column_major_strides((3, 4, 7, 53, 9))),
    
    # 6D行列转置 (主要优化目标)
    ((3, 4, 50, 50, 5, 7),
     row_major_strides((3, 4, 50, 50, 5, 7)),
     column_major_strides((3, 4, 50, 50, 5, 7))),
]

_TOLERANCE_MAP = {
    infinicore.float16: {"atol": 0, "rtol": 0},
    infinicore.float32: {"atol": 0, "rtol": 0},
}

_TENSOR_DTYPES = [infinicore.float16, infinicore.float32]


def parse_test_cases():
    test_cases = []
    for data in _TEST_CASES_DATA:
        shape, in_strides, out_strides = data
        
        for dtype in _TENSOR_DTYPES:
            tol = _TOLERANCE_MAP.get(dtype, {"atol": 0, "rtol": 0})
            
            # 输入tensor规格
            in_spec = TensorSpec.from_tensor(shape, in_strides, dtype)
            # 输出tensor规格：预先创建一个具有目标 strides 的 out，避免每次 iteration 分配
            out_spec = TensorSpec.from_tensor(shape, out_strides, dtype)
            
            test_cases.append(
                TestCase(
                    inputs=[in_spec],
                    kwargs={},  # out 由框架根据 output_spec 自动创建并传入 operator
                    output_spec=out_spec,
                    comparison_target="out",  # in-place(out) benchmark：只测 copy_ 内核
                    tolerance=tol,
                    description=f"rearrange {shape} {dtype}",
                )
            )
    
    return test_cases


class OpTest(BaseOperatorTest):
    """Rearrange operator test - stride重排操作"""
    
    def __init__(self):
        super().__init__("Rearrange")
    
    def get_test_cases(self):
        return parse_test_cases()
    
    def torch_operator(self, input_tensor, out):
        """PyTorch实现：out 已是目标 strides（由 output_spec 创建）"""
        out.copy_(input_tensor)
        return out
    
    def infinicore_operator(self, input_tensor, out):
        """InfiniCore实现：out 已是目标 strides（由 output_spec 创建）"""
        out.copy_(input_tensor)
        return out


def main():
    """Main entry point"""
    runner = GenericTestRunner(OpTest)
    runner.run_and_exit()


if __name__ == "__main__":
    main()

