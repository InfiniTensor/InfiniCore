import torch
from ast import List
import numpy as np
import gguf
from typing import List

from .. import InfiniopTestWriter, InfiniopTestCase, np_dtype_to_ggml, gguf_strides, contiguous_gguf_strides

class LinearTestCase(InfiniopTestCase):
    def __init__(
        self,
        x: torch.Tensor,
        w: torch.Tensor,
        b: torch.Tensor | None,
        y: torch.Tensor,
        x_shape: List[int] | None,
        w_shape: List[int] | None,
        b_shape: List[int] | None, 
        y_shape: List[int] | None,
        x_strides: List[int] | None,
        w_strides: List[int] | None,
        b_strides: List[int] | None,
        y_strides: List[int] | None,
    ):
        super().__init__("linear")
        self.x = x
        self.w = w
        self.b = b
        self.y = y
        self.x_shape = x_shape
        self.w_shape = w_shape
        self.b_shape = b_shape
        self.y_shape = y_shape
        self.x_strides = x_strides or contiguous_gguf_strides(x.shape)
        self.w_strides = w_strides or contiguous_gguf_strides(w.shape)
        self.b_strides = b_strides if b is not None else None
        self.y_strides = y_strides or contiguous_gguf_strides(y_shape)

    def write_test(self, test_writer: "InfiniopTestWriter"):
        super().write_test(test_writer)

        # 写入输入张量 x
        test_writer.add_array(test_writer.gguf_key("x.shape"), self.x_shape)
        test_writer.add_array(test_writer.gguf_key("x.strides"), self.x_strides)
        x_numpy = self.x.detach().cpu().numpy()
        test_writer.add_tensor(test_writer.gguf_key("x"), x_numpy, raw_dtype=np_dtype_to_ggml(x_numpy.dtype))

        # 写入权重矩阵 w
        test_writer.add_array(test_writer.gguf_key("w.shape"), self.w_shape)
        test_writer.add_array(test_writer.gguf_key("w.strides"), self.w_strides)
        w_numpy = self.w.detach().cpu().numpy()
        test_writer.add_tensor(test_writer.gguf_key("w"), w_numpy, raw_dtype=np_dtype_to_ggml(w_numpy.dtype))

        # 写入偏置向量 b（如果存在）
        if self.b is not None:
            test_writer.add_array(test_writer.gguf_key("b.shape"), self.b_shape)
            if self.b_strides is not None:
                test_writer.add_array(test_writer.gguf_key("b.strides"), self.b_strides)
            b_numpy = self.b.detach().cpu().numpy()
            test_writer.add_tensor(test_writer.gguf_key("b"), b_numpy, raw_dtype=np_dtype_to_ggml(b_numpy.dtype))
        
        # 写入输出张量 y
        test_writer.add_array(test_writer.gguf_key("y.shape"), self.y_shape)
        test_writer.add_array(
            test_writer.gguf_key("y.strides"),
            gguf_strides(*self.y_strides if self.y_strides is not None else contiguous_gguf_strides(self.y_shape))
        )
        y_numpy = self.y.detach().cpu().numpy()
        test_writer.add_tensor(
            test_writer.gguf_key("y"), y_numpy, raw_dtype=np_dtype_to_ggml(y_numpy.dtype)
        )

        # 计算预期结果
        # 转换为FP64进行高精度计算
        x_fp64 = self.x.to(torch.float64)
        w_fp64 = self.w.to(torch.float64)
        b_fp64 = self.b.to(torch.float64) if self.b is not None else None
        
        # 使用FP64计算预期结果
        ans = torch.nn.functional.linear(x_fp64, w_fp64, b_fp64)
        ans_numpy = y.detach().cpu().numpy()

        # 写入预期结果
        test_writer.add_tensor(test_writer.gguf_key("ans"), ans_numpy, raw_dtype=gguf.GGMLQuantizationType.F64)


if __name__ == "__main__":
    test_writer = InfiniopTestWriter("linear.gguf")
    test_cases = []

    # 测试用例配置：(x_shape, w_shape, b_shape=None, y_shape, x_strides=None, w_strides=None, b_strides=None, y_strides)
    _TEST_CASES_ = [
        # 基本测试：2D输入，带偏置
        ((3, 4), (2, 4), (2,), (3, 2), None, None, None, None),
        ((5, 4), (3, 4), (3,), (5, 3), None, None, None, None),
        
        # 基本测试：2D输入，不带偏置
        ((3, 4), (2, 4), None, (3, 2), None, None, None, None),
        ((5, 4), (3, 4), None, (5, 3), None, None, None, None),
        
        # 多维输入测试：带偏置
        ((2, 3, 4), (5, 4), (5,), (2, 3, 5), None, None, None, None),
        ((1, 2, 3, 4), (6, 4), (6,), (1, 2, 3, 6), None, None, None, None),
        
        # 多维输入测试：不带偏置
        ((2, 3, 4), (5, 4), None, (2, 3, 5), None, None, None, None),
        ((1, 2, 3, 4), (6, 4), None, (1, 2, 3, 6), None, None, None, None),

        # 更大特征维度
        ((5, 10, 256), (512, 256), (512,), (5, 10, 512), None, None, None, None),  
        ((5, 10, 256), (512, 256), None, (5, 10, 512), None, None, None, None),  
        
        # 自定义步长测试
        ((4, 4), (3, 4), (3,), (4, 3), [8, 2], [6, 3], [3], None),
        ((2, 3, 4), (5, 4), (5,), (2, 3, 5), [24, 8, 2], [8, 2], [5], None),
    ]

    _TENSOR_DTYPES_ = [
        torch.float32,
        torch.float16,
        # torch.bfloat16,
    ]

    for dtype in _TENSOR_DTYPES_:
        for x_shape, w_shape, b_shape, y_shape, x_strides, w_strides, b_strides, y_strides in _TEST_CASES_:
            # 生成输入和权重
            x = torch.rand(*x_shape, dtype=dtype)
            w = torch.rand(*w_shape, dtype=dtype)
            
            # 生成偏置（如果需要）
            b = torch.rand(*b_shape, dtype=dtype) if b_shape is not None else None

            y = torch.empty(y_shape, dtype=dtype)
            
            test_case = LinearTestCase(
                x=x,
                w=w,
                b=b,
                y=y,
                x_shape=x_shape, 
                w_shape=w_shape,
                b_shape=b_shape,
                y_shape=y_shape,
                x_strides=x_strides,
                w_strides=w_strides,
                b_strides=b_strides,
                y_strides=y_strides,
            )
            test_cases.append(test_case)

    test_writer.add_tests(test_cases)
    test_writer.save()