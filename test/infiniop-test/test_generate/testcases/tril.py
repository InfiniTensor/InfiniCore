import torch
from ast import List
import numpy as np
import gguf
from typing import List

from .. import InfiniopTestWriter, InfiniopTestCase, np_dtype_to_ggml, gguf_strides, contiguous_gguf_strides

def random_tensor(
    shape: List[int],
    dtype: torch.dtype,
    low: int | None = None,
    high: int | None = None
) -> torch.Tensor:
    """
    仅支持 dtype 为 BF16, F16, F32, F64, I8, I16, I32, I64 的随机张量生成。
    对于整数类型，默认范围[-100, 100)。
    对于浮点类型，默认范围[-5, 5)。
    可通过 low、high 参数自定义范围。
    其他类型将报错不支持。
    """
    if dtype in (bfloat16, torch.float16, torch.float32, torch.float64, torch.bfloat16):
        l = -5.0 if low is None else float(low)
        h = 5.0 if high is None else float(high)
        return (h - l) * torch.rand(*shape, dtype=dtype) + l
    elif dtype in (torch.int8, torch.int16, torch.int32, torch.int64):
        l = -100 if low is None else low
        h = 100 if high is None else high
        return torch.randint(l, h, shape, dtype=dtype)
    else:
        raise ValueError(f"Unsupported dtype: {dtype}. Only BF16, F16, F32, F64, I8, I16, I32, I64 are supported.")


class TrilTestCase(InfiniopTestCase):
    def __init__(
        self,
        input: torch.Tensor,
        output: torch.Tensor,
        diagonal: int | None,
        shape: List[int] | None,
        input_strides: List[int] | None,
        output_strides: List[int] | None,
    ):
        super().__init__("tril")
        self.input = input
        self.output = output
        self.diagonal = diagonal
        self.shape = shape
        self.input_strides = input_strides 
        self.output_strides = output_strides

    def write_test(self, test_writer: "InfiniopTestWriter"):
        super().write_test(test_writer)
        
        # 写入diagonal的值
        test_writer.add_int32(test_writer.gguf_key("diagonal"), self.diagonal)
        
        # 写入输入张量的形状和步长和数据
        test_writer.add_array(test_writer.gguf_key("input.shape"), self.shape)
        if self.input_strides is not None:
            test_writer.add_array(test_writer.gguf_key("input.strides"), self.input_strides)
        input_numpy = self.input.detach().cpu().numpy()
        test_writer.add_tensor(test_writer.gguf_key("input"), input_numpy, raw_dtype=np_dtype_to_ggml(input_numpy.dtype),)

        # 写入输出张量的形状和步长和数据
        test_writer.add_array(test_writer.gguf_key("output.shape"), self.shape)
        test_writer.add_array(
            test_writer.gguf_key("output.strides"),
            gguf_strides(*self.output_strides if self.output_strides is not None else contiguous_gguf_strides(self.shape))
        )        
        output_numpy = self.output.detach().cpu().numpy()
        test_writer.add_tensor(
            test_writer.gguf_key("output"), output_numpy, raw_dtype=np_dtype_to_ggml(output_numpy.dtype)
        )

        # 计算预期结果
        ans = torch.tril(self.input, diagonal=self.diagonal)
        ans_numpy = ans.detach().cpu().numpy()
        
        # 写入预期结果
        test_writer.add_tensor(
            test_writer.gguf_key("ans"),
            ans_numpy,
            raw_dtype=np_dtype_to_ggml(ans_numpy.dtype),
        )


def gen_gguf(dtype: torch.dtype, filename: str):
    test_writer = InfiniopTestWriter(filename)
    test_cases = []

    # 测试用例配置：(shape, diagonal, input_strides=None, output_strides=None)
    _TEST_CASES_ = [
        # 基本二维测试
        ((3, 3), 0, None, None),  # 主对角线
        ((3, 3), 1, None, None),  # 主对角线+上一条对角线
        ((3, 3), -1, None, None),  # 主对角线-下一条对角线
        
        # 非方阵测试
        ((4, 2), 0, None, None),
        ((2, 4), 0, None, None),
        ((4, 2), 1, None, None),
        ((2, 4), -1, None, None),
        
        # 高维特征测试（特征维度较大的2D张量）
        ((64, 64), 0, None, None),  # 大方阵
        ((128, 64), 0, None, None),  # 高宽比大的矩阵
        ((64, 128), 0, None, None),  # 宽高比大的矩阵
    ]

    for shape, diagonal, input_strides, output_strides in _TEST_CASES_:
        input = random_tensor(shape, dtype)
        output = torch.empty(shape, dtype=dtype)
        
        test_case = TrilTestCase(
            input=input,
            output=output,
            diagonal=diagonal,
            shape=shape,
            input_strides=input_strides,
            output_strides=output_strides,  
        )
        test_cases.append(test_case)

    test_writer.add_tests(test_cases)
    test_writer.save()

if __name__ == "__main__":
    _TENSOR_DTYPES_ = [
        bfloat16,
        torch.float32,
        torch.float64,
        torch.float16,
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
    ]

    dtype_filename_map = {
        bfloat16: "tril_bf16.gguf",
        torch.float32: "tril_f32.gguf",
        torch.float64: "tril_f64.gguf",
        torch.float16: "tril_f16.gguf",
        torch.int8: "tril_i8.gguf",
        torch.int16: "tril_i16.gguf",
        torch.int32: "tril_i32.gguf",
        torch.int64: "tril_i64.gguf",
    }
    
    # 生成测试用例
    for dtype in _TENSOR_DTYPES_:
        filename = dtype_filename_map[dtype]
        gen_gguf(dtype, filename)