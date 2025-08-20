import torch
from ast import List
import numpy as np
import gguf
from typing import List
from ml_dtypes import bfloat16

from .. import InfiniopTestWriter, InfiniopTestCase, np_dtype_to_ggml, gguf_strides, contiguous_gguf_strides, tensor_to_numpy

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

class ScatterTestCase(InfiniopTestCase):
    def __init__(
        self,
        output: torch.Tensor,
        index: torch.Tensor,
        input: torch.Tensor,
        dim: int,
        output_shape: List[int] | None,
        input_shape: List[int] | None,
        index_shape: List[int] | None,
        output_strides: List[int] | None,
        input_strides: List[int] | None,
        index_strides: List[int] | None,
    ):
        super().__init__("scatter_")
        self.output = output  
        self.index = index
        self.input = input
        self.dim = dim
        self.output_shape = output_shape
        self.input_shape = input_shape
        self.index_shape = index_shape
        self.output_strides = output_strides 
        self.input_strides = input_strides 
        self.index_strides = index_strides

    def write_test(self, test_writer: "InfiniopTestWriter"):
        super().write_test(test_writer)

        # 写入维度信息
        test_writer.add_int32(test_writer.gguf_key("dim"), self.dim)

        # 写入输出张量形状和步长和数据
        test_writer.add_array(test_writer.gguf_key("output.shape"), self.output_shape)
        if self.output_strides is not None:
            test_writer.add_array(test_writer.gguf_key("output.strides"), gguf_strides(*self.output_strides))
        # output_numpy = self.output.detach().cpu().numpy()
        output_numpy = tensor_to_numpy(self.output)
        test_writer.add_tensor(
                    test_writer.gguf_key("output"), output_numpy, raw_dtype=np_dtype_to_ggml(output_numpy.dtype)
                )
        # 写入索引张量形状、步长和数据
        test_writer.add_array(test_writer.gguf_key("index.shape"), self.index_shape)
        if self.index_strides is not None:
            test_writer.add_array(test_writer.gguf_key("index.strides"), gguf_strides(*self.index_strides))
        # index_numpy = self.index.detach().cpu().numpy()
        index_numpy = tensor_to_numpy(self.index)
        test_writer.add_tensor(test_writer.gguf_key("index"), index_numpy, raw_dtype=np_dtype_to_ggml(index_numpy.dtype),)

        # 写入源张量形状、步长和数据
        test_writer.add_array(test_writer.gguf_key("input.shape"), self.input_shape)
        if self.input_strides is not None:
            test_writer.add_array(test_writer.gguf_key("input.strides"), gguf_strides(*self.input_strides))
        # input_numpy = self.input.detach().cpu().numpy()
        input_numpy = tensor_to_numpy(self.input)
        test_writer.add_tensor(test_writer.gguf_key("input"), input_numpy, raw_dtype=np_dtype_to_ggml(input_numpy.dtype),)

        # 计算预期结果（调用PyTorch的scatter_）
        ans = torch.zeros(self.output_shape, dtype=self.output.dtype)
        ans.scatter_(self.dim, self.index, self.input)
        # ans_numpy = ans.detach().cpu().numpy()
        ans_numpy = tensor_to_numpy(ans)
        test_writer.add_tensor(
            test_writer.gguf_key("ans"),
            ans_numpy,
            raw_dtype=np_dtype_to_ggml(ans_numpy.dtype),
        )


def gen_gguf(dtype: torch.dtype, filename: str):
    test_writer = InfiniopTestWriter(filename)
    test_cases = []

    # 测试用例格式：(output_shape, input_shape, dim, index_shape, input_strides=None, output_strides=None, index_strides=None)
    _TEST_CASES_ = [
        ((4, 4), (2, 4), 0, (2, 4), None, None, None),
        ((3, 4, 5), (3, 2, 5), 1, (3, 2, 5), [10, 5, 1], [20, 5, 1], None),
        ((1000, 1000), (100, 1000), 0, (100, 1000), None, None, None),
        ((3, 4), (2, 4), 0, (2, 4), [1, 2], None, None),
        ((2, 3, 4), (2, 3, 4), 2, (2, 3, 4), None, None, None),
        ((5, 5), (3, 5), 0, (3, 5), None, None, None),
    ]

    for output_shape, input_shape, dim, index_shape, input_strides, output_strides, index_strides in _TEST_CASES_:
        if dtype == bfloat16:  
            dtype = torch.bfloat16
        output = torch.empty(output_shape, dtype=dtype)
        input = random_tensor(input_shape, dtype)  # 使用随机张量生成src
        # 动态生成 index，范围为 [0, output_shape[dim])
        index = random_tensor(index_shape, torch.int64, 0, output_shape[dim])
        
        test_case = ScatterTestCase(
            output=output,
            index=index,
            input=input,
            dim=dim,
            output_shape=output_shape,
            input_shape=input_shape,
            index_shape=index_shape,
            output_strides=output_strides,
            input_strides=input_strides,
            index_strides=index_strides,
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
        bfloat16: "scatter_bf16.gguf",
        torch.float16: "scatter_f16.gguf",
        torch.float32: "scatter_f32.gguf",  
        torch.float64: "scatter_f64.gguf",
        torch.int8: "scatter_i8.gguf",
        torch.int16: "scatter_i16.gguf",
        torch.int32: "scatter_i32.gguf",
        torch.int64: "scatter_i64.gguf",
    }

    # 生成测试用例
    for dtype in _TENSOR_DTYPES_:
        filename = dtype_filename_map[dtype]
        gen_gguf(dtype, filename)