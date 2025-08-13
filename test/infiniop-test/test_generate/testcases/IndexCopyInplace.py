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

class IndexCopyTestCase(InfiniopTestCase):
    def __init__(
        self,
        input: torch.Tensor,
        output: torch.Tensor,
        index: torch.Tensor,
        dim: int,
        input_shape: List[int] | None,
        output_shape: List[int] | None,
        index_shape: List[int] | None,
        input_strides: List[int] | None,
        output_strides: List[int] | None,
        index_strides: List[int] | None,
    ):
        super().__init__("index_copy_inplace")
        self.input = input
        self.output = output
        self.index = index
        self.dim = dim
        self.input_shape = input_shape 
        self.output_shape = output_shape 
        self.index_shape = index_shape
        self.input_strides = input_strides 
        self.output_strides = output_strides 
        self.index_strides = index_strides

    def write_test(self, test_writer: "InfiniopTestWriter"):
        super().write_test(test_writer)

        # 写入维度信息
        test_writer.add_int32(test_writer.gguf_key("dim"), self.dim)
        # 写入索引信息
        index_numpy = tensor_to_numpy(self.index)
        # index_numpy = self.index.detach().cpu().numpy()
        test_writer.add_tensor(
            test_writer.gguf_key("index"),
            index_numpy,
            raw_dtype=np_dtype_to_ggml(index_numpy.dtype),
        )

        # 写入形状信息
        test_writer.add_array(test_writer.gguf_key("input.shape"), self.input_shape)
        test_writer.add_array(test_writer.gguf_key("output.shape"), self.output_shape)
        test_writer.add_array(test_writer.gguf_key("index.shape"), self.index_shape)

        # 写入strides信息
        test_writer.add_array(test_writer.gguf_key("input.strides"), gguf_strides(*self.input_strides))
        test_writer.add_array(test_writer.gguf_key("output.strides"), gguf_strides(*self.output_strides))
        test_writer.add_array(test_writer.gguf_key("index.strides"), gguf_strides(*self.index_strides))

        # 转换torch tensor为numpy用于写入文件
        # input_numpy = self.input.detach().cpu().numpy()
        # output_numpy = self.output.detach().cpu().numpy()
        input_numpy = tensor_to_numpy(self.input)
        output_numpy = tensor_to_numpy(self.output)

        # 写入张量数据
        test_writer.add_tensor(
            test_writer.gguf_key("input"),
            input_numpy,
            raw_dtype=np_dtype_to_ggml(input_numpy.dtype),
        )
        test_writer.add_tensor(
            test_writer.gguf_key("output"),
            output_numpy,
            raw_dtype=np_dtype_to_ggml(output_numpy.dtype),
        )

        # 计算并写入答案
        ans = torch.zeros(self.output_shape, dtype=self.output.dtype)
        ans.index_copy_(self.dim, self.index, self.input)
        # ans_numpy = ans.detach().cpu().numpy()
        ans_numpy = tensor_to_numpy(ans)
        test_writer.add_tensor(
            test_writer.gguf_key("ans"),
            ans_numpy,
            raw_dtype=np_dtype_to_ggml(input_numpy.dtype),
        )

def gen_gguf(dtype: torch.dtype, filename: str):
    test_writer = InfiniopTestWriter(filename)
    test_cases = []

    # (output_shape, input_shape, index_shape, dim, input_strides, output_strides, index_strides)
    _TEST_CASES_ = [
        ((5, 4), (3, 4), (3,), 0, (4, 1), (4, 1), (1,)),
        ((4, 5), (4, 3), (3,), 1, (3, 1), (5, 1), (1,)),
        ((3, 4, 5), (2, 4, 5), (2,), 0, (20, 5, 1), (20, 5, 1), (1,)),
        ((3, 4, 5), (3, 2, 5), (2,), 1, (20, 5, 1), (20, 5, 1), (1,)),
        ((3, 4, 5), (3, 4, 2), (2,), 2, (20, 5, 1), (20, 5, 1), (1,)),
        ((1000, 800), (500, 800), (500,), 0, (800, 1), (800, 1), (1,)),
        ((800, 1000), (800, 500), (500,), 1, (1, 800), (1, 800), (1,)),
        ((100, 80, 60), (50, 80, 60), (50,), 0, (4800, 60, 1), (4800, 60, 1), (1,)),
        ((100, 80, 60), (100, 40, 60), (40,), 1, (4800, 60, 1), (4800, 60, 1), (1,)),
        ((100, 80, 60), (100, 80, 30), (30,), 2, (4800, 60, 1), (4800, 60, 1), (1,)),
        ((5, 3), (3, 3), (3,), 0, (3, 1), (6, 2), (1,)),
        ((6, 4), (3, 4), (3,), 0, (4, 1), (8, 2), (1,)),
        ((4, 6), (4, 3), (3,), 1, (1, 4), (2, 8), (1,)),
        ((3, 4, 5), (2, 4, 5), (2,), 0, [40, 10, 2], [20, 5, 1], (1,)),
    ]

    for output_shape, input_shape, index_shape, dim, input_strides, output_strides, index_strides in _TEST_CASES_:
        if dtype == bfloat16:  # 确保已经 import bfloat16
            dtype = torch.bfloat16
        # 生成输入张量
        input = random_tensor(input_shape, dtype)

        # 动态生成 index，范围是 [0, output_shape[dim])
        index = random_tensor(index_shape, torch.int64, 0, output_shape[dim])

        output = torch.empty(output_shape, dtype=dtype)
        
        test_case = IndexCopyTestCase(
            input=input,
            output=output,
            index=index,
            dim=dim,
            input_shape=input_shape,
            output_shape=output_shape,
            index_shape=index_shape,
            input_strides=input_strides,
            output_strides=output_strides,
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
        bfloat16: "IndexCopyInplace_bf16.gguf",
        torch.float32: "IndexCopyInplace_f32.gguf",
        torch.float64: "IndexCopyInplace_f64.gguf",
        torch.float16: "IndexCopyInplace_f16.gguf",
        torch.int8: "IndexCopyInplace_i8.gguf",
        torch.int16: "IndexCopyInplace_i16.gguf",
        torch.int32: "IndexCopyInplace_i32.gguf",
        torch.int64: "IndexCopyInplace_i64.gguf",
    }

    # 生成测试用例
    for dtype in _TENSOR_DTYPES_:
        filename = dtype_filename_map[dtype]
        gen_gguf(dtype, filename)