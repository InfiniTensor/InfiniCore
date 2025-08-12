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

class GatherTestCase(InfiniopTestCase):
    def __init__(
        self,
        input: torch.Tensor,
        dim: int,
        index: torch.Tensor,
        output: torch.Tensor,
        input_shape: List[int] | None,
        output_shape: List[int] | None,
        input_strides: List[int] | None,
        index_strides: List[int] | None,
        output_strides: List[int] | None,
    ):
        super().__init__("gather")
        self.input = input
        self.dim = dim
        self.index = index
        self.output = output
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.input_strides = input_strides 
        self.index_strides = index_strides
        self.output_strides = output_strides

    def write_test(self, test_writer: "InfiniopTestWriter"):
        super().write_test(test_writer)

        # 写入输入张量形状和步长信息
        test_writer.add_array(test_writer.gguf_key("input.shape"), self.input_shape)
        if self.input_strides is not None:
            test_writer.add_array(test_writer.gguf_key("input_strides"), self.input_strides)
        # 写入输出的形状和步长信息
        if self.output_shape is not None:
            test_writer.add_array(test_writer.gguf_key("output.shape"), self.output_shape)
        test_writer.add_array(
            test_writer.gguf_key("output.strides"),
            gguf_strides(*self.output_strides if self.output_strides is not None else contiguous_gguf_strides(self.output_shape))
        )

        # 写入索引张量的形状和步长信息
        test_writer.add_array(test_writer.gguf_key("index.shape"), self.index.shape)    
        if self.index_strides is not None:
            test_writer.add_array(test_writer.gguf_key("index_strides"), self.index_strides)
        
        # 写入维度信息
        test_writer.add_int32(test_writer.gguf_key("dim"), self.dim)

        # 写入输入张量和索引
        input_numpy = self.input.detach().cpu().numpy()
        index_numpy = self.index.detach().cpu().numpy()
        output_numpy = self.output.detach().cpu().numpy()

        test_writer.add_tensor(
            test_writer.gguf_key("input"),
            input_numpy,
            raw_dtype=np_dtype_to_ggml(input_numpy.dtype),
        )
        
        test_writer.add_tensor(
            test_writer.gguf_key("index"),
            index_numpy,
            raw_dtype=np_dtype_to_ggml(index_numpy.dtype),
        )

        test_writer.add_tensor(
            test_writer.gguf_key("output"),
            output_numpy,
            raw_dtype=np_dtype_to_ggml(output_numpy.dtype),
        )

        
        # 计算并写入输出
        ans = torch.gather(self.input, self.dim, self.index)
        ans_numpy = ans.detach().cpu().numpy()
        test_writer.add_tensor(
            test_writer.gguf_key("ans"),
            ans_numpy,
            raw_dtype=np_dtype_to_ggml(ans_numpy.dtype),
        )


def gen_gguf(dtype: torch.dtype, filename: str):
    test_writer = InfiniopTestWriter(filename)
    test_cases = []

    # 测试用例配置 
    # (input_shape, dim, index_shape, output_shape, input_strides, index_strides, output_strides)
    _TEST_CASES_ = [
        ((2, 2), 1, (2, 2), (2, 2), None, None, None),
        ((3, 3), 0, (3, 3), (3, 3), None, None, None),
        ((3, 3), 1, (3, 3), (3, 3), None, None, None),
        ((4, 3), 1, (2, 2), (2, 2), None, None, None),
        ((2, 3, 4), 0, (2, 3, 4), (2, 3, 4), None, None, None),
        ((2, 3, 4), 1, (2, 3, 4), (2, 3, 4), None, None, None),
        ((2, 3, 4), 2, (2, 3, 4), (2, 3, 4), None, None, None),
        # 非标准步长
        ((2, 2), 1, (2, 2), (2, 2), [4, 1], [2, 1], [2, 1]),
        ((3, 3), 0, (3, 3), (3, 3), [6, 1], [3, 1], [3, 1]),
        ((4, 3), 1, (2, 2), (2, 2), [6, 2], [2, 1], [2, 1]),
        ((2, 3, 4), 2, (2, 3, 4), (2, 3, 4), [24, 4, 1], [12, 4, 1], [12, 4, 1]),
        ((2, 2), 1, (2, 2), (2, 2), [4, 1], [2, 1], [2, 1]),
    ]
    
    for input_shape, dim, index_shape, output_shape, input_strides, index_strides, output_strides in _TEST_CASES_:
        # 生成输入张量
        input_tensor = random_tensor(input_shape, dtype)
        output = torch.empty(output_shape, dtype=dtype)
        # 生成index，范围为[0, input_shape[dim])
        index = random_tensor(index_shape, torch.int64, 0, input_shape[dim])

        test_case = GatherTestCase(
            input=input_tensor,
            dim=dim,
            index=index,
            output=output,
            input_shape=input_shape,
            output_shape=output_shape,
            input_strides=input_strides,
            index_strides=index_strides,
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
        bfloat16: "gather_bf16.gguf",
        torch.float16: "gather_f16.gguf",
        torch.float32: "gather_f32.gguf",
        torch.float64: "gather_f64.gguf",
        torch.int8: "gather_i8.gguf",
        torch.int16: "gather_i16.gguf",
        torch.int32: "gather_i32.gguf",
        torch.int64: "gather_i64.gguf",
    }

    # 生成测试用例
    for dtype in _TENSOR_DTYPES_:
        filename = dtype_filename_map[dtype]
        gen_gguf(dtype, filename)